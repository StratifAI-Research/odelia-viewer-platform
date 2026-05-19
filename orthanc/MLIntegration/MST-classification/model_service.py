"""
MST Model Service - Orchestrates the entire inference pipeline
Single Responsibility: Model orchestration and inference
"""
import logging
import sys
import torch
from pathlib import Path
from typing import Optional, Tuple

from shared.timing_utils import time_operation
from shared.config import StorageConfig

from config import MSTConfig
from exceptions import ModelNotLoadedError, InferenceError
from model_loader import load_model as load_mst_model, download_model_files
from dicom_converter import convert_series_to_nifti, convert_multiphase_to_subtraction_nifti, compute_subtraction_nifti
from preprocessing import prepare_for_inference, generate_attention_overlays
from response_builder import build_bilateral_response
from retrieval_strategy import RetrievalStrategy, WadoRSRetrieval

logger = logging.getLogger(__name__)


class MSTModelService:
    """Service for MST model inference"""

    def __init__(self, mst_config: MSTConfig, storage_config: StorageConfig):
        """
        Initialize MST model service

        Args:
            mst_config: MST service configuration
            storage_config: Storage configuration
        """
        self.mst_config = mst_config
        self.storage_config = storage_config
        self.model = None
        self.predict_fn = None
        self.model_info = None

    def initialize_model(self) -> None:
        """Download and load model on startup"""
        try:
            logger.info("=" * 60)
            logger.info("MST Classification Service - Initializing")
            logger.info("=" * 60)

            # Download model files if not already present
            logger.info(f"Checking model files in {self.mst_config.model_path}")
            required_files = ["models.py", "predict_attention.py", "state_dict.pt", "model_config.json"]
            files_exist = all(
                (self.mst_config.model_path / f).exists()
                for f in required_files
            )

            if not files_exist:
                logger.info("Model files not found, downloading from HuggingFace...")
                download_model_files()
            else:
                logger.info("Model files already present, skipping download")

            # Load model
            logger.info(f"Loading model on device: {self.mst_config.device}")
            self.model, self.predict_fn, self.model_info = load_mst_model()

            # Move model to device
            if self.model is not None:
                self.model = self.model.to(self.mst_config.device)
                logger.info(f"Model loaded successfully on {self.mst_config.device}")
                logger.info(f"  Model: {self.model_info['model_name']}")
                logger.info(f"  Architecture: {self.model_info['architecture']}")

            logger.info("=" * 60)
            logger.info("Service ready to accept requests")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            import traceback
            traceback.print_exc()
            raise

    def analyze_mri_series(self, request_data: dict) -> dict:
        """
        Analyze MRI series using MST model.

        Dispatches to mode-specific pipelines based on input_configuration_id:
          - pre_post:    two series (pre + post contrast), compute subtraction
          - subtraction: single pre-computed subtraction volume
          - multiphase:  single multi-phase series, extract temporal groups & subtract

        Falls back to legacy flat retrieval when no configuration is specified.

        Args:
            request_data: Request dictionary with wado_rs_retrieval, and optionally
                          input_configuration_id and input_mapping

        Returns:
            Analysis result dictionary with bilateral classification and attention maps

        Raises:
            ModelNotLoadedError: If model is not loaded
            ValueError: If input mapping is invalid
            InferenceError: If analysis fails
        """
        if self.model is None:
            raise ModelNotLoadedError("Model not loaded")

        config_id = request_data.get("input_configuration_id")
        input_mapping = request_data.get("input_mapping")

        if config_id == "pre_post" and input_mapping:
            return self._analyze_pre_post(input_mapping)
        elif config_id == "subtraction" and input_mapping:
            return self._analyze_subtraction(input_mapping)
        elif config_id == "multiphase" and input_mapping:
            return self._analyze_multiphase(input_mapping)
        else:
            return self._analyze_flat(request_data)

    # ------------------------------------------------------------------
    # Mode-specific analysis pipelines
    # ------------------------------------------------------------------

    def _analyze_pre_post(self, input_mapping: dict) -> dict:
        """Pre + Post Contrast mode: retrieve two series, compute subtraction, run inference."""
        try:
            with time_operation("retrieve_pre_series", logger):
                pre_folder, _ = self._retrieve_role(input_mapping, "pre")

            with time_operation("retrieve_post_series", logger):
                post_folder, _ = self._retrieve_role(input_mapping, "post")

            with time_operation("convert_pre_to_nifti", logger):
                pre_nifti = convert_series_to_nifti(pre_folder)

            with time_operation("convert_post_to_nifti", logger):
                post_nifti = convert_series_to_nifti(post_folder)

            with time_operation("compute_subtraction", logger):
                sub_nifti = compute_subtraction_nifti(pre_nifti, post_nifti)

            return self._infer_and_respond(sub_nifti)

        except (ValueError, InferenceError):
            raise
        except Exception as e:
            logger.error(f"Error in pre_post analysis: {e}")
            import traceback
            traceback.print_exc()
            raise InferenceError(f"Pre+Post analysis failed: {str(e)}") from e

    def _analyze_subtraction(self, input_mapping: dict) -> dict:
        """Subtraction mode: retrieve pre-computed subtraction volume, run inference."""
        try:
            with time_operation("retrieve_subtraction_series", logger):
                dicom_folder, _ = self._retrieve_role(input_mapping, "sub")

            with time_operation("dicom_to_nifti_conversion", logger):
                nifti_path = convert_series_to_nifti(dicom_folder)

            return self._infer_and_respond(nifti_path)

        except (ValueError, InferenceError):
            raise
        except Exception as e:
            logger.error(f"Error in subtraction analysis: {e}")
            import traceback
            traceback.print_exc()
            raise InferenceError(f"Subtraction analysis failed: {str(e)}") from e

    def _analyze_multiphase(self, input_mapping: dict) -> dict:
        """Multi-phase mode: retrieve series, extract temporal groups, compute subtraction, run inference."""
        try:
            with time_operation("retrieve_multiphase_series", logger):
                dicom_folder, _ = self._retrieve_role(input_mapping, "multiphase")

            with time_operation("multiphase_to_subtraction_nifti", logger):
                nifti_path = convert_multiphase_to_subtraction_nifti(dicom_folder)

            return self._infer_and_respond(nifti_path)

        except (ValueError, InferenceError):
            raise
        except Exception as e:
            logger.error(f"Error in multiphase analysis: {e}")
            import traceback
            traceback.print_exc()
            raise InferenceError(f"Multi-phase analysis failed: {str(e)}") from e

    def _analyze_flat(self, request_data: dict) -> dict:
        """Legacy fallback: flat single-series retrieval (no manifest / input mapping)."""
        try:
            with time_operation("retrieve_dicom", logger):
                retrieval_strategy = self._create_retrieval_strategy(request_data)
                dicom_folder, series_uid = retrieval_strategy.retrieve()

            with time_operation("dicom_to_nifti_conversion", logger):
                nifti_path = convert_series_to_nifti(dicom_folder)

            return self._infer_and_respond(nifti_path)

        except (ValueError, InferenceError):
            raise
        except Exception as e:
            logger.error(f"Error in flat analysis: {e}")
            import traceback
            traceback.print_exc()
            raise InferenceError(f"Analysis failed: {str(e)}") from e

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _infer_and_respond(self, nifti_path: Path) -> dict:
        """Common tail shared by all modes: prepare -> infer -> build response."""
        with time_operation("load_nifti_as_torchio", logger):
            img = prepare_for_inference(nifti_path, self.mst_config.model_path)

        with time_operation("model_inference", logger):
            probs, weight = self._run_inference(img)

        logger.info("Inference complete")
        logger.info(f"  Left breast probabilities: {probs['left']}")
        logger.info(f"  Right breast probabilities: {probs['right']}")

        with time_operation("generate_attention_maps_total", logger):
            attention_maps = generate_attention_overlays(
                img.data,
                weight.data,
                self.mst_config.model_path
            )

        return build_bilateral_response(probs, attention_maps, self.model_info)

    def _retrieve_role(self, input_mapping: dict, role_key: str) -> Tuple[Path, str]:
        """
        Retrieve a single input role's DICOM series via WADO-RS.

        Args:
            input_mapping: Dict of role_key -> {series_uid, wado_rs_url}
            role_key: The role to retrieve (e.g. "pre", "post", "sub", "multiphase")

        Returns:
            Tuple of (dicom_folder_path, series_uid)

        Raises:
            ValueError: If the role is missing or has no WADO-RS URL
        """
        info = input_mapping.get(role_key)
        if not info or not info.get("wado_rs_url"):
            raise ValueError(f"Missing or invalid mapping for input '{role_key}'")

        wado_entry = [{
            "retrieval_url": info["wado_rs_url"],
            "study_uid": info.get("study_uid", ""),
            "series_uid": info["series_uid"]
        }]
        return WadoRSRetrieval(wado_entry, self.storage_config).retrieve()

    def _create_retrieval_strategy(self, request_data: dict) -> RetrievalStrategy:
        """Create retrieval strategy for legacy flat requests."""
        wado_rs_retrieval = request_data.get("wado_rs_retrieval")

        if not wado_rs_retrieval:
            raise ValueError(
                "Missing required field 'wado_rs_retrieval'. "
                "Legacy 'seriesInstanceUID' format is no longer supported."
            )

        return WadoRSRetrieval(wado_rs_retrieval, self.storage_config)

    def _run_inference(self, img) -> Tuple[dict, any]:
        """Run MST model inference on a TorchIO ScalarImage."""
        if str(self.mst_config.model_path) not in sys.path:
            sys.path.insert(0, str(self.mst_config.model_path))

        from predict_attention import run_prediction

        with torch.no_grad():
            probs, weight = run_prediction(img, self.model)

        return probs, weight

    def get_health_status(self) -> dict:
        """Get service health status."""
        return {
            "status": "healthy",
            "model_loaded": self.model is not None,
            "device": self.mst_config.device,
            "model_info": self.model_info if self.model_info else None
        }
