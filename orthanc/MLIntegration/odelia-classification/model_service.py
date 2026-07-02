"""
Generalized ODELIA model service (ODV-214) - orchestrates the inference pipeline.

Serves the single subunit baked into the image (one image = one model) and
preserves the MST service's initialize_model / analyze_* contract. The exact
single-channel preprocessing is delegated to a shared transform in ODV-217; a
minimal deterministic transform is used here so the block is runnable.
"""

import logging
import os
from pathlib import Path

import torch
from config import ModelServiceConfig
from dicom_converter import (
    compute_subtraction_nifti,
    convert_multiphase_to_subtraction_nifti,
    convert_series_to_nifti,
)
from exceptions import InferenceError, ModelNotLoadedError
from model_loader import build_model
from models import assert_forward_contract
from preprocessing import prepare_single_channel
from response_builder import build_classification_response
from retrieval_strategy import RetrievalStrategy, WadoRSRetrieval
from shared.config import StorageConfig
from shared.dicom_storage import PathContainmentError, resolve_within
from shared.timing_utils import time_operation

logger = logging.getLogger(__name__)


class ModelService:
    """Service for generalized model inference."""

    def __init__(self, config: ModelServiceConfig, storage_config: StorageConfig) -> None:
        """
        Initialize the model service.

        Args:
            config: Model service configuration
            storage_config: Storage configuration
        """
        self.config = config
        self.storage_config = storage_config
        self.model = None
        self.model_info = None

    def initialize_model(self) -> None:
        """Build the model on startup and run the pre-flight contract check.

        Weights are init-only here; strict-loading a baked state_dict is a later
        ticket. Pre-flight (a synthetic forward) fails loudly before the service
        accepts traffic if the model can't honor the inference contract.
        """
        try:
            logger.info("=" * 60)
            logger.info(f"ODELIA model service - Initializing (model={self.config.model_name})")
            logger.info("=" * 60)

            self.model, self.model_info = build_model(self.config.model_name)
            self._run_preflight()

            logger.info(f"Model '{self.config.model_name}' ready on {self.config.device}")
            logger.info("=" * 60)
            logger.info("Service ready to accept requests")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            import traceback

            traceback.print_exc()
            raise

    def _run_preflight(self) -> None:
        """Synthetic forward at startup: prove the tensor contract holds.

        Checks construction + the fixed inference contract (input -> (1, 3)
        logits) on the real device. Strict weight-loading is validated in a
        later ticket. ``SKIP_PREFLIGHT`` bypasses this for debugging only.
        """
        if os.getenv("SKIP_PREFLIGHT", "").strip():
            logger.warning("SKIP_PREFLIGHT set - skipping startup contract check")
            return
        shape = assert_forward_contract(self.model, self.config.model_name)
        logger.info("Pre-flight OK: %s forward -> %s", self.config.model_name, shape)

    def analyze_mri_series(self, request_data: dict) -> dict:
        """
        Analyze an MRI series with the served model.

        Dispatches to mode-specific pipelines based on input_configuration_id:
          - pre_post:    two series (pre + post contrast), compute subtraction
          - subtraction: single pre-computed subtraction volume
          - multiphase:  single multi-phase series, extract temporal groups & subtract

        Falls back to legacy flat retrieval when no configuration is specified.

        Args:
            request_data: Request dictionary with wado_rs_retrieval, and optionally
                          input_configuration_id and input_mapping

        Returns:
            Classification result dictionary (per-class probabilities + argmax)

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
        if config_id == "subtraction" and input_mapping:
            return self._analyze_subtraction(input_mapping)
        if config_id == "multiphase" and input_mapping:
            return self._analyze_multiphase(input_mapping)
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
            raise InferenceError(f"Pre+Post analysis failed: {e!s}") from e

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
            raise InferenceError(f"Subtraction analysis failed: {e!s}") from e

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
            raise InferenceError(f"Multi-phase analysis failed: {e!s}") from e

    def _analyze_flat(self, request_data: dict) -> dict:
        """Legacy fallback: flat single-series retrieval (no manifest / input mapping)."""
        try:
            with time_operation("retrieve_dicom", logger):
                retrieval_strategy = self._create_retrieval_strategy(request_data)
                dicom_folder, series_uid = retrieval_strategy.retrieve()

            # Defence-in-depth: keep the request-derived folder within the image
            # root before any filesystem use (ODV-203 path-injection barrier).
            dicom_folder = self._resolve_within(dicom_folder)

            with time_operation("dicom_to_nifti_conversion", logger):
                nifti_path = convert_series_to_nifti(dicom_folder)

            return self._infer_and_respond(nifti_path)

        except (ValueError, InferenceError):
            raise
        except Exception as e:
            logger.error(f"Error in flat analysis: {e}")
            import traceback

            traceback.print_exc()
            raise InferenceError(f"Analysis failed: {e!s}") from e

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _infer_and_respond(self, nifti_path: Path) -> dict:
        """Common tail shared by all modes: prepare -> infer -> build response.

        The single-channel transform is the ODV-217 seam: it is replaced there
        with the exact shared MediSwarm bilateral/unilateral preprocessing.
        """
        with time_operation("prepare_single_channel", logger):
            tensor = prepare_single_channel(nifti_path, self.config.device)

        with time_operation("model_inference", logger):
            probs = self._run_inference(tensor)

        logger.info(f"Inference complete: class probabilities = {probs}")
        return build_classification_response(probs, self.model_info)

    def _resolve_within(self, candidate: Path) -> Path:
        """
        Apply the ODV-203 path-injection barrier to a request-derived folder.

        A containment failure is mapped to ``InferenceError`` (not ``ValueError``)
        so it routes to the generic 500 handler instead of the client-facing 400 --
        the rejected path is logged server-side only and never surfaced to the
        client. This matches the breast/medgemma services, where the barrier is
        funnelled into a generic 500.
        """
        try:
            return resolve_within(self.storage_config.image_folder, candidate)
        except PathContainmentError as e:
            logger.warning("Path containment barrier rejected request-derived folder: %s", e)
            raise InferenceError("path containment check failed") from e

    def _retrieve_role(self, input_mapping: dict, role_key: str) -> tuple[Path, str]:
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

        wado_entry = [
            {
                "retrieval_url": info["wado_rs_url"],
                "study_uid": info.get("study_uid", ""),
                "series_uid": info["series_uid"],
            }
        ]
        dicom_folder, series_uid = WadoRSRetrieval(wado_entry, self.storage_config).retrieve()
        # Defence-in-depth: keep the request-derived folder within the image root
        # before any filesystem use (ODV-203 path-injection barrier).
        dicom_folder = self._resolve_within(dicom_folder)
        return dicom_folder, series_uid

    def _create_retrieval_strategy(self, request_data: dict) -> RetrievalStrategy:
        """Create retrieval strategy for legacy flat requests."""
        wado_rs_retrieval = request_data.get("wado_rs_retrieval")

        if not wado_rs_retrieval:
            raise ValueError(
                "Missing required field 'wado_rs_retrieval'. "
                "Legacy 'seriesInstanceUID' format is no longer supported."
            )

        return WadoRSRetrieval(wado_rs_retrieval, self.storage_config)

    def _run_inference(self, tensor: torch.Tensor) -> list[float]:
        """Run a classification forward pass and return per-class probabilities."""
        if self.model is None:
            raise ModelNotLoadedError("Model not loaded")

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)

        return probs.squeeze(0).cpu().tolist()

    def get_health_status(self) -> dict:
        """Get service health status."""
        return {
            "status": "healthy",
            "model_loaded": self.model is not None,
            "device": self.config.device,
            "model_info": self.model_info if self.model_info else None,
        }
