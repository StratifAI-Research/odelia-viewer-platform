"""
Breast Cancer Model Service - Orchestrates the entire inference pipeline
Single Responsibility: Model orchestration and inference
"""
import logging
import torch
from pathlib import Path
from monai.networks import nets

from shared.timing_utils import time_operation
from shared.config import StorageConfig

from config import BreastCancerConfig
from exceptions import ModelNotLoadedError, InferenceError
from dicom_converter import convert_to_unilateral_nifti
from preprocessing import get_preprocessing_pipeline, preprocess_for_side
from response_builder import build_bilateral_classification
from retrieval_strategy import RetrievalStrategy, WadoRSRetrieval

logger = logging.getLogger(__name__)


class BreastCancerModelService:
    """Service for breast cancer model inference"""

    def __init__(self, bc_config: BreastCancerConfig, storage_config: StorageConfig):
        """
        Initialize breast cancer model service

        Args:
            bc_config: Breast cancer service configuration
            storage_config: Storage configuration
        """
        self.bc_config = bc_config
        self.storage_config = storage_config
        self.model = None

    def initialize_model(self) -> None:
        """Load model on startup"""
        try:
            logger.info("=" * 60)
            logger.info("Breast Cancer Classification Service - Initializing")
            logger.info("=" * 60)

            logger.info(f"Loading ResNet model on device: {self.bc_config.device}")

            # Load ResNet model
            self.model = nets.ResNet(
                "basic",
                [2, 2, 2, 2],
                [64, 128, 256, 512],
                n_input_channels=2,
                num_classes=1
            )

            # Uncomment when checkpoint is available
            # checkpoint = torch.load(self.bc_config.model_path, map_location=torch.device(self.bc_config.device))
            # self.model.load_state_dict(checkpoint)

            self.model.to(self.bc_config.device)
            self.model.eval()

            logger.info(f"Model loaded successfully on {self.bc_config.device}")
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
        Analyze MRI series using breast cancer classification model

        Args:
            request_data: Request dictionary with:
                - wado_rs_retrieval: List of WADO-RS retrieval info

        Returns:
            Analysis result dictionary with bilateral classification

        Raises:
            ModelNotLoadedError: If model is not loaded
            InferenceError: If analysis fails
        """
        if self.model is None:
            raise ModelNotLoadedError("Model not loaded")

        try:
            # Step 1: Retrieve DICOM data using appropriate strategy
            with time_operation("retrieve_dicom", logger):
                retrieval_strategy = self._create_retrieval_strategy(request_data)
                dicom_folder, series_uid = retrieval_strategy.retrieve()

            # Step 2: Convert DICOM to unilateral NIfTI
            with time_operation("dicom_to_nifti_conversion", logger):
                nifties = convert_to_unilateral_nifti(dicom_folder)

            # Step 3: Prepare preprocessing
            with time_operation("prepare_preprocessing", logger):
                transform = get_preprocessing_pipeline()

            # Step 4: Process each side
            results = {}
            for side in ["left", "right"]:
                logger.info(f"Processing {side} breast...")

                try:
                    side_result = self._process_side(side, nifties, transform)
                    results[side] = side_result
                except Exception as e:
                    logger.error(f"Error processing {side} side: {e}")
                    results[side] = {"error": f"Processing error for {side} side: {str(e)}"}

            # Step 5: Build response
            response = build_bilateral_classification(results["left"], results["right"])

            return response

        except Exception as e:
            logger.error(f"Error during MRI analysis: {e}")
            import traceback
            traceback.print_exc()
            raise InferenceError(f"Analysis failed: {str(e)}") from e

    def _create_retrieval_strategy(self, request_data: dict) -> RetrievalStrategy:
        """
        Create appropriate retrieval strategy based on request format

        Args:
            request_data: Request dictionary

        Returns:
            RetrievalStrategy instance
        """
        wado_rs_retrieval = request_data.get("wado_rs_retrieval")

        if not wado_rs_retrieval:
            raise ValueError("Missing required field 'wado_rs_retrieval'. Legacy 'seriesInstanceUID' format is no longer supported.")

        return WadoRSRetrieval(
            wado_rs_retrieval,
            self.storage_config
        )

    def _process_side(self, side: str, nifties: dict, transform) -> dict:
        """
        Process one side (left or right) breast

        Args:
            side: "left" or "right"
            nifties: Dictionary of NIfTI images
            transform: Preprocessing transform pipeline

        Returns:
            Classification result for this side
        """
        pre_key = f"Pre_{side}"
        post_key = f"Post_1_{side}"

        # Check if required images exist
        if pre_key not in nifties:
            return {"error": f"Missing Pre contrast image for {side} side"}

        if post_key not in nifties:
            return {"error": f"Missing Post contrast image for {side} side"}

        pre = nifties[pre_key]
        post = nifties[post_key]
        logger.info(f"  {side}: Pre shape={pre.data.shape}, Post shape={post.data.shape}")

        # Preprocess
        with time_operation(f"{side}_preprocessing", logger):
            model_input = preprocess_for_side(pre, post, transform, self.bc_config.device)

        # Run inference
        with time_operation(f"{side}_inference", logger):
            with torch.inference_mode():
                prob = torch.sigmoid(self.model(model_input)).item()

        logger.info(f"  {side}: Model output probability={prob:.4f}")

        result = {
            "prediction": "Cancerous" if prob > 0.5 else "Not Cancerous",
            "confidence": round((prob if prob > 0.5 else 1 - prob) * 100, 2)
        }

        logger.info(f"  {side}: Result={result}")
        return result

    def get_health_status(self) -> dict:
        """
        Get service health status

        Returns:
            Dictionary with health information
        """
        return {
            "status": "healthy",
            "model_loaded": self.model is not None,
            "device": self.bc_config.device
        }
