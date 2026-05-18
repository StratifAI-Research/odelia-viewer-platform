"""
MedGemma Model Service - Orchestrates the entire inference pipeline
Single Responsibility: Model orchestration and inference
"""
import logging
import torch
from pathlib import Path
from typing import Optional

from shared.timing_utils import time_operation
from shared.config import StorageConfig

from config import MedGemmaConfig
from exceptions import ModelNotLoadedError, ModelAuthenticationError, InferenceError, ResponseParsingError
from preprocessing import extract_central_slices
from prompt_templates import build_messages
from response_parser import parse_bilateral_response
from response_builder import build_bilateral_response
from retrieval_strategy import RetrievalStrategy, WadoRSRetrieval

logger = logging.getLogger(__name__)


class MedGemmaModelService:
    """Service for MedGemma MRI inference"""

    def __init__(self, config: MedGemmaConfig, storage_config: StorageConfig):
        """
        Initialize MedGemma model service

        Args:
            config: MedGemma service configuration
            storage_config: Storage configuration
        """
        self.config = config
        self.storage_config = storage_config
        self.model = None
        self.processor = None
        self.model_info = {
            "model_name": "MedGemma",
            "architecture": "Vision-Language Model",
            "version": "1.5-4b-it"
        }

    def initialize_model(self) -> None:
        """Download and load model on startup"""
        try:
            logger.info("=" * 60)
            logger.info("MedGemma MRI Classification Service - Initializing")
            logger.info("=" * 60)

            # Import transformers here to fail fast if not available
            from transformers import AutoProcessor, AutoModelForImageTextToText

            logger.info(f"Loading MedGemma model: {self.config.model_id}")
            logger.info(f"Device: {self.config.device}")
            logger.info(f"Dtype: {self.config.torch_dtype}")

            # Check for HF token
            if not self.config.hf_token:
                raise ModelAuthenticationError(
                    "HF_TOKEN environment variable not set. "
                    "MedGemma is a gated model - you must accept the license at "
                    "https://huggingface.co/google/medgemma-1.5-4b-it and provide a valid token."
                )

            # Determine torch dtype
            if self.config.torch_dtype == "bfloat16":
                torch_dtype = torch.bfloat16
            elif self.config.torch_dtype == "float16":
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32

            # Load processor
            logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                self.config.model_id,
                token=self.config.hf_token
            )

            # Load model
            logger.info("Loading model (this may take a while)...")
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.config.model_id,
                token=self.config.hf_token,
                torch_dtype=torch_dtype,
                device_map="auto" if self.config.device == "cuda" else None
            )

            # Move to device if not using device_map
            if self.config.device != "cuda":
                self.model = self.model.to(self.config.device)

            self.model.eval()

            logger.info(f"Model loaded successfully")
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
        Analyze MRI series using MedGemma model

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

            # Step 2: Extract central slices as PIL images
            with time_operation("extract_slices", logger):
                slices = extract_central_slices(dicom_folder, self.config.num_slices)
                logger.info(f"Extracted {len(slices)} slices for analysis")

            # Step 3: Build message content with interleaved images/text
            with time_operation("build_prompt", logger):
                messages = build_messages(slices)

            # Step 4: Run inference
            with time_operation("model_inference", logger):
                generated_text = self._run_inference(messages)

            # Log full raw output for debugging
            logger.info(f"Generated response length: {len(generated_text)} chars")
            logger.info(f"MedGemma raw output:\n{generated_text}")

            # Step 5: Parse response
            with time_operation("parse_response", logger):
                parsed_result = parse_bilateral_response(generated_text)

            # Step 6: Build final response
            response = build_bilateral_response(parsed_result, self.model_info)

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
            raise ValueError("Missing required field 'wado_rs_retrieval'")

        return WadoRSRetrieval(
            wado_rs_retrieval,
            self.storage_config
        )

    def _run_inference(self, messages: list) -> str:
        """
        Run MedGemma inference with granular timing

        Args:
            messages: List of message dicts with interleaved content

        Returns:
            Generated text response
        """
        # Step 1: Apply chat template and prepare inputs (tokenization)
        with time_operation("tokenize_input", logger):
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )
            # Move inputs to device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[1]
            logger.info(f"Input token count: {input_len}")

        # Step 2: Generate response (model inference)
        with time_operation("model_generate", logger):
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False  # Deterministic for classification
                )
            output_len = outputs.shape[1]
            logger.info(f"Output token count: {output_len} (generated: {output_len - input_len})")

        # Step 3: Decode output (detokenization)
        with time_operation("decode_output", logger):
            generated_tokens = outputs[0][input_len:]
            generated_text = self.processor.decode(generated_tokens, skip_special_tokens=True)

        return generated_text

    def get_health_status(self) -> dict:
        """
        Get service health status

        Returns:
            Dictionary with health information
        """
        return {
            "status": "healthy",
            "model_loaded": self.model is not None,
            "device": self.config.device,
            "model_info": self.model_info if self.model_info else None
        }
