"""
Generalized ODELIA model service (ODV-214) - Flask microservice.
Thin HTTP layer delegating to model_service; the served model is the single
subunit baked into the image (one image = one model).
"""

import logging
import os
from pathlib import Path

from config import ModelServiceConfig
from exceptions import InferenceError, ModelNotLoadedError
from flask import Flask, jsonify, request
from flask_cors import CORS
from model_service import ModelService
from shared.config import StorageConfig
from shared.security_banner import print_security_banner

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global model service
model_service: ModelService = None


def initialize_service():
    """Initialize configurations and model service"""
    global model_service

    # Load configurations from environment
    config = ModelServiceConfig.from_env()

    storage_config = StorageConfig(
        image_folder=Path(os.getenv("IMAGE_FOLDER", "./images")), cleanup_on_start=True
    )

    # Create necessary directories
    Path(storage_config.image_folder).mkdir(parents=True, exist_ok=True)

    # Initialize model service (builds the model and runs the pre-flight check)
    model_service = ModelService(config, storage_config)
    model_service.initialize_model()


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify(model_service.get_health_status())


@app.route("/analyze/mri", methods=["POST"])
def analyze_mri():
    """
    Analyze MRI series using the served model.

    Supports three input modes via input_configuration_id:
      - pre_post:    Two series (pre + post contrast), subtraction computed
      - subtraction: Single pre-computed subtraction volume
      - multiphase:  Single multi-phase series, temporal subtraction computed

    Falls back to flat single-series retrieval when no configuration is specified.

    Input format (manifest-based):
    {
        "wado_rs_retrieval": [...],
        "study_uid": "1.2.3...",
        "input_configuration_id": "pre_post",
        "input_mapping": {
            "pre":  {"series_uid": "...", "wado_rs_url": "..."},
            "post": {"series_uid": "...", "wado_rs_url": "..."}
        }
    }
    """
    try:
        # Delegate all logic to model service
        result = model_service.analyze_mri_series(request.json)
        return jsonify(result)

    except ModelNotLoadedError as e:
        logger.error(f"Model not loaded: {e}")
        return jsonify({"error": "Model not loaded"}), 503

    except ValueError as e:
        logger.error(f"Invalid request: {e}")
        return jsonify({"error": str(e)}), 400

    except InferenceError as e:
        logger.error(f"Inference error: {e}")
        return jsonify({"error": "Inference failed"}), 500

    except Exception:
        logger.exception("Unexpected error during MRI analysis")
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    print_security_banner("odelia-model-service")

    # Initialize service before starting server
    initialize_service()

    # Start Flask server (ODV-218 allocates per-model ports via PORT)
    port = int(os.getenv("PORT", "5556"))
    logger.info(f"Starting Flask server on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
