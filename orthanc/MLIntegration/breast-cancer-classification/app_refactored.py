"""
Breast Cancer Classification Service - Flask microservice
Refactored: Thin HTTP layer delegating to model_service
"""
import os
import logging
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

from shared.config import StorageConfig
from config import BreastCancerConfig
from model_service import BreastCancerModelService
from exceptions import ModelNotLoadedError, InferenceError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global model service
model_service: BreastCancerModelService = None


def initialize_service():
    """Initialize configurations and model service"""
    global model_service

    # Load configurations from environment
    bc_config = BreastCancerConfig.from_env()

    storage_config = StorageConfig(
        image_folder=Path(os.getenv("IMAGE_FOLDER", "./images")),
        cleanup_on_start=True
    )

    # Create necessary directories
    os.makedirs(storage_config.image_folder, exist_ok=True)

    # Initialize model service
    model_service = BreastCancerModelService(bc_config, storage_config)
    model_service.initialize_model()


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify(model_service.get_health_status())


@app.route("/analyze/mri", methods=["POST"])
def analyze_mri():
    """
    Analyze MRI series using breast cancer classification model

    Input format:
    {
        "wado_rs_retrieval": [
            {
                "retrieval_url": "http://orthanc-viewer:8042/dicom-web/studies/{study}/series/{series}",
                "study_uid": "1.2.3...",
                "series_uid": "1.2.3..."
            }
        ],
        "study_uid": "1.2.3..."
    }

    Returns:
    {
        "left": {
            "prediction": "Cancerous" | "Not Cancerous",
            "confidence": 87.5
        },
        "right": {
            "prediction": "Cancerous" | "Not Cancerous",
            "confidence": 65.2
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
        return jsonify({"error": str(e)}), 500

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


if __name__ == "__main__":
    # Initialize service before starting server
    initialize_service()

    # Start Flask server
    logger.info("Starting Flask server on 0.0.0.0:5555")
    app.run(host="0.0.0.0", port=5555, debug=False, use_reloader=False)
