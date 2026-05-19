"""
MedGemma MRI Classification Service - Flask microservice
Refactored: Thin HTTP layer delegating to model_service
"""
import os
import logging
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

from shared.config import StorageConfig
from config import MedGemmaConfig
from model_service import MedGemmaModelService
from exceptions import ModelNotLoadedError, ModelAuthenticationError, InferenceError, ResponseParsingError

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global model service
model_service: MedGemmaModelService = None


def initialize_service():
    """Initialize configurations and model service"""
    global model_service

    # Load configurations from environment
    medgemma_config = MedGemmaConfig.from_env()

    storage_config = StorageConfig(
        image_folder=Path(os.getenv("IMAGE_FOLDER", "./images")),
        cleanup_on_start=True
    )

    # Create necessary directories
    os.makedirs(storage_config.image_folder, exist_ok=True)

    # Initialize model service
    model_service = MedGemmaModelService(medgemma_config, storage_config)
    model_service.initialize_model()


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify(model_service.get_health_status())


@app.route("/analyze/mri", methods=["POST"])
def analyze_mri():
    """
    Analyze MRI series using MedGemma model

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
            "prediction": "No lesion" | "Benign" | "Malignant",
            "confidence": 87.5
        },
        "right": {
            "prediction": "No lesion" | "Benign" | "Malignant",
            "confidence": 65.2
        },
        "model_metadata": {
            "model_name": "MedGemma",
            "architecture": "Vision-Language Model",
            "version": "1.5-4b-it"
        }
    }
    """
    try:
        # Delegate all logic to model service
        result = model_service.analyze_mri_series(request.json)
        return jsonify(result)

    except ModelNotLoadedError as e:
        logger.error(f"Model not loaded: {e}")
        return jsonify({"error": "Model not loaded", "details": str(e)}), 503

    except ModelAuthenticationError as e:
        logger.error(f"Authentication error: {e}")
        return jsonify({"error": "Authentication failed", "details": str(e)}), 401

    except ValueError as e:
        logger.error(f"Invalid request: {e}")
        return jsonify({"error": "Invalid request", "details": str(e)}), 400

    except ResponseParsingError as e:
        logger.error(f"Response parsing error: {e}")
        return jsonify({
            "error": "Failed to parse model response",
            "details": str(e),
            "raw_response": e.raw_response[:500] if e.raw_response else None
        }), 422

    except InferenceError as e:
        logger.error(f"Inference error: {e}")
        return jsonify({"error": "Inference failed", "details": str(e)}), 500

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


if __name__ == "__main__":
    # Initialize service before starting server
    initialize_service()

    # Start Flask server
    logger.info("Starting Flask server on 0.0.0.0:5557")
    app.run(host="0.0.0.0", port=5557, debug=False, use_reloader=False)
