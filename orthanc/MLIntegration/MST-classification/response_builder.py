"""
Response formatting for MST classification results
Single Responsibility: Format analysis results into API response
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)


def build_bilateral_response(probs: dict, attention_maps: dict, model_info: dict) -> dict:
    """
    Build bilateral classification response format

    Args:
        probs: Dictionary with 'left' and 'right' probability arrays
        attention_maps: Dictionary with attention map data (data, shape, dtype)
        model_info: Dictionary with model metadata

    Returns:
        Response dictionary with bilateral classification and attention maps
    """
    # Class names for MST model
    class_names = ["No lesion", "Benign", "Malignant"]

    left_probs = probs['left']
    right_probs = probs['right']

    left_class_idx = int(np.argmax(left_probs))
    right_class_idx = int(np.argmax(right_probs))

    # Create bilateral classification format matching the viewer's expectations
    left_classification = {
        "prediction": class_names[left_class_idx],
        "confidence": float(left_probs[left_class_idx]) * 100.0  # Convert to percentage
    }

    right_classification = {
        "prediction": class_names[right_class_idx],
        "confidence": float(right_probs[right_class_idx]) * 100.0  # Convert to percentage
    }

    # Store model metadata separately
    model_metadata = {
        "model_name": model_info.get("model_name", "MST"),
        "architecture": model_info.get("architecture", "Vision Transformer"),
        "version": model_info.get("version", "1.0")
    }

    response = {
        "left": left_classification,
        "right": right_classification,
        "model_metadata": model_metadata,
        "attention_maps": attention_maps
    }

    logger.info(f"Built bilateral response:")
    logger.info(f"  Left: {left_classification['prediction']} ({left_classification['confidence']:.1f}%)")
    logger.info(f"  Right: {right_classification['prediction']} ({right_classification['confidence']:.1f}%)")
    logger.info(f"  Attention maps: {attention_maps['shape'][0]} slices")

    return response
