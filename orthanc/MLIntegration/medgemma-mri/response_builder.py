"""
Response formatting for MedGemma classification results
Single Responsibility: Format analysis results into API response
"""
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def build_bilateral_response(parsed_result: Dict[str, Any], model_info: Dict[str, str]) -> Dict[str, Any]:
    """
    Build bilateral classification response format matching existing API contract.

    Args:
        parsed_result: Dictionary with 'left' and 'right' classification results
                      Each side has: {"prediction": str, "confidence": float}
        model_info: Dictionary with model metadata

    Returns:
        Response dictionary with bilateral classification compatible with
        existing breast-cancer-classification and MST-classification APIs
    """
    left_result = parsed_result.get("left", {"prediction": "No lesion", "confidence": 50.0})
    right_result = parsed_result.get("right", {"prediction": "No lesion", "confidence": 50.0})

    # Build response matching existing API format
    response = {
        "left": {
            "prediction": left_result["prediction"],
            "confidence": round(left_result["confidence"], 1)
        },
        "right": {
            "prediction": right_result["prediction"],
            "confidence": round(right_result["confidence"], 1)
        },
        "model_metadata": {
            "model_name": model_info.get("model_name", "MedGemma"),
            "architecture": model_info.get("architecture", "Vision-Language Model"),
            "version": model_info.get("version", "1.5-4b-it")
        }
    }

    logger.info(f"Built bilateral response:")
    logger.info(f"  Left: {response['left']['prediction']} ({response['left']['confidence']}%)")
    logger.info(f"  Right: {response['right']['prediction']} ({response['right']['confidence']}%)")

    return response




