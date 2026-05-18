"""
Response formatting for breast cancer classification results
Single Responsibility: Format analysis results into API response
"""
import logging

logger = logging.getLogger(__name__)


def build_bilateral_classification(left_result: dict, right_result: dict) -> dict:
    """
    Build bilateral classification response

    Args:
        left_result: Classification result for left breast
        right_result: Classification result for right breast

    Returns:
        Response dictionary with bilateral classification
    """
    response = {
        "left": left_result,
        "right": right_result
    }

    logger.info(f"Built bilateral response:")
    logger.info(f"  Left: {left_result}")
    logger.info(f"  Right: {right_result}")

    return response
