"""
Response formatting for the generalized model service (ODV-214).
Single Responsibility: format inference results into an API response.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_classification_response(
    probs: list[float], model_info: dict[str, Any] | None
) -> dict[str, Any]:
    """Build a generic classification response from per-class probabilities.

    Args:
        probs: Per-class probability list from the model's softmax output.
        model_info: Model metadata (name, num_classes, weights source, ...).

    Returns:
        Response dict with the probabilities and the argmax predicted class.
    """
    predicted_class = max(range(len(probs)), key=probs.__getitem__) if probs else None
    return {
        "model_info": model_info or {},
        "probabilities": probs,
        "predicted_class": predicted_class,
    }
