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


def build_multiview_response(
    results: list[tuple[str, list[float]]], model_info: dict[str, Any] | None
) -> dict[str, Any]:
    """Build a labelled per-view response from ``(label, probabilities)`` pairs.

    One entry per forward pass — 1, 2 (left/right) or N. Each view reuses the
    single-view builder for its probabilities + argmax.
    """
    views = []
    for label, probs in results:
        single = build_classification_response(probs, model_info)
        views.append(
            {
                "label": label,
                "probabilities": single["probabilities"],
                "predicted_class": single["predicted_class"],
            }
        )
    return {"model_info": model_info or {}, "views": views}
