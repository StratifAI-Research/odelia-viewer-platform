"""
Response formatting for the generalized model service (ODV-214).
Single Responsibility: format inference results into an API response.
"""

import logging
from typing import Any

from exceptions import InferenceError

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


CLASS_NAMES = ["No lesion", "Benign", "Malignant"]

UNTRAINED_SUFFIX = " (untrained)"


def _reported_name(name: str, weights: str) -> str:
    """Mark names built from init-only weights so an SR cannot read as trained output."""
    return f"{name}{UNTRAINED_SUFFIX}" if weights == "init-only" else name


def build_bilateral_response(
    results: list[tuple[str, list[float]]], model_info: dict[str, Any] | None
) -> dict[str, Any]:
    """Build the router-facing bilateral response from labelled per-view results.

    Adds the ``left``/``right`` keys the router's SR path detects and keeps the
    generalized ``views`` payload alongside. ``model_metadata`` carries model
    identity only -- weight provenance must not reach DICOM.
    """
    info = model_info or {}
    response = build_multiview_response(results, model_info)
    for view in response["views"]:
        probs = view["probabilities"]
        if len(probs) != len(CLASS_NAMES):
            raise InferenceError(
                f"view {view['label']!r} returned {len(probs)} probabilities "
                f"for {len(CLASS_NAMES)} class names"
            )
        idx = view["predicted_class"]
        response[view["label"]] = {
            "prediction": CLASS_NAMES[idx],
            "confidence": float(probs[idx]) * 100.0,
        }
    weights = info.get("weights", "")
    response["model_metadata"] = {
        "model_name": _reported_name(info.get("model_name", "ODELIA"), weights),
        "architecture": _reported_name(info.get("architecture", "Unknown"), weights),
        "version": info.get("version", "Unknown"),
    }
    return response
