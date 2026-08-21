"""Model construction for the generalized ODELIA model-service block (ODV-214).

Builds a supported model by name via the local ``models`` package. Weights are
init-only here; loading a trained state_dict from HuggingFace is deferred to
ODV-216.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
from config import SERVICE_VERSION, resolve_device
from models import create_model

logger = logging.getLogger(__name__)


def build_model(model_name: str, num_classes: int = 3) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Build a model by name on the configured device (init weights only).

    ``MODEL_DEVICE`` must be set (``cpu``/``cuda``); resolving it here fails
    loudly otherwise.
    """
    device = resolve_device()
    logger.info("Building model '%s' (num_classes=%d) on %s", model_name, num_classes, device)

    model = create_model(model_name, num_classes=num_classes)
    model.eval()
    model.to(device)

    info: dict[str, Any] = {
        "model_name": model_name,
        "architecture": model_name,
        "version": SERVICE_VERSION,
        "num_classes": num_classes,
        "device": device,
        "weights": "init-only",  # ODV-216 overlays a trained state_dict
    }
    logger.info("Model '%s' built (%s)", model_name, model.__class__.__name__)
    return model, info


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _model, _info = build_model(os.getenv("MODEL_NAME", "MST"))
    print(f"Built: {_info}")
