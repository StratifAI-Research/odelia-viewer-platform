"""Construct the baked ODELIA subunit and strictly load its local weights."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
from config import SERVICE_VERSION, resolve_device
from models import available_models, create_model
from weights import load_weights

logger = logging.getLogger(__name__)


def build_model(model_name: str, num_classes: int = 3) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Build and load a model on the configured device, without downloads.

    ``MODEL_DEVICE`` must be set (``cpu``/``cuda``); resolving it here fails
    loudly otherwise.
    """
    device = resolve_device()
    logger.info("Building model '%s' (num_classes=%d) on %s", model_name, num_classes, device)

    directory = Path(__file__).parent / "models" / available_models()[model_name]
    lock_path = Path(__file__).parent / "models-lock.json"
    lock = json.loads(lock_path.read_text()) if lock_path.is_file() else {}
    weights_path = directory / "weights.safetensors"
    if not weights_path.is_file() and (
        os.getenv("ALLOW_INIT_WEIGHTS") != "1"
        or (directory / "release.json").exists()
        or lock.get("weights", "init-only") != "init-only"
    ):
        raise FileNotFoundError(
            f"Missing trained weights: {weights_path}. "
            "ALLOW_INIT_WEIGHTS=1 permits an explicitly untrained development preview."
        )
    model = create_model(model_name, num_classes=num_classes)
    provenance = (
        load_weights(model, directory, model_name, lock_path)
        if weights_path.is_file()
        else {"weights": "init-only"}
    )
    model.eval()
    model.to(device)

    info: dict[str, Any] = {
        "model_name": model_name,
        "architecture": model_name,
        "version": SERVICE_VERSION,
        "num_classes": num_classes,
        "device": device,
        **provenance,
    }
    if provenance.get("model_release"):
        info["service_version"] = SERVICE_VERSION
        info["version"] = provenance["model_release"]
    logger.info("Model '%s' built (%s)", model_name, model.__class__.__name__)
    return model, info


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _model, _info = build_model(os.getenv("MODEL_NAME", "MST"))
    print(f"Built: {_info}")
