"""
Model construction for the generalized ODELIA model-service block (ODV-214).

Builds any model by name through the vendored MediSwarm ``create_model``
factory (mediswarm/models/models_config.py). Weights are init-only here;
loading a trained state_dict from HuggingFace is deferred to ODV-216.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Vendored MediSwarm _shared/custom (models + create_model + base_model).
MEDISWARM_ROOT = Path(__file__).resolve().parent / "mediswarm"


def _ensure_mediswarm_on_path() -> None:
    """Put the vendored MediSwarm root on sys.path for its top-level imports.

    The vendored modules import each other as top-level packages
    (``from models import ...``, ``from env_config import ...``), matching
    MediSwarm's own layout, so the root must be importable.
    """
    root = str(MEDISWARM_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def build_model(model_name: str, num_classes: int = 3) -> tuple[Any, dict[str, Any]]:
    """Instantiate a model by name via the vendored create_model factory.

    Args:
        model_name: MODEL_NAME to build (e.g. "Swin3D", "MST", "ResNet50",
            or a "challenge_*" team model).
        num_classes: Classification head width.

    Returns:
        (model, model_info) with the model in eval mode.
    """
    _ensure_mediswarm_on_path()
    from models.models_config import create_model  # vendored MediSwarm

    logger.info(f"Building model '{model_name}' (num_classes={num_classes}) via create_model")

    # Pass env_vars explicitly so create_model never falls back to MediSwarm's
    # training-oriented load_environment_variables(), which requires
    # SCRATCH_DIR / SITE_NAME / DATA_DIR that a serving container won't set.
    env_vars = {
        "model_name": model_name,
        "mediswarm_version": os.getenv("MEDISWARM_VERSION", "vendored"),
    }
    model = create_model(
        logger=logger,
        model_name=model_name,
        num_classes=num_classes,
        env_vars=env_vars,
    )
    model.eval()

    model_info = {
        "model_name": model_name,
        "num_classes": num_classes,
        "source": "mediswarm/create_model",
        "weights": "init-only",  # ODV-216 overlays a trained state_dict
    }
    logger.info(f"✓ Model '{model_name}' built ({model.__class__.__name__})")
    return model, model_info


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    m, info = build_model(os.getenv("MODEL_NAME", "MST"), int(os.getenv("NUM_CLASSES", "3")))
    print(f"Built: {info}")
