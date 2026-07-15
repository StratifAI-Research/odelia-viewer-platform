"""Resolve the preprocessor for a model: a model-local override, else the default."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from pathlib import Path

from models import available_models

from preprocessing.pipeline import preprocess as default_preprocess
from preprocessing.types import VolumeView

logger = logging.getLogger(__name__)

Preprocessor = Callable[[Path, str], list[VolumeView]]


def resolve_preprocessor(model_name: str) -> Preprocessor:
    """Return ``models/<dir>/preprocess.preprocess`` if present, else the default.

    A genuinely absent override module falls back to the default. An import
    error raised from *inside* an existing override (e.g. a missing dependency)
    propagates rather than being swallowed into a silent fallback to the default
    transform. The resolved preprocessor is logged so an ignored override is
    diagnosable.
    """
    dir_name = available_models().get(model_name)
    if dir_name:
        module_name = f"models.{dir_name}.preprocess"
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            # Only the override module itself being absent counts as no-override;
            # a missing import *within* an existing override must not be swallowed.
            if exc.name != module_name:
                raise
            module = None
        if module is not None and hasattr(module, "preprocess"):
            logger.info("Preprocessor for %s: override %s", model_name, module_name)
            return module.preprocess  # type: ignore[no-any-return]
    logger.info("Preprocessor for %s: default MediSwarm pipeline", model_name)
    return default_preprocess
