"""Resolve the preprocessor for a model: a model-local override, else the default."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path

from models import available_models

from preprocessing.pipeline import preprocess as default_preprocess
from preprocessing.types import VolumeView

Preprocessor = Callable[[Path, str], list[VolumeView]]


def resolve_preprocessor(model_name: str) -> Preprocessor:
    """Return ``models/<dir>/preprocess.preprocess`` if present, else the default."""
    dir_name = available_models().get(model_name)
    if dir_name:
        try:
            module = importlib.import_module(f"models.{dir_name}.preprocess")
        except ModuleNotFoundError:
            module = None
        if module is not None and hasattr(module, "preprocess"):
            return module.preprocess  # type: ignore[no-any-return]
    return default_preprocess
