"""Discover and build ODELIA model subunits (ODV-214).

A subunit is a package dir under ``models/`` containing ``loader.py``, which
exposes the contract:

    NAME: str                        canonical model name
    INPUT_SIZE: tuple[int, int, int] (D, H, W) the service preprocesses to
    create(num_classes=3, loss_kwargs=None) -> nn.Module

There is no central registry to edit — adding a model is dropping in a subunit
dir. Discovery reads only ``loader.py`` module attributes (no model
construction, no network), so a subunit with a broken heavy dependency cannot
break discovery of the others.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from types import ModuleType

import torch
from torch import nn

_PKG = __package__  # "models"
_ROOT = Path(__file__).resolve().parent


def _subunit_dirs() -> list[str]:
    """Names of package dirs that carry a loader.py (the subunits)."""
    return sorted(e.name for e in _ROOT.iterdir() if e.is_dir() and (e / "loader.py").is_file())


def _loader(dir_name: str) -> ModuleType:
    return importlib.import_module(f"{_PKG}.{dir_name}.loader")


def available_models() -> dict[str, str]:
    """Map canonical model ``NAME`` -> package dir for every discovered subunit."""
    return {_loader(d).NAME: d for d in _subunit_dirs()}


def create_model(
    model_name: str, num_classes: int = 3, loss_kwargs: dict | None = None
) -> nn.Module:
    """Build a subunit model by canonical name (init weights)."""
    models = available_models()
    if model_name not in models:
        raise ValueError(f"Unsupported model name: {model_name!r} (available: {sorted(models)})")
    return _loader(models[model_name]).create(num_classes=num_classes, loss_kwargs=loss_kwargs)


def input_size(model_name: str) -> tuple[int, int, int]:
    """The (D, H, W) the service must preprocess to for ``model_name``."""
    models = available_models()
    if model_name not in models:
        raise ValueError(f"Unsupported model name: {model_name!r}")
    d, h, w = _loader(models[model_name]).INPUT_SIZE
    return d, h, w


def resolve_baked_model() -> str:
    """Return the single subunit's name that this image serves.

    One image = one model: the build prunes non-selected subunits, so exactly
    one remains and is served with no runtime selection. In a dev checkout where
    every subunit is present, ``MODEL_NAME`` disambiguates; without it we fail
    loudly rather than guess.
    """
    models = available_models()
    if len(models) == 1:
        return next(iter(models))
    override = os.getenv("MODEL_NAME")
    if override:
        if override not in models:
            raise ValueError(
                f"MODEL_NAME={override!r} not among available subunits {sorted(models)}"
            )
        return override
    raise RuntimeError(
        f"Expected exactly one baked subunit but found {sorted(models)}; "
        "set MODEL_NAME to pick one (dev checkout)."
    )


def assert_forward_contract(model: nn.Module, model_name: str) -> tuple[int, ...]:
    """Run one synthetic forward and assert the fixed inference contract.

    Input  ``(1, 1, D, H, W)`` at the subunit's INPUT_SIZE, on the model's
    device. Output must be ``(1, 3)`` logits that softmax cleanly. Shared by the
    startup pre-flight and the roster test so "contract holds" has one
    definition. Returns the observed output shape.
    """
    d, h, w = input_size(model_name)
    device = next(model.parameters()).device
    with torch.inference_mode():
        out = model(torch.zeros(1, 1, d, h, w, device=device))
    shape = tuple(out.shape)
    if shape != (1, 3):
        raise AssertionError(
            f"{model_name}: contract violation — expected (1, 3) logits, got {shape}"
        )
    torch.softmax(out, dim=1)
    return shape
