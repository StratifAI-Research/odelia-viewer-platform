"""ODELIA model subunits: flat, self-contained packages + a discovery API.

Each subunit dir (``mst/``, ``bcn_aim/``, ...) ships a ``loader.py`` exposing
``NAME``, ``INPUT_SIZE`` and ``create()``. ``loader_util`` discovers them; there
is no central factory to edit when adding a model. ``base`` holds the shared
inference base the in-repo subunits build on (convenience, not part of the
cross-repo contract).
"""

from .base import BasicClassifier, ModelWrapper
from .loader_util import (
    assert_forward_contract,
    available_models,
    create_model,
    input_size,
    resolve_baked_model,
)

__all__ = [
    "BasicClassifier",
    "ModelWrapper",
    "assert_forward_contract",
    "available_models",
    "create_model",
    "input_size",
    "resolve_baked_model",
]
