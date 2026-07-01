"""Factory for building an ODELIA roster classifier by name.

The roster is MST plus the five ODELIA challenge models, each pinned to the
config it was trained with (mirrors MediSwarm's registry). Models are built with
random / init weights here (ODV-214); trained-weight loading is ODV-216.
Challenge models are imported lazily so a missing challenge dependency never
breaks building MST.
"""

from __future__ import annotations

import importlib

from torch import nn

from .mst import MST

# Challenge roster: name -> (submodule, factory attr, training-config kwargs).
# Configs are the values used in training (MediSwarm CHALLENGE_MODELS
# persistor_args); pretrained_path is dropped so construction is network-free.
_CHALLENGE = {
    "DivideAndConquer": (
        "divide_and_conquer.model",
        "create_model",
        {"n_input_channels": 1, "spatial_dims": 3, "pretrained_path": None},
    ),
    "BCN_AIM": (
        "bcn_aim.swinunetr",
        "create_model",
        {"img_size": 224, "n_input_channels": 1, "spatial_dims": 3},
    ),
    "agaldran": (
        "agaldran.model_factory",
        "model_factory",
        {"arch": "mvit_v2_s", "in_ch": 1, "pretrained_path": None, "seed": 123},
    ),
    "LME_ABMIL": (
        "lme_abmil.model",
        "create_model",
        {"model_type": "swin", "n_input_channels": 3},
    ),
    "Pimed": (
        "pimed.model",
        "create_model",
        {"model_name": "resnet18", "n_input_channels": 1, "spatial_dims": 3, "norm": "batch"},
    ),
}


def create_model(
    model_name: str, num_classes: int = 3, loss_kwargs: dict | None = None
) -> nn.Module:
    """Build a roster model by name at its trained config.

    Supported: ``MST`` and the challenge roster (``DivideAndConquer``,
    ``BCN_AIM``, ``agaldran``, ``LME_ABMIL``, ``Pimed``). ``loss_kwargs`` is
    forwarded so a ``_class_weight`` buffer is registered when present, keeping
    the state_dict aligned with class-weighted checkpoints (ODV-216).
    """
    if model_name == "MST":
        return MST(
            n_input_channels=1,
            num_classes=num_classes,
            spatial_dims=3,
            loss_kwargs=loss_kwargs,
        )
    if model_name in _CHALLENGE:
        submodule, attr, kwargs = _CHALLENGE[model_name]
        module = importlib.import_module(f"{__package__}.challenge.{submodule}")
        factory = getattr(module, attr)
        return factory(num_classes=num_classes, loss_kwargs=loss_kwargs, **kwargs)

    raise ValueError(
        f"Unsupported model name: {model_name!r} (expected MST or one of {sorted(_CHALLENGE)})"
    )
