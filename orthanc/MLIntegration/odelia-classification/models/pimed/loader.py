"""Pimed subunit loader — the contract entrypoint (ODV-214)."""

from __future__ import annotations

from torch import nn

NAME = "Pimed"
INPUT_SIZE = (32, 224, 224)


def create(num_classes: int = 3, loss_kwargs: dict | None = None) -> nn.Module:
    """Build Pimed at its trained config (init weights)."""
    from .model import create_model as _build

    return _build(
        model_name="resnet18",
        n_input_channels=1,
        spatial_dims=3,
        norm="batch",
        num_classes=num_classes,
        loss_kwargs=loss_kwargs,
    )
