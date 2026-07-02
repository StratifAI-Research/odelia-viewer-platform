"""agaldran subunit loader — the contract entrypoint (ODV-214)."""

from __future__ import annotations

from torch import nn

NAME = "agaldran"
INPUT_SIZE = (32, 224, 224)


def create(num_classes: int = 3, loss_kwargs: dict | None = None) -> nn.Module:
    """Build the agaldran model at its trained config (init weights)."""
    from .model_factory import model_factory as _build

    return _build(
        arch="mvit_v2_s",
        in_ch=1,
        pretrained_path=None,
        seed=123,
        num_classes=num_classes,
        loss_kwargs=loss_kwargs,
    )
