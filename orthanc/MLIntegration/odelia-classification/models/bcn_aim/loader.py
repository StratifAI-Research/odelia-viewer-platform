"""BCN_AIM subunit loader — the contract entrypoint (ODV-214)."""

from __future__ import annotations

from torch import nn

NAME = "BCN_AIM"
# (D, H, W): SwinUNETR depth divisible by 32; spatial 224.
INPUT_SIZE = (32, 224, 224)


def create(num_classes: int = 3, loss_kwargs: dict | None = None) -> nn.Module:
    """Build BCN_AIM at its trained config (init weights)."""
    from .swinunetr import create_model as _build

    return _build(
        img_size=224,
        n_input_channels=1,
        spatial_dims=3,
        num_classes=num_classes,
        loss_kwargs=loss_kwargs,
    )
