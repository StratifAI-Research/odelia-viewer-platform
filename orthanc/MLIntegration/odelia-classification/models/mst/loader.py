"""MST subunit loader — the contract entrypoint (ODV-214)."""

from __future__ import annotations

from torch import nn

NAME = "MST"
# (D, H, W): depth 32, DINOv2 patch-14 spatial 224. Matches MediSwarm training.
INPUT_SIZE = (32, 224, 224)


def create(num_classes: int = 3, loss_kwargs: dict | None = None) -> nn.Module:
    """Build MST at its trained config (init weights)."""
    from .model import MST

    return MST(
        n_input_channels=1,
        num_classes=num_classes,
        spatial_dims=3,
        loss_kwargs=loss_kwargs,
    )
