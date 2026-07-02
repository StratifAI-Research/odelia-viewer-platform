"""LME_ABMIL subunit loader — the contract entrypoint (ODV-214).

The model consumes the standard single-channel subtraction tensor: its own
``forward`` transposes and expands the intensity channel to RGB (``C==1`` path),
matching MediSwarm's single-channel dataloader. No channel assembly here.
"""

from __future__ import annotations

from torch import nn

NAME = "LME_ABMIL"
# (D, H, W): ABMIL Swin requires spatial 224; depth is the MIL bag.
INPUT_SIZE = (32, 224, 224)


def create(num_classes: int = 3, loss_kwargs: dict | None = None) -> nn.Module:
    """Build LME_ABMIL at its trained config (init weights)."""
    from .model import create_model as _build

    return _build(
        model_type="swin",
        n_input_channels=3,
        num_classes=num_classes,
        loss_kwargs=loss_kwargs,
    )
