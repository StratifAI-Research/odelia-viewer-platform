"""Preprocessing output types."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class VolumeView:
    """One model-ready volume to run a single forward pass on.

    ``label`` is a free tag (e.g. ``"left"``/``"right"`` for the MediSwarm
    default, ``"volume"`` for a single-view model). ``tensor`` is
    ``[1, 1, D, H, W]`` on the target device.
    """

    label: str
    tensor: torch.Tensor
