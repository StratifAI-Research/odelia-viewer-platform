"""MediSwarm-faithful default preprocessor: subtraction NIfTI -> per-side views.

Fuses MediSwarm ``step3`` (bilateral prep + left/right split) with the
``ODELIA_Dataset3D`` unilateral inference transform (augmentation off).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torchio as tio

from preprocessing.transforms import (
    ZNormalization,
    crop_breast_height,
    image_to_tensor,
)
from preprocessing.types import VolumeView

_TARGET_SPACING = (0.7, 0.7, 3)
_BILATERAL_SHAPE = (512, 512, 32)
_UNILATERAL_SHAPE = (224, 224, 32)
_SPLIT = {
    "right": tio.Crop((256, 0, 0, 0, 0, 0)),
    "left": tio.Crop((0, 256, 0, 0, 0, 0)),
}


def _mask(x: torch.Tensor) -> torch.Tensor:
    return (x > x.min()) & (x < x.max())


def preprocess(sub_nifti: Path, device: str) -> list[VolumeView]:
    """Turn a subtraction NIfTI into left/right model-ready views."""
    image = tio.ScalarImage(str(sub_nifti))
    image = tio.ToCanonical()(image)
    image = tio.Resample(_TARGET_SPACING)(image)

    pad_value = float(image.data.min().item())  # global-min constant (not per-axis 'minimum')
    image = tio.CropOrPad(_BILATERAL_SHAPE, padding_mode=pad_value)(image)
    image = crop_breast_height(image)(image)

    views: list[VolumeView] = []
    for side in ("left", "right"):
        side_img = _SPLIT[side](image)
        side_img = tio.Flip((1, 0))(side_img)
        side_img = tio.CropOrPad(_UNILATERAL_SHAPE)(side_img)
        side_img = ZNormalization(
            percentiles=(0.5, 99.5),
            per_channel=True,
            per_slice=False,
            masking_method=_mask,
        )(side_img)
        tensor = image_to_tensor(side_img).unsqueeze(0).to(device)
        views.append(VolumeView(label=side, tensor=tensor))
    return views
