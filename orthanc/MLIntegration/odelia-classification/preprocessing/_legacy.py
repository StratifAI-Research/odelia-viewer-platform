"""
Minimal single-channel preprocessing for the generalized model service (ODV-214).

This is the ODV-217 seam: it is replaced there by the exact shared MediSwarm
bilateral/unilateral transform. Here we provide a deterministic, dependency-light
transform so the block is runnable end-to-end.
"""

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def prepare_single_channel(
    nifti_path: Path, device: str, target_shape: tuple[int, int, int]
) -> torch.Tensor:
    """Load a NIfTI volume as a normalized ``[1, 1, D, H, W]`` tensor on ``device``.

    Resizes to ``target_shape`` — the served subunit's INPUT_SIZE — so the model
    receives exactly the resolution it was built and pre-flight-validated for.
    ODV-217 replaces this with the shared MediSwarm transform (exact resampling,
    normalization and cropping); this placeholder is intentionally minimal.
    """
    import torchio as tio

    logger.info(f"Loading NIfTI for inference: {nifti_path}")
    image = tio.ScalarImage(str(nifti_path))

    d, h, w = target_shape
    transform = tio.Compose(
        [
            tio.ToCanonical(),
            tio.Resize((d, h, w)),
            tio.ZNormalization(),
        ]
    )
    data = transform(image).data.float()  # [C, ...]
    if data.shape[0] != 1:
        data = data[:1]  # enforce single channel
    return data.unsqueeze(0).to(device)  # [1, 1, D, H, W]
