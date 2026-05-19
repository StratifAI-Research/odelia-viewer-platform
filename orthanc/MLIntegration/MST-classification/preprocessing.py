"""
MST-specific preprocessing and attention map generation
Single Responsibility: Data preprocessing for MST model
"""
import sys
import os
import logging
import numpy as np
import torch
import base64
from pathlib import Path

logger = logging.getLogger(__name__)


def prepare_for_inference(nifti_path: Path, model_path: Path):
    """
    Prepare NIfTI image for MST model inference

    Args:
        nifti_path: Path to NIfTI file
        model_path: Path to model directory (needed for imports)

    Returns:
        TorchIO ScalarImage ready for inference
    """
    # Add model path to sys.path to import the reference implementation
    if str(model_path) not in sys.path:
        sys.path.insert(0, str(model_path))

    import torchio as tio

    logger.info(f"Loading NIfTI as TorchIO image: {nifti_path}")
    img = tio.ScalarImage(str(nifti_path))

    return img


def generate_attention_overlays(img_tensor: torch.Tensor, weight_tensor: torch.Tensor, model_path: Path) -> dict:
    """
    Generate RGB overlay images combining MRI and attention maps

    Args:
        img_tensor: Input image tensor [B, C, D, H, W]
        weight_tensor: Attention weight tensor [B, C, D, H, W]
        model_path: Path to model directory (needed for imports)

    Returns:
        Dictionary with:
            - data: Base64 encoded RGB overlay data
            - shape: Shape of overlay array [slices, rows, cols, 3]
            - dtype: Data type (uint8)
    """
    # Add model path to sys.path to import the reference implementation
    if str(model_path) not in sys.path:
        sys.path.insert(0, str(model_path))

    from predict_attention import minmax_norm, tensor_cam2image

    logger.info("Generating attention overlay images...")

    # Prepare tensors in the format expected by tensor_cam2image
    # Input format: [B, C, D, H, W] for both img and weight
    img_prepared = img_tensor.swapaxes(1, -1).unsqueeze(0)  # [1, C, D, H, W]
    weight_prepared = weight_tensor.swapaxes(1, -1).unsqueeze(0)  # [1, C, D, H, W]

    # Normalize to [0, 1] for visualization
    img_norm = minmax_norm(img_prepared)
    weight_norm = minmax_norm(weight_prepared)

    # Generate RGB overlay images
    overlay = tensor_cam2image(img_norm, weight_norm, batch=0, alpha=0.5)
    logger.info(f"Generated {overlay.shape[0]} RGB overlay images with shape {overlay.shape}")

    # Convert overlay to numpy and prepare ALL slices for transmission
    # overlay is [num_slices, 3, H, W] from tensor_cam2image
    # Transpose to [num_slices, rows, cols, 3] for DICOM pixel data
    overlay_np = overlay.cpu().numpy()
    overlay_np = np.transpose(overlay_np, (0, 2, 3, 1))  # [D, 3, H, W] -> [D, H, W, 3]

    # Reverse slice order to match DICOM orientation
    overlay_np = overlay_np[::-1]

    logger.info(f"Transposed overlay shape: {overlay_np.shape} [slices, rows(H), cols(W), RGB]")

    # Convert to uint8 and encode as base64 for efficient transmission
    overlay_uint8 = (overlay_np * 255).astype(np.uint8)
    overlay_bytes = overlay_uint8.tobytes()
    overlay_b64 = base64.b64encode(overlay_bytes).decode('ascii')

    return {
        "data": overlay_b64,
        "shape": list(overlay_uint8.shape),  # [num_slices, rows, cols, 3]
        "dtype": "uint8"
    }
