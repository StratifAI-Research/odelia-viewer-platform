"""
Breast cancer classification preprocessing
Single Responsibility: Data preprocessing and transforms for ResNet model
"""
import logging
import torch
import torchio as tio
import numpy as np
from typing import Union, Tuple
from torchio.transforms.transform import TypeMaskingMethod
from torchio import Subject, Image

logger = logging.getLogger(__name__)


class ImageToTensor:
    """Convert TorchIO Image to tensor"""
    def __call__(self, image: Image):
        return image.data.swapaxes(1, -1)


def parse_per_channel(per_channel, channels):
    """Parse per-channel configuration"""
    return [(ch,) for ch in range(channels)] if per_channel else [tuple(range(channels))]


class ZNormalization(tio.ZNormalization):
    """Custom Z-normalization with percentile clipping"""

    def __init__(self, percentiles: Union[float, Tuple[float, float]] = (0, 100), per_channel=True,
                 masking_method: TypeMaskingMethod = None, **kwargs):
        super().__init__(masking_method=masking_method, **kwargs)
        self.percentiles = percentiles
        self.per_channel = per_channel

    def apply_normalization(self, subject: Subject, image_name: str, mask: torch.Tensor) -> None:
        image = subject[image_name]
        per_channel = parse_per_channel(self.per_channel, image.shape[0])
        image.set_data(torch.cat([
            self._znorm(image.data[chs,], mask[chs,], image_name, image.path)
            for chs in per_channel])
        )

    def _znorm(self, image_data, mask, image_name, image_path):
        cutoff = torch.quantile(image_data.masked_select(mask).float(), torch.tensor(self.percentiles) / 100.0)
        torch.clamp(image_data, *cutoff.to(image_data.dtype).tolist(), out=image_data)
        standardized = self.znorm(image_data, mask)
        if standardized is None:
            raise RuntimeError(f'Standard deviation is 0 for masked values in image "{image_name}" ({image_path})')
        return standardized


class RandomCropOrPad(tio.CropOrPad):
    """Random crop or pad transform"""

    @staticmethod
    def _get_six_bounds_parameters(parameters: np.ndarray):
        return tuple(np.random.randint(0, size + 1) for size in parameters for _ in (0, 1))


def get_preprocessing_pipeline() -> tio.Compose:
    """
    Get preprocessing pipeline for breast cancer classification

    Returns:
        TorchIO Compose transform
    """
    return tio.Compose([
        RandomCropOrPad((256, 256, 32)),
        ZNormalization(per_channel=True, percentiles=(0.5, 99.5), masking_method=lambda x: x > 0),
        ImageToTensor()
    ])


def preprocess_for_side(pre_img: Image, post_img: Image, transform: tio.Compose, device: str) -> torch.Tensor:
    """
    Preprocess pre and post contrast images for one side

    Args:
        pre_img: Pre-contrast TorchIO image
        post_img: Post-contrast TorchIO image
        transform: Preprocessing transform pipeline
        device: Device to move tensor to

    Returns:
        Preprocessed tensor ready for model inference
    """
    # Concatenate pre and post images
    model_input = torch.cat((pre_img.data, post_img.data), dim=0)
    logger.debug(f"Combined input shape: {model_input.shape}")

    # Apply preprocessing
    model_input = transform(model_input)[None].to(device)
    logger.debug(f"After preprocessing shape: {model_input.shape}")

    return model_input
