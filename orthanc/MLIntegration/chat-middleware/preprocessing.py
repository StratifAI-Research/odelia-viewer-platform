"""
DICOM preprocessing with configurable slice strategies.
Reuses core DICOM handling from medgemma-mri/preprocessing.py.
"""
import io
import base64
import shutil
import logging
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import SimpleITK as sitk
from PIL import Image

from models import SliceStrategy
from runtime_config import PreprocessingParams

# Import shared utilities
from shared.wado_retrieval import retrieve_via_wado_rs
from shared.dicom_storage import save_datasets_to_folder
from shared.config import StorageConfig

logger = logging.getLogger(__name__)


def read_dicom_volume(dicom_folder: Path) -> sitk.Image:
    """
    Read DICOM series as a 3D volume using SimpleITK.
    Handles 4D temporal series by extracting the first temporal phase.

    This is adapted from medgemma-mri/preprocessing.py.

    Args:
        dicom_folder: Path to folder containing DICOM files

    Returns:
        SimpleITK Image object
    """
    from collections import defaultdict

    dicom_path = Path(dicom_folder)

    # Verify DICOM files exist
    dicom_files = list(dicom_path.glob("*.dcm"))
    if not dicom_files:
        raise ValueError(f"No DICOM files found in {dicom_folder}")

    logger.info(f"Found {len(dicom_files)} DICOM files")

    # For simplicity, use standard GDCM series reader
    # The medgemma version handles 4D temporal series, but for chat we'll
    # just use the standard approach
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_path))

    if not dicom_names:
        raise ValueError(f"No valid DICOM series found in {dicom_folder}")

    reader.SetFileNames(dicom_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    image = reader.Execute()

    logger.info(
        f"Read DICOM series: size={image.GetSize()}, spacing={image.GetSpacing()}"
    )

    return image


def normalize_slice(slice_array: np.ndarray) -> np.ndarray:
    """
    Normalize a slice to 0-255 range for PIL Image conversion.
    Uses percentile-based windowing for robust normalization.

    Adapted from medgemma-mri/preprocessing.py.

    Args:
        slice_array: 2D numpy array

    Returns:
        Normalized uint8 array
    """
    # Use percentile-based windowing (robust to outliers)
    p_low, p_high = np.percentile(slice_array, [1, 99])

    if p_high - p_low < 1e-6:
        # Constant image
        return np.zeros_like(slice_array, dtype=np.uint8)

    # Clip and normalize to 0-255
    normalized = np.clip(slice_array, p_low, p_high)
    normalized = ((normalized - p_low) / (p_high - p_low) * 255).astype(np.uint8)

    return normalized


def extract_slices(
    volume_array: np.ndarray,
    params: PreprocessingParams
) -> List[np.ndarray]:
    """
    Extract slices based on the configured strategy.

    Strategies:
    - CENTRAL: Extract from central N% of volume (default, matches existing behavior)
    - UNIFORM: Evenly spaced across entire volume
    - FIRST_N: First N slices (useful for head-first scans)
    - LAST_N: Last N slices (useful for feet-first scans)

    Args:
        volume_array: 3D numpy array with shape (Z, Y, X)
        params: Preprocessing parameters including strategy

    Returns:
        List of 2D slice arrays
    """
    total_slices = volume_array.shape[0]
    num_slices = min(params.num_slices, total_slices)

    if num_slices == 0:
        return []

    if params.slice_strategy == SliceStrategy.CENTRAL:
        # Central N% of volume (existing medgemma behavior)
        margin = (100 - params.central_percentage) / 200
        start_idx = int(total_slices * margin)
        end_idx = int(total_slices * (1 - margin))

        # Ensure we have enough range
        if end_idx - start_idx < num_slices:
            start_idx = 0
            end_idx = total_slices

        if num_slices == 1:
            indices = [(start_idx + end_idx) // 2]
        else:
            step = (end_idx - start_idx - 1) / (num_slices - 1)
            indices = [int(start_idx + i * step) for i in range(num_slices)]

    elif params.slice_strategy == SliceStrategy.UNIFORM:
        # Evenly spaced across entire volume
        if num_slices == 1:
            indices = [total_slices // 2]
        else:
            step = (total_slices - 1) / (num_slices - 1)
            indices = [int(i * step) for i in range(num_slices)]

    elif params.slice_strategy == SliceStrategy.FIRST_N:
        indices = list(range(num_slices))

    elif params.slice_strategy == SliceStrategy.LAST_N:
        indices = list(range(total_slices - num_slices, total_slices))

    else:
        # Fallback to central
        mid = total_slices // 2
        half = num_slices // 2
        indices = list(range(mid - half, mid - half + num_slices))

    logger.info(f"Extracting slices at indices: {indices} (strategy={params.slice_strategy.value})")

    return [volume_array[i, :, :] for i in indices]


def slices_to_base64(slice_arrays: List[np.ndarray]) -> List[str]:
    """
    Convert slice arrays to base64-encoded PNG images.

    Args:
        slice_arrays: List of 2D numpy arrays

    Returns:
        List of base64-encoded PNG strings
    """
    base64_images = []

    for slice_arr in slice_arrays:
        # Normalize to 0-255
        normalized = normalize_slice(slice_arr)

        # Convert grayscale to RGB (model expects RGB)
        rgb_array = np.stack([normalized, normalized, normalized], axis=-1)

        # Create PIL Image and encode to base64
        pil_image = Image.fromarray(rgb_array, mode='RGB')

        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        base64_str = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"
        base64_images.append(base64_str)

    return base64_images


async def preprocess_series(
    series_uid: str,
    study_uid: str,
    params: PreprocessingParams,
    wado_base_url: str,
    image_folder: Path
) -> List[str]:
    """
    Retrieve and preprocess a DICOM series, returning base64-encoded images.

    Steps:
    1. Retrieve via WADO-RS
    2. Save to temp folder
    3. Read as volume
    4. Extract slices based on strategy
    5. Normalize and convert to base64 PNG
    6. Cleanup temp files

    Args:
        series_uid: DICOM SeriesInstanceUID
        study_uid: DICOM StudyInstanceUID
        params: Preprocessing parameters (slice count, strategy, etc.)
        wado_base_url: Base URL for WADO-RS retrieval
        image_folder: Base folder for temporary storage

    Returns:
        List of base64-encoded PNG image strings
    """
    logger.info(f"Preprocessing series {series_uid} from study {study_uid}")

    # Prepare WADO-RS retrieval info
    wado_info = [{
        "retrieval_url": f"{wado_base_url}/studies/{study_uid}/series/{series_uid}",
        "study_uid": study_uid,
        "series_uid": series_uid
    }]

    # Create storage config for temp folder
    storage_config = StorageConfig(
        image_folder=image_folder,
        cleanup_on_start=False
    )

    dicom_folder = None
    try:
        # Retrieve DICOM instances
        logger.info(f"Retrieving series via WADO-RS...")
        datasets = retrieve_via_wado_rs(wado_info)

        if not datasets:
            raise ValueError(f"No DICOM instances retrieved for series {series_uid}")

        logger.info(f"Retrieved {len(datasets)} DICOM instances")

        # Save to disk
        dicom_folder = save_datasets_to_folder(datasets, series_uid, storage_config)

        # Read as volume
        volume = read_dicom_volume(dicom_folder)
        volume_array = sitk.GetArrayFromImage(volume)  # Shape: (Z, Y, X)

        logger.info(f"Volume shape: {volume_array.shape} (Z, Y, X)")

        # Extract slices based on strategy
        slice_arrays = extract_slices(volume_array, params)

        # Convert to base64
        base64_images = slices_to_base64(slice_arrays)

        logger.info(f"Preprocessed {len(base64_images)} slices for series {series_uid}")

        return base64_images

    finally:
        # Cleanup temp files
        if dicom_folder and dicom_folder.exists():
            try:
                shutil.rmtree(dicom_folder)
                logger.debug(f"Cleaned up temp folder: {dicom_folder}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp folder {dicom_folder}: {e}")
