"""
MedGemma-specific preprocessing: DICOM to PIL Image conversion
Single Responsibility: Extract representative slices from MRI volumes
"""
import logging
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from PIL import Image
from typing import List
from collections import defaultdict

logger = logging.getLogger(__name__)


def get_dicom_metadata(dicom_file: str) -> tuple:
    """
    Extract temporal position and spatial position from DICOM file using SimpleITK.

    Returns:
        (temporal_position, slice_location, instance_number)
    """
    try:
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(dicom_file))
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()

        # Get temporal position
        temporal_pos = 0
        if reader.HasMetaDataKey("0020|0100"):  # TemporalPositionIdentifier
            temporal_pos = int(reader.GetMetaData("0020|0100"))
        elif reader.HasMetaDataKey("0018|1060"):  # TriggerTime
            temporal_pos = int(float(reader.GetMetaData("0018|1060")))

        # Get spatial position for sorting
        slice_location = 0.0
        if reader.HasMetaDataKey("0020|1041"):  # SliceLocation
            slice_location = float(reader.GetMetaData("0020|1041"))
        elif reader.HasMetaDataKey("0020|0032"):  # ImagePositionPatient
            position = reader.GetMetaData("0020|0032")
            slice_location = float(position.split("\\")[2])

        # Get instance number as fallback
        instance_number = 0
        if reader.HasMetaDataKey("0020|0013"):  # InstanceNumber
            instance_number = int(reader.GetMetaData("0020|0013"))

        return (temporal_pos, slice_location, instance_number)

    except Exception as e:
        logger.debug(f"Could not read metadata from {dicom_file}: {e}")
        return (0, 0.0, 0)


def read_dicom_volume(dicom_folder: Path) -> sitk.Image:
    """
    Read DICOM series as a 3D volume using SimpleITK.
    Handles 4D temporal series by extracting the first temporal phase.

    Args:
        dicom_folder: Path to folder containing DICOM files

    Returns:
        SimpleITK Image object
    """
    dicom_path = Path(dicom_folder)

    # Verify DICOM files exist
    dicom_files = list(dicom_path.glob("*.dcm"))
    if not dicom_files:
        raise ValueError(f"No DICOM files found in {dicom_folder}")

    logger.info(f"Found {len(dicom_files)} DICOM files")

    # Extract metadata for all files and group by temporal position
    file_metadata = []
    for dcm_file in dicom_files:
        temporal_pos, slice_loc, instance_num = get_dicom_metadata(str(dcm_file))
        file_metadata.append({
            'path': str(dcm_file),
            'temporal_pos': temporal_pos,
            'slice_location': slice_loc,
            'instance_number': instance_num
        })

    # Group by temporal position
    temporal_groups = defaultdict(list)
    for item in file_metadata:
        temporal_groups[item['temporal_pos']].append(item)

    num_temporal_phases = len(temporal_groups)
    logger.info(f"Detected {num_temporal_phases} temporal phase(s): {sorted(temporal_groups.keys())}")

    # Determine which files to use
    if num_temporal_phases == 1:
        # Simple 3D series - use standard GDCM series reader
        logger.info("Single temporal phase detected - converting all files")
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_path))

        if not dicom_names:
            raise ValueError(f"No valid DICOM series found in {dicom_folder}")

        reader.SetFileNames(dicom_names)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        image = reader.Execute()

    else:
        # 4D series - extract first temporal position
        sorted_positions = sorted(temporal_groups.keys())
        selected_key = sorted_positions[0]
        selected_files_metadata = temporal_groups[selected_key]

        # Sort files by spatial position
        selected_files_metadata.sort(key=lambda x: (x['slice_location'], x['instance_number']), reverse=True)
        selected_files = [item['path'] for item in selected_files_metadata]

        logger.info(
            f"4D temporal series detected. Extracting first temporal position "
            f"(key={selected_key}, {len(selected_files)} slices)"
        )

        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(selected_files)
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


def extract_central_slices(dicom_folder: Path, num_slices: int = 5) -> List[Image.Image]:
    """
    Extract evenly-spaced slices from the central portion of MRI volume.

    Args:
        dicom_folder: Path to DICOM files
        num_slices: Number of slices to extract

    Returns:
        List of PIL Images (normalized, grayscale converted to RGB)
    """
    # Read DICOM volume
    volume = read_dicom_volume(dicom_folder)

    # Convert to numpy array
    volume_array = sitk.GetArrayFromImage(volume)  # Shape: (Z, Y, X)

    total_slices = volume_array.shape[0]
    logger.info(f"Volume shape: {volume_array.shape} (Z, Y, X)")

    if total_slices < num_slices:
        logger.warning(f"Volume has only {total_slices} slices, requested {num_slices}")
        num_slices = total_slices

    # Select slices from central 60% of volume
    start_pct = 0.2
    end_pct = 0.8
    start_idx = int(total_slices * start_pct)
    end_idx = int(total_slices * end_pct)

    # Ensure we have enough range
    if end_idx - start_idx < num_slices:
        start_idx = 0
        end_idx = total_slices

    # Calculate evenly-spaced indices
    if num_slices == 1:
        indices = [(start_idx + end_idx) // 2]
    else:
        step = (end_idx - start_idx - 1) / (num_slices - 1)
        indices = [int(start_idx + i * step) for i in range(num_slices)]

    logger.info(f"Extracting slices at indices: {indices} (from central 60% of {total_slices} total)")

    # Extract and convert slices to PIL Images
    pil_images = []
    for idx in indices:
        slice_array = volume_array[idx, :, :]

        # Normalize to 0-255
        normalized = normalize_slice(slice_array)

        # Convert grayscale to RGB (MedGemma expects RGB)
        rgb_array = np.stack([normalized, normalized, normalized], axis=-1)

        # Create PIL Image
        pil_image = Image.fromarray(rgb_array, mode='RGB')
        pil_images.append(pil_image)

        logger.debug(f"Extracted slice {idx}: shape={slice_array.shape}, "
                    f"range=[{slice_array.min():.1f}, {slice_array.max():.1f}]")

    logger.info(f"Extracted {len(pil_images)} PIL images from volume")

    return pil_images
