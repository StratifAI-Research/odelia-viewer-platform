"""
DICOM to NIfTI conversion utilities using SimpleITK
"""
import SimpleITK as sitk
import numpy as np
from pathlib import Path
import logging
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
        reader.ReadImageInformation()  # Read metadata without loading pixels

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
            # Use Z coordinate from ImagePositionPatient
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


def _gather_temporal_groups(dicom_folder: str) -> tuple:
    """
    Scan a DICOM folder and return (dicom_path, temporal_groups dict).

    Each temporal group maps temporal_position_key -> list of file metadata dicts.
    """
    dicom_path = Path(dicom_folder)
    dicom_files = list(dicom_path.glob("*.dcm"))
    if not dicom_files:
        raise ValueError(f"No DICOM files found in {dicom_folder}")

    logger.info(f"Found {len(dicom_files)} DICOM files")

    file_metadata = []
    for dcm_file in dicom_files:
        temporal_pos, slice_loc, instance_num = get_dicom_metadata(str(dcm_file))
        file_metadata.append({
            'path': str(dcm_file),
            'temporal_pos': temporal_pos,
            'slice_location': slice_loc,
            'instance_number': instance_num
        })

    temporal_groups = defaultdict(list)
    for item in file_metadata:
        temporal_groups[item['temporal_pos']].append(item)

    logger.info(f"Detected {len(temporal_groups)} temporal phase(s): {sorted(temporal_groups.keys())}")
    return dicom_path, temporal_groups


def _read_temporal_group(file_metadata_list: list) -> sitk.Image:
    """
    Read a single temporal group (list of file metadata dicts) into a SimpleITK Image.
    Files are sorted spatially (superior-to-inferior) to match GDCM default ordering.
    """
    sorted_meta = sorted(
        file_metadata_list,
        key=lambda x: (x['slice_location'], x['instance_number']),
        reverse=True
    )
    sorted_files = [item['path'] for item in sorted_meta]

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(sorted_files)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    return reader.Execute()


def _compute_subtraction_array(pre_image: sitk.Image, post_image: sitk.Image) -> np.ndarray:
    """
    Compute subtraction volume: (post - pre), shifted to non-negative origin
    (sub - sub.min()), cast to uint16.
    """
    dyn0 = sitk.GetArrayFromImage(pre_image)
    dyn1 = sitk.GetArrayFromImage(post_image)
    sub = dyn1.astype(np.float64) - dyn0.astype(np.float64)
    sub = sub - sub.min()
    return sub.astype(np.uint16)


def dicom_to_nifti(dicom_folder: str) -> str:
    """
    Convert DICOM series to NIfTI format using SimpleITK.
    Handles 4D temporal series by extracting the first temporal phase.

    Args:
        dicom_folder: Path to folder containing DICOM files

    Returns:
        Path to created NIfTI file
    """
    dicom_path, temporal_groups = _gather_temporal_groups(dicom_folder)

    if len(temporal_groups) == 1:
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
        sorted_positions = sorted(temporal_groups.keys())
        selected_key = sorted_positions[0]
        selected_meta = temporal_groups[selected_key]

        logger.info(
            f"4D temporal series detected. Extracting first temporal position "
            f"(key={selected_key}, {len(selected_meta)} slices)"
        )
        image = _read_temporal_group(selected_meta)

    logger.info(
        f"Read DICOM series: size: {image.GetSize()}, spacing: {image.GetSpacing()}, "
        f"direction: {image.GetDirection()}"
    )

    nifti_path = dicom_path / "mri_series.nii.gz"
    sitk.WriteImage(image, str(nifti_path))

    logger.info(f"NIfTI created: {nifti_path}")
    return str(nifti_path)


def dicom_to_nifti_subtraction(dicom_folder: str) -> str:
    """
    Convert a multi-phase DICOM series to a subtraction NIfTI.
    Extracts the first two temporal groups and computes (group1 - group0).

    Args:
        dicom_folder: Path to folder containing multi-phase DICOM files

    Returns:
        Path to created subtraction NIfTI file

    Raises:
        ValueError: If fewer than 2 temporal phases are found
    """
    dicom_path, temporal_groups = _gather_temporal_groups(dicom_folder)
    sorted_positions = sorted(temporal_groups.keys())

    if len(sorted_positions) < 2:
        raise ValueError(
            f"Multi-phase subtraction requires at least 2 temporal phases, "
            f"found {len(sorted_positions)}: {sorted_positions}"
        )

    pre_image = _read_temporal_group(temporal_groups[sorted_positions[0]])
    post_image = _read_temporal_group(temporal_groups[sorted_positions[1]])

    logger.info(
        f"Computing subtraction: phase {sorted_positions[1]} - phase {sorted_positions[0]} "
        f"({len(temporal_groups[sorted_positions[0]])} slices each)"
    )

    sub_array = _compute_subtraction_array(pre_image, post_image)

    result = sitk.GetImageFromArray(sub_array)
    result.CopyInformation(pre_image)

    nifti_path = dicom_path / "mri_subtraction.nii.gz"
    sitk.WriteImage(result, str(nifti_path))

    logger.info(f"Subtraction NIfTI created: {nifti_path}")
    return str(nifti_path)


def compute_subtraction_from_nifti(pre_path: str, post_path: str, output_path: str = None) -> str:
    """
    Compute subtraction NIfTI from two separate pre/post NIfTI files.
    Result = (post - pre), floored to 0, cast to uint16.
    Spatial metadata is preserved from the pre-contrast volume.

    Args:
        pre_path: Path to pre-contrast NIfTI
        post_path: Path to post-contrast NIfTI
        output_path: Optional explicit output path; defaults to sibling of post_path

    Returns:
        Path to created subtraction NIfTI file
    """
    pre_image = sitk.ReadImage(pre_path)
    post_image = sitk.ReadImage(post_path)

    logger.info(
        f"Computing subtraction from two NIfTI volumes: "
        f"pre={pre_image.GetSize()}, post={post_image.GetSize()}"
    )

    sub_array = _compute_subtraction_array(pre_image, post_image)

    result = sitk.GetImageFromArray(sub_array)
    result.CopyInformation(pre_image)

    if output_path is None:
        output_path = str(Path(post_path).parent / "mri_subtraction.nii.gz")

    sitk.WriteImage(result, output_path)
    logger.info(f"Subtraction NIfTI created: {output_path}")
    return output_path
