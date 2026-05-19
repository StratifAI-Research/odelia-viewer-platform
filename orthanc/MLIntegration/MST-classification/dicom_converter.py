"""
DICOM to NIfTI conversion wrapper
Single Responsibility: Provide clean interface for DICOM conversion
"""
import logging
from pathlib import Path

from dicom_utils import dicom_to_nifti, dicom_to_nifti_subtraction, compute_subtraction_from_nifti

logger = logging.getLogger(__name__)


def convert_series_to_nifti(dicom_folder: Path) -> Path:
    """
    Convert DICOM series to NIfTI format

    Wraps the existing dicom_utils.dicom_to_nifti function with
    proper type handling and error reporting.

    Args:
        dicom_folder: Path to folder containing DICOM files

    Returns:
        Path to created NIfTI file

    Raises:
        ValueError: If conversion fails
    """
    logger.info(f"Converting DICOM series to NIfTI: {dicom_folder}")

    try:
        nifti_path = dicom_to_nifti(str(dicom_folder))
        nifti_path = Path(nifti_path)

        if not nifti_path.exists():
            raise ValueError(f"NIfTI file was not created: {nifti_path}")

        logger.info(f"Successfully converted to NIfTI: {nifti_path}")
        return nifti_path

    except Exception as e:
        logger.error(f"DICOM to NIfTI conversion failed: {e}")
        raise ValueError(f"Conversion failed: {e}") from e


def convert_multiphase_to_subtraction_nifti(dicom_folder: Path) -> Path:
    """
    Convert a multi-phase DICOM series to a subtraction NIfTI.
    Extracts the first two temporal groups and computes (group1 - group0).

    Args:
        dicom_folder: Path to folder containing multi-phase DICOM files

    Returns:
        Path to created subtraction NIfTI file

    Raises:
        ValueError: If conversion fails or fewer than 2 temporal phases
    """
    logger.info(f"Converting multi-phase DICOM to subtraction NIfTI: {dicom_folder}")

    try:
        nifti_path = dicom_to_nifti_subtraction(str(dicom_folder))
        nifti_path = Path(nifti_path)

        if not nifti_path.exists():
            raise ValueError(f"Subtraction NIfTI was not created: {nifti_path}")

        logger.info(f"Successfully created subtraction NIfTI: {nifti_path}")
        return nifti_path

    except Exception as e:
        logger.error(f"Multi-phase subtraction conversion failed: {e}")
        raise ValueError(f"Multi-phase conversion failed: {e}") from e


def compute_subtraction_nifti(pre_nifti: Path, post_nifti: Path) -> Path:
    """
    Compute subtraction NIfTI from two separate pre/post NIfTI files.

    Args:
        pre_nifti: Path to pre-contrast NIfTI
        post_nifti: Path to post-contrast NIfTI

    Returns:
        Path to created subtraction NIfTI file

    Raises:
        ValueError: If computation fails
    """
    logger.info(f"Computing subtraction: {post_nifti} - {pre_nifti}")

    try:
        result_path = compute_subtraction_from_nifti(str(pre_nifti), str(post_nifti))
        result_path = Path(result_path)

        if not result_path.exists():
            raise ValueError(f"Subtraction NIfTI was not created: {result_path}")

        logger.info(f"Successfully created subtraction NIfTI: {result_path}")
        return result_path

    except Exception as e:
        logger.error(f"Subtraction computation failed: {e}")
        raise ValueError(f"Subtraction failed: {e}") from e
