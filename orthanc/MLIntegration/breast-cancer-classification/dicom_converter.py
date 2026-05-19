"""
DICOM to NIfTI conversion wrapper
Single Responsibility: Provide clean interface for DICOM conversion
"""
import logging
from pathlib import Path

from dicom2nfti_onthefly import dicom_to_unilateral_nifti

logger = logging.getLogger(__name__)


def convert_to_unilateral_nifti(dicom_folder: Path) -> dict:
    """
    Convert DICOM series to unilateral NIfTI format

    Wraps the existing dicom2nfti_onthefly.dicom_to_unilateral_nifti function
    with proper type handling and error reporting.

    Args:
        dicom_folder: Path to folder containing DICOM files

    Returns:
        Dictionary of NIfTI images keyed by side and sequence name
        (e.g., 'Pre_left', 'Post_1_right')

    Raises:
        ValueError: If conversion fails
    """
    logger.info(f"Converting DICOM series to unilateral NIfTI: {dicom_folder}")

    try:
        nifties = dicom_to_unilateral_nifti(dicom_folder)

        if not nifties:
            raise ValueError("No NIfTI images were created")

        logger.info(f"Successfully converted to {len(nifties)} unilateral NIfTI images")
        logger.info(f"Available images: {list(nifties.keys())}")

        return nifties

    except Exception as e:
        logger.error(f"DICOM to NIfTI conversion failed: {e}")
        raise ValueError(f"Conversion failed: {e}") from e
