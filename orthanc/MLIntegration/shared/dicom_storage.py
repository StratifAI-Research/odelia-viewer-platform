"""
DICOM file storage utilities
Single Responsibility: File system operations for DICOM data
"""
import os
import shutil
import logging
from pathlib import Path
from typing import List
from pydicom.dataset import Dataset

from .config import StorageConfig

logger = logging.getLogger(__name__)


def create_series_folder(series_uid: str, storage_config: StorageConfig, clean: bool = True) -> Path:
    """
    Create a folder for storing DICOM series

    Args:
        series_uid: DICOM SeriesInstanceUID
        storage_config: StorageConfig with base image folder
        clean: If True, remove existing folder before creating new one

    Returns:
        Path to created series folder
    """
    series_folder = Path(storage_config.image_folder) / series_uid

    # Clean up if exists and clean=True
    if series_folder.exists() and clean:
        logger.info(f"Removing existing series folder: {series_folder}")
        shutil.rmtree(series_folder)

    os.makedirs(series_folder, exist_ok=True)
    logger.info(f"Created series folder: {series_folder}")

    return series_folder


def save_datasets_to_folder(datasets: List[Dataset], series_uid: str, storage_config: StorageConfig) -> Path:
    """
    Save DICOM datasets to a folder on disk

    Args:
        datasets: List of DICOM datasets to save
        series_uid: DICOM SeriesInstanceUID (used for folder name)
        storage_config: StorageConfig with base image folder

    Returns:
        Path to folder containing saved DICOM files
    """
    if not datasets:
        raise ValueError("No datasets provided to save")

    # Create series folder
    series_folder = create_series_folder(series_uid, storage_config, clean=True)

    # Save each dataset as a DICOM file
    for idx, ds in enumerate(datasets):
        dicom_path = series_folder / f"instance_{idx:04d}.dcm"
        ds.save_as(dicom_path)

    logger.info(f"Saved {len(datasets)} DICOM files to {series_folder}")
    return series_folder


def save_dicom_bytes_to_folder(dicom_files: List[bytes], series_uid: str, storage_config: StorageConfig) -> Path:
    """
    Save DICOM file bytes to a folder on disk

    Args:
        dicom_files: List of DICOM file contents as bytes
        series_uid: DICOM SeriesInstanceUID (used for folder name)
        storage_config: StorageConfig with base image folder

    Returns:
        Path to folder containing saved DICOM files
    """
    if not dicom_files:
        raise ValueError("No DICOM files provided to save")

    # Create series folder
    series_folder = create_series_folder(series_uid, storage_config, clean=True)

    # Save each DICOM file
    for idx, dicom_data in enumerate(dicom_files):
        dicom_path = series_folder / f"instance_{idx:04d}.dcm"
        with open(dicom_path, "wb") as f:
            f.write(dicom_data)

    logger.info(f"Saved {len(dicom_files)} DICOM files to {series_folder}")
    return series_folder
