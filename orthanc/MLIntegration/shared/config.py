"""
Shared configuration classes for ML Integration services
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class StorageConfig:
    """Configuration for DICOM file storage"""
    image_folder: Path
    cleanup_on_start: bool = True
