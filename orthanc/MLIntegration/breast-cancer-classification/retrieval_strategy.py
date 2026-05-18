"""
Retrieval strategies for DICOM data
Implements Strategy pattern for different retrieval methods
"""
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

from shared.wado_retrieval import retrieve_via_wado_rs
from shared.dicom_storage import save_datasets_to_folder
from shared.config import StorageConfig

logger = logging.getLogger(__name__)


class RetrievalStrategy(ABC):
    """Abstract base class for DICOM retrieval strategies"""

    @abstractmethod
    def retrieve(self) -> Tuple[Path, str]:
        """
        Retrieve DICOM data

        Returns:
            Tuple of (dicom_folder_path, series_uid)
        """
        pass


class WadoRSRetrieval(RetrievalStrategy):
    """WADO-RS retrieval strategy"""

    def __init__(self, wado_rs_retrieval: list, storage_config: StorageConfig):
        """
        Initialize WADO-RS retrieval

        Args:
            wado_rs_retrieval: List of dicts with retrieval_url, study_uid, series_uid
            storage_config: Storage configuration
        """
        self.wado_rs_retrieval = wado_rs_retrieval
        self.storage_config = storage_config

    def retrieve(self) -> Tuple[Path, str]:
        """
        Retrieve DICOM via WADO-RS

        Returns:
            Tuple of (dicom_folder_path, series_uid)
        """
        logger.info("Using WADO-RS retrieval")

        # Retrieve DICOM datasets
        datasets = retrieve_via_wado_rs(self.wado_rs_retrieval)

        if not datasets:
            raise ValueError("No DICOM instances retrieved via WADO-RS")

        # Extract series UID from first dataset
        series_uid = str(datasets[0].SeriesInstanceUID)
        logger.info(f"Retrieved {len(datasets)} DICOM instances for series {series_uid}")

        # Save datasets to disk
        dicom_folder = save_datasets_to_folder(datasets, series_uid, self.storage_config)

        return dicom_folder, series_uid
