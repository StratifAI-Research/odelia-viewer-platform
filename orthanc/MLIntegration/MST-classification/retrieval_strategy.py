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

        if len(self.wado_rs_retrieval) > 1:
            logger.warning(f"Multiple series detected ({len(self.wado_rs_retrieval)}). Only the first series will be processed.")
            logger.warning("TODO: Add multi-series processing support")

        # Process only the first series
        first_series = [self.wado_rs_retrieval[0]]
        series_uid = first_series[0].get("series_uid", "unknown")

        # Use shared.wado_retrieval to retrieve DICOM datasets
        logger.info(f"Retrieving series {series_uid} via WADO-RS")
        datasets = retrieve_via_wado_rs(first_series)

        if not datasets:
            raise ValueError(f"No DICOM instances retrieved for series {series_uid}")

        logger.info(f"Retrieved {len(datasets)} DICOM instances")

        # Save datasets to disk
        dicom_folder = save_datasets_to_folder(datasets, series_uid, self.storage_config)

        return dicom_folder, series_uid
