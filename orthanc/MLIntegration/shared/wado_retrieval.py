"""
WADO-RS retrieval helper for AI models
Single Responsibility: Retrieve DICOM instances via DICOMweb WADO-RS protocol
"""
import logging
from pydicom.dataset import Dataset
from typing import List
from dicomweb_client.api import DICOMwebClient

from .exceptions import DicomRetrievalError

logger = logging.getLogger(__name__)


def retrieve_via_wado_rs(wado_rs_retrieval: List[dict]) -> List[Dataset]:
    """
    Retrieve DICOM instances via WADO-RS using dicomweb-client

    Args:
        wado_rs_retrieval: List of dicts with:
            - retrieval_url: Full WADO-RS URL or base DICOMweb URL
            - study_uid: StudyInstanceUID
            - series_uid: SeriesInstanceUID

    Returns:
        List of DICOM datasets (pydicom.Dataset instances)

    Raises:
        DicomRetrievalError: If retrieval fails
    """
    all_datasets = []

    for retrieval_info in wado_rs_retrieval:
        retrieval_url = retrieval_info.get("retrieval_url", "")
        study_uid = retrieval_info.get("study_uid", "")
        series_uid = retrieval_info.get("series_uid", "")

        # Extract base URL from retrieval_url if it's a full WADO-RS URL
        # Expected format: http://host/dicom-web/studies/{study}/series/{series}
        base_url = retrieval_url
        if "/studies/" in retrieval_url:
            base_url = retrieval_url.split("/studies/")[0]

        logger.info(f"Retrieving series {series_uid} via WADO-RS from {base_url}")

        try:
            # Create DICOMweb client
            client = DICOMwebClient(url=base_url)

            # Retrieve all instances in the series
            # Returns List[pydicom.Dataset]
            datasets = client.retrieve_series(
                study_instance_uid=study_uid,
                series_instance_uid=series_uid
            )

            logger.info(f"Retrieved {len(datasets)} instances for series {series_uid}")
            all_datasets.extend(datasets)

        except Exception as e:
            logger.error(f"Error retrieving via WADO-RS: {str(e)}")
            import traceback
            traceback.print_exc()
            raise DicomRetrievalError(f"WADO-RS retrieval failed: {str(e)}") from e

    return all_datasets
