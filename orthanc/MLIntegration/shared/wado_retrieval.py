"""
WADO-RS retrieval helper for AI models
Single Responsibility: Retrieve DICOM instances via DICOMweb WADO-RS protocol
"""

import logging
import os
from urllib.parse import urlsplit

import requests
from dicomweb_client.api import DICOMwebClient
from pydicom.dataset import Dataset

from .exceptions import DicomRetrievalError

logger = logging.getLogger(__name__)


def retrieve_via_wado_rs(wado_rs_retrieval: list[dict[str, str]]) -> list[Dataset]:
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
            raw_allowlist = os.getenv("ROUTER_HOST_ALLOWLIST", "").strip()
            if raw_allowlist:
                parsed = urlsplit(base_url)
                allowed = {
                    host.strip().lower() for host in raw_allowlist.split(",") if host.strip()
                }
                if (
                    parsed.scheme not in {"http", "https"}
                    or parsed.hostname not in allowed
                    or parsed.username is not None
                    or parsed.password is not None
                    or parsed.fragment
                    or any(c.isspace() for c in base_url)
                    or "\\" in base_url
                    or (parsed.port is not None and not 1 <= parsed.port <= 65535)
                ):
                    raise ValueError("WADO URL not allowed by ROUTER_HOST_ALLOWLIST")
                session = requests.Session()
                session.max_redirects = 0
                client = DICOMwebClient(url=base_url, session=session)
            else:
                client = DICOMwebClient(url=base_url)

            # Retrieve all instances in the series
            # Returns List[pydicom.Dataset]
            datasets = client.retrieve_series(
                study_instance_uid=study_uid, series_instance_uid=series_uid
            )

            logger.info(f"Retrieved {len(datasets)} instances for series {series_uid}")
            all_datasets.extend(datasets)

        except Exception as e:
            logger.error(f"Error retrieving via WADO-RS: {e!s}")
            import traceback

            traceback.print_exc()
            raise DicomRetrievalError(f"WADO-RS retrieval failed: {e!s}") from e

    return all_datasets
