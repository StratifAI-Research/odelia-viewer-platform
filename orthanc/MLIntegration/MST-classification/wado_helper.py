"""
WADO-RS retrieval helper for AI models
Retrieves DICOM instances via DICOMweb WADO-RS protocol
"""

import io
import requests
from email import message_from_bytes
from pydicom import dcmread


def parse_multipart_dicom(response_content, boundary):
    """
    Parse multipart/related response containing DICOM instances

    Args:
        response_content: Raw bytes from WADO-RS response
        boundary: Boundary string from Content-Type header

    Returns:
        List of DICOM datasets
    """
    # Parse multipart message
    msg = message_from_bytes(
        b'Content-Type: multipart/related; boundary="' + boundary.encode() + b'"\r\n\r\n' + response_content
    )

    dicom_datasets = []

    if msg.is_multipart():
        for part in msg.get_payload():
            # Get the content type of this part
            content_type = part.get_content_type()

            # Check if this part contains DICOM data
            if 'application/dicom' in content_type:
                # Get the binary content
                part_content = part.get_payload(decode=True)

                # Parse DICOM
                try:
                    ds = dcmread(io.BytesIO(part_content))
                    dicom_datasets.append(ds)
                except Exception as e:
                    print(f"Error parsing DICOM part: {str(e)}")

    return dicom_datasets


def retrieve_via_wado_rs(wado_rs_retrieval, orthanc_url=None):
    """
    Retrieve DICOM instances via WADO-RS

    Args:
        wado_rs_retrieval: List of dicts with:
            - retrieval_url: Full WADO-RS URL
            - study_uid: StudyInstanceUID
            - series_uid: SeriesInstanceUID
        orthanc_url: Optional Orthanc URL for authentication token (not used for WADO-RS)

    Returns:
        List of DICOM datasets
    """
    all_datasets = []

    for retrieval_info in wado_rs_retrieval:
        retrieval_url = retrieval_info["retrieval_url"]
        series_uid = retrieval_info["series_uid"]

        print(f"Retrieving series {series_uid} via WADO-RS from {retrieval_url}")

        try:
            # WADO-RS request with proper Accept header
            response = requests.get(
                retrieval_url,
                headers={
                    "Accept": "multipart/related; type=application/dicom; transfer-syntax=*"
                },
                timeout=300  # 5 minutes timeout for large series
            )

            if response.status_code != 200:
                print(f"Error retrieving via WADO-RS: {response.status_code} - {response.text}")
                continue

            # Extract boundary from Content-Type header
            content_type = response.headers.get('Content-Type', '')
            boundary = None
            if 'boundary=' in content_type:
                # Properly parse boundary by splitting on semicolons first
                for part in content_type.split(';'):
                    part = part.strip()
                    if part.startswith('boundary='):
                        boundary = part.split('boundary=')[1].strip().strip('"')
                        break

            if not boundary:
                print(f"No boundary found in Content-Type: {content_type}")
                continue

            # Parse multipart response
            datasets = parse_multipart_dicom(response.content, boundary)
            print(f"Retrieved {len(datasets)} instances for series {series_uid}")
            all_datasets.extend(datasets)

        except Exception as e:
            print(f"Error retrieving via WADO-RS: {str(e)}")
            import traceback
            traceback.print_exc()

    return all_datasets


def fallback_to_orthanc_rest(series_uid, orthanc_url):
    """
    Fallback: Retrieve DICOM instances via Orthanc REST API
    Used for backward compatibility when WADO-RS retrieval fails

    Args:
        series_uid: SeriesInstanceUID
        orthanc_url: Orthanc base URL (e.g., http://orthanc-viewer:8042)

    Returns:
        List of DICOM datasets
    """
    print(f"Falling back to Orthanc REST API for series {series_uid}")

    try:
        # Lookup series ID from UID
        lookup_response = requests.post(
            f"{orthanc_url}/tools/lookup",
            data=series_uid,
            timeout=30
        )

        if lookup_response.status_code != 200:
            print(f"Error looking up series: {lookup_response.status_code}")
            return []

        lookup_result = lookup_response.json()
        series_result = [r for r in lookup_result if r["Type"] == "Series"]

        if not series_result:
            print(f"Series {series_uid} not found in Orthanc")
            return []

        series_id = series_result[0]["ID"]

        # Get instances
        instances_response = requests.get(
            f"{orthanc_url}/series/{series_id}/instances",
            timeout=30
        )

        if instances_response.status_code != 200:
            print(f"Error getting instances: {instances_response.status_code}")
            return []

        instances = instances_response.json()

        # Download each instance
        datasets = []
        for instance in instances:
            instance_id = instance["ID"]
            dicom_response = requests.get(
                f"{orthanc_url}/instances/{instance_id}/file",
                timeout=60
            )

            if dicom_response.status_code == 200:
                ds = dcmread(io.BytesIO(dicom_response.content))
                datasets.append(ds)

        print(f"Retrieved {len(datasets)} instances via REST API fallback")
        return datasets

    except Exception as e:
        print(f"Error in REST API fallback: {str(e)}")
        import traceback
        traceback.print_exc()
        return []
