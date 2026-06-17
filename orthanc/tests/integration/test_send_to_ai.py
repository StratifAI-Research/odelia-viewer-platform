"""Integration coverage for the /send-to-ai DICOMweb routing endpoint.

Reproduces the production failure where routing a study to an AI model whose
display name contains spaces (e.g. "MST AI model") failed: first a 400 from an
over-strict server-name validator, then a 500 because the server was configured
via the core REST API which cannot reach the DICOMweb plugin route. Requires a
running orthanc-viewer with at least one study loaded.
"""

import pytest
import requests

# Real AI target as the frontend sends it (app-config.js aiEndpoints[].name).
MODEL_DISPLAY_NAME = "MST AI model"
ROUTER_DICOMWEB_URL = "http://orthanc-router-mst:8042/dicom-web"


@pytest.mark.integration
def test_send_to_ai_routes_spaced_model_name(base_url):
    studies = requests.get(f"{base_url}/studies", timeout=10).json()
    if not studies:
        pytest.skip("no study loaded in orthanc-viewer")
    study_id = studies[0]

    series = requests.get(f"{base_url}/studies/{study_id}/series", timeout=10).json()
    series_uids = [s["MainDicomTags"]["SeriesInstanceUID"] for s in series]
    assert series_uids, "study has no series"

    payload = {
        "study_id": study_id,
        "target": MODEL_DISPLAY_NAME,
        "target_url": ROUTER_DICOMWEB_URL,
        "series_uids": series_uids,
        "input_mapping": {"Multi-phase Series": series_uids[0]},
    }
    r = requests.post(f"{base_url}/send-to-ai", json=payload, timeout=120)

    # The two regressions this guards against:
    assert r.status_code != 400, f"server-name validation rejected a real model name: {r.text[:300]}"
    assert "Unknown resource" not in r.text and "Error configuring DICOMweb server" not in r.text, (
        f"DICOMweb server config failed (core-vs-plugin REST API): {r.text[:300]}"
    )
    assert r.status_code == 200, f"expected 200, got {r.status_code}: {r.text[:300]}"
