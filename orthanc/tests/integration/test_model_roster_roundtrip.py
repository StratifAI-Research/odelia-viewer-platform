"""Per-model send-to-AI round-trip: viewer -> router -> service -> SR (ODV-219).

Proves the acceptance criterion that a routed study produces an SR back in the
viewer's Orthanc -- once via the legacy flat path and once via the manifest-driven
multiphase configuration. Assertions are structural: weights are init-only (ODV-216), so
prediction values carry no information.
"""

import io
import os
import time

import pytest
import requests
from pydicom import dcmread

from batch.roster import ROSTER, ROSTER_IDS

_HOST = os.environ.get("ROSTER_HOST", "http://localhost")
_SR_CODE_MEANINGS = {"Malignant", "Benign", "Clinical finding absent"}
_STATE_TAG = "00741000"  # Procedure Step State
_TIMEOUT_S = int(os.environ.get("ROUNDTRIP_TIMEOUT_S", "900"))
_IN_FLIGHT = {"SCHEDULED", "IN PROGRESS", "IN_PROGRESS"}
_SR_UPLOAD_TIMEOUT_S = 30


def _skip_if_down(model) -> None:
    """Skip models whose pair is not running, so a partial roster is runnable."""
    try:
        requests.get(f"{_HOST}:{model.backend_port}/health", timeout=5)
        requests.get(f"{_HOST}:{model.router_port}/ups-rs/workitems", timeout=5)
    except requests.RequestException as exc:
        pytest.skip(f"{model.model_name} pair not running: {exc}")


def _workitem_state(router_port: int, workitem_uid: str) -> str | None:
    """Read one workitem's Procedure Step State from the router."""
    r = requests.get(f"{_HOST}:{router_port}/ups-rs/workitems/{workitem_uid}", timeout=10)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json().get(_STATE_TAG, {}).get("Value", [None])[0]


def _sr_instance_ids(base_url: str, study_id: str) -> set[str]:
    """Instance IDs of SR series in the study (Modality lives at series level)."""
    series = requests.get(f"{base_url}/studies/{study_id}/series", timeout=30).json()
    return {
        inst
        for s in series
        if s["MainDicomTags"].get("Modality") == "SR"
        for inst in s.get("Instances", [])
    }


@pytest.fixture(scope="module")
def study(base_url):
    """A study plus its largest MR series -- the multiphase input the model needs.

    The viewer study also holds SC heatmaps and SRs from earlier AI runs, so the
    input series must be chosen by modality, never by position. Studies whose
    PatientName starts with SYNTH are QA fixtures with degenerate intensities
    (two-valued subtractions), which the MediSwarm ZNormalization mask rejects.
    """
    studies = requests.get(f"{base_url}/studies", timeout=10).json()
    real = []
    for sid in studies:
        try:
            meta = requests.get(f"{base_url}/studies/{sid}", timeout=10).json()
        except requests.RequestException as exc:
            pytest.skip(f"could not read study {sid}: {exc}")
        name = meta.get("PatientMainDicomTags", {}).get("PatientName", "")
        if not name.startswith("SYNTH"):
            real.append(sid)
    if not real:
        pytest.skip("no non-synthetic study loaded in orthanc-viewer")
    study_id = real[0]

    series = requests.get(f"{base_url}/studies/{study_id}/series", timeout=10).json()
    mr = [s for s in series if s["MainDicomTags"].get("Modality") == "MR"]
    if not mr:
        pytest.skip("study has no MR series to route")

    mr.sort(key=lambda s: len(s.get("Instances", [])), reverse=True)
    input_uid = mr[0]["MainDicomTags"]["SeriesInstanceUID"]
    return study_id, input_uid


@pytest.mark.parametrize("model", ROSTER, ids=ROSTER_IDS)
@pytest.mark.parametrize("input_mode", ["flat", "multiphase"])
def test_send_to_ai_produces_sr(base_url, study, model, input_mode):
    _skip_if_down(model)
    study_id, input_uid = study
    before_srs = _sr_instance_ids(base_url, study_id)

    payload = {
        "study_id": study_id,
        "target": model.ai_name,
        "target_url": f"http://{model.router_host}:8042/dicom-web",
        "series_uids": [input_uid],
    }
    if input_mode == "multiphase":
        # Manifest-driven path: config id + role key from manifest.json. The
        # viewer API takes bare series UIDs; the router processor builds the
        # per-role WADO-RS dicts itself.
        payload["input_configuration_id"] = "multiphase"
        payload["input_mapping"] = {"multiphase": input_uid}
    r = requests.post(f"{base_url}/send-to-ai", json=payload, timeout=120)
    assert r.status_code == 200, f"send-to-ai failed: {r.text[:300]}"

    workitem_uid = r.json().get("workitem_uid")
    assert workitem_uid, f"send-to-ai returned no workitem_uid: {r.text[:300]}"

    deadline = time.time() + _TIMEOUT_S
    new_srs: set[str] = set()
    final_state = None
    while time.time() < deadline:
        final_state = _workitem_state(model.router_port, workitem_uid)
        if final_state is not None and final_state not in _IN_FLIGHT:
            new_srs = _sr_instance_ids(base_url, study_id) - before_srs
            break
        time.sleep(5)

    assert final_state is not None, "router never reported a state for the workitem"
    assert final_state == "COMPLETED", (
        f"workitem ended {final_state}, not COMPLETED "
        "(CANCELED means the model backend failed or rejected the request -- "
        "check its logs; known causes: the flat-path Unknown-response-format "
        "regression, or multiphase preprocessing rejecting the input series)"
    )

    sr_deadline = time.time() + _SR_UPLOAD_TIMEOUT_S
    while not new_srs and time.time() < sr_deadline:
        time.sleep(5)
        new_srs = _sr_instance_ids(base_url, study_id) - before_srs

    assert new_srs, "workitem completed but no SR was written back to the viewer"

    sr_bytes = requests.get(f"{base_url}/instances/{sorted(new_srs)[-1]}/file", timeout=30).content
    sr = dcmread(io.BytesIO(sr_bytes))

    assert sr.Modality == "SR"
    assert sr.StudyInstanceUID
    assert sr.ReferencedImageSequence, "SR does not reference the source images"

    items = sr.ContentSequence[0].ContentSequence
    sides = {
        item.ConceptNameCodeSequence[0].CodeMeaning: item
        for item in items
        if "Side Probability" in item.ConceptNameCodeSequence[0].CodeMeaning
    }
    assert set(sides) == {"Left Side Probability", "Right Side Probability"}

    for item in sides.values():
        assert item.ConceptCodeSequence[0].CodeMeaning in _SR_CODE_MEANINGS
        confidence = float(item.MeasuredValueSequence[0].NumericValue)
        assert 0.0 <= confidence <= 100.0

    algorithm = items[-1]
    assert algorithm.TextValue in (model.model_name, f"{model.model_name} (untrained)")
