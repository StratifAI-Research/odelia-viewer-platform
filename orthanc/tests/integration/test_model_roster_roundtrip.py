"""Per-model send-to-AI round-trip: viewer -> router -> service -> SR (ODV-219).

Proves the acceptance criterion that a routed study produces an SR back in the
viewer's Orthanc. Assertions are structural: weights are init-only (ODV-216), so
prediction values carry no information.
"""

import io
import os
import time

import pytest
import requests
from pydicom import dcmread

from ._roster import ROSTER, ROSTER_IDS

_HOST = os.environ.get("ROSTER_HOST", "http://localhost")
_CLASS_NAMES = {"No lesion", "Benign", "Malignant"}
_STATE_TAG = "00741000"  # Procedure Step State
_TIMEOUT_S = int(os.environ.get("ROUNDTRIP_TIMEOUT_S", "900"))


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
    instances = requests.get(f"{base_url}/studies/{study_id}/instances", timeout=30).json()
    return {
        i["ID"]
        for i in instances
        if i.get("MainDicomTags", {})
        .get("SOPClassUID", "")
        .startswith("1.2.840.10008.5.1.4.1.1.88")
    }


@pytest.fixture(scope="module")
def study(base_url):
    """A study plus its largest MR series -- the multiphase input the model needs.

    The viewer study also holds SC heatmaps and SRs from earlier AI runs, so the
    input series must be chosen by modality, never by position.
    """
    studies = requests.get(f"{base_url}/studies", timeout=10).json()
    if not studies:
        pytest.skip("no study loaded in orthanc-viewer")
    study_id = studies[0]

    series = requests.get(f"{base_url}/studies/{study_id}/series", timeout=10).json()
    mr = [s for s in series if s["MainDicomTags"].get("Modality") == "MR"]
    if not mr:
        pytest.skip("study has no MR series to route")

    mr.sort(key=lambda s: len(s.get("Instances", [])), reverse=True)
    input_uid = mr[0]["MainDicomTags"]["SeriesInstanceUID"]
    return study_id, input_uid


@pytest.mark.parametrize("model", ROSTER, ids=ROSTER_IDS)
def test_send_to_ai_produces_sr(base_url, study, model):
    _skip_if_down(model)
    study_id, input_uid = study
    before_srs = _sr_instance_ids(base_url, study_id)

    payload = {
        "study_id": study_id,
        "target": model.ai_name,
        "target_url": f"http://{model.router_host}:8042/dicom-web",
        "series_uids": [input_uid],
        "input_mapping": {"Multi-phase Series": input_uid},
    }
    r = requests.post(f"{base_url}/send-to-ai", json=payload, timeout=120)
    assert r.status_code == 200, f"send-to-ai failed: {r.text[:300]}"

    workitem_uid = r.json().get("workitem_uid")
    assert workitem_uid, f"send-to-ai returned no workitem_uid: {r.text[:300]}"

    deadline = time.time() + _TIMEOUT_S
    new_srs: set[str] = set()
    final_state = None
    while time.time() < deadline:
        final_state = _workitem_state(model.router_port, workitem_uid)
        if final_state not in ("SCHEDULED", "IN PROGRESS", None):
            new_srs = _sr_instance_ids(base_url, study_id) - before_srs
            break
        time.sleep(5)

    assert final_state is not None, "router never reported a state for the workitem"
    assert final_state == "COMPLETED", (
        f"workitem ended {final_state}, not COMPLETED "
        "(a CANCELED here is the Unknown-response-format regression)"
    )
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
        assert item.ConceptCodeSequence[0].CodeMeaning in _CLASS_NAMES
        confidence = float(item.MeasuredValueSequence[0].NumericValue)
        assert 0.0 <= confidence <= 100.0

    algorithm = items[-1]
    assert algorithm.TextValue == model.model_name
