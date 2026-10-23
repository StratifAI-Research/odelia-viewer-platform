"""Per-model serving smoke for the default trained roster and optional preview.

Needs the model pairs up; no study required. Marked integration automatically by
tests/conftest.py, so CI (-m unit) does not run it.
"""

import os

import pytest
import requests

from ._roster import ROSTER, ROSTER_IDS

_HOST = os.environ.get("ROSTER_HOST", "http://localhost")


def _skip_if_down(url: str) -> requests.Response:
    try:
        return requests.get(url, timeout=5)
    except requests.RequestException as exc:
        pytest.skip(f"roster service not reachable at {url}: {exc}")


@pytest.mark.parametrize("model", ROSTER, ids=ROSTER_IDS)
def test_service_health_reports_its_model(model):
    r = _skip_if_down(f"{_HOST}:{model.backend_port}/health")
    assert r.status_code == 200, r.text[:300]

    health = r.json()
    assert health["status"] == "healthy"
    assert health["model_info"]["model_name"] == model.model_name
    if model.model_name != "Pimed":
        assert health["model_info"]["weights"] == "trained"
        expected_release = "2026-08" if model.model_name == "DivideAndConquer" else "2026-05"
        assert health["model_info"]["version"] == expected_release


@pytest.mark.parametrize("model", ROSTER, ids=ROSTER_IDS)
def test_router_serves_its_manifest(model):
    r = _skip_if_down(f"{_HOST}:{model.router_port}/manifest")
    assert r.status_code == 200, r.text[:300]

    manifest = r.json()
    assert manifest["input_configurations"], "manifest exposes no input configurations"


@pytest.mark.parametrize("model", ROSTER, ids=ROSTER_IDS)
def test_router_ups_rs_workitem_query_answers(model):
    r = _skip_if_down(f"{_HOST}:{model.router_port}/ups-rs/workitems")
    assert r.status_code == 200, r.text[:300]
    assert isinstance(r.json(), list)
