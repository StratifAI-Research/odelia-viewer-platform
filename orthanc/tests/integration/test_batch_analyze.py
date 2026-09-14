"""End-to-end batch send-to-AI over sample_data (ODV-221 acceptance).

Uploads the bundled MR study to the viewer's Orthanc and runs the batch tool
against the live MST roster pair, asserting an SR is produced. Requires the
viewer stack and the odelia-models MST pair; skips otherwise.
"""

import os
from pathlib import Path

import pytest
import requests

from batch.cli import discover_files, resolve_models
from batch.client import OrthancRouterClient
from batch.pipeline import run_batch
from batch.roster import ROSTER

pytestmark = pytest.mark.integration

_HOST = os.environ.get("ROSTER_HOST", "http://localhost")
_SAMPLE_MRI = Path(__file__).resolve().parents[2] / "sample_data" / "mri"


def _mst_model():
    return next(m for m in ROSTER if m.model_name == "MST")


def _skip_if_pair_down(model) -> None:
    try:
        requests.get(f"{_HOST}:{model.backend_port}/health", timeout=5)
        requests.get(f"{_HOST}:{model.router_port}/ups-rs/workitems", timeout=5)
    except requests.RequestException as exc:
        pytest.skip(f"MST pair not running: {exc}")


def test_batch_preloads_sr_for_sample_study(base_url):
    model = _mst_model()
    _skip_if_pair_down(model)
    if not _SAMPLE_MRI.is_dir():
        pytest.skip(f"sample MRI not found at {_SAMPLE_MRI}")

    client = OrthancRouterClient(base_url, roster_host=_HOST)
    files = discover_files(_SAMPLE_MRI)
    specs = resolve_models(["MST"])

    report = run_batch(client, files, specs, poll_timeout_s=900, poll_interval_s=5)

    assert report.uploaded_files == len(files)
    assert len(report.studies) == 1
    result = report.studies[0].model_results[0]
    assert result.final_state == "COMPLETED", result.error
    assert result.created, result.error
    assert result.new_sr_ids
