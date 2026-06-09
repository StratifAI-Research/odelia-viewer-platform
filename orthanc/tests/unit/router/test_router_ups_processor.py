"""Unit tests for router/ups/processor.py — notify and process_workitem.

Coverage:
  - notify_subscriber: happy path (200), HTTP error swallowed (non-200), network error swallowed
  - notify_all_subscribers: calls notify_subscriber for each registered subscriber
  - process_workitem: smoke test — at minimum transitions to IN_PROGRESS before heavy work
"""
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Iterator
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

import os as _os
_ROUTER_DIR = _os.path.abspath(
    _os.path.join(_os.path.dirname(__file__), '..', '..', '..', 'router')
)


def _ensure_router_path():
    """Ensure router/ is at front of sys.path so ups.* resolves to router/ups/."""
    if _ROUTER_DIR in sys.path:
        sys.path.remove(_ROUTER_DIR)
    sys.path.insert(0, _ROUTER_DIR)


def _evict_ups():
    _ensure_router_path()
    for key in [k for k in sys.modules if k == "ups" or k.startswith("ups.")]:
        del sys.modules[key]


def _make_workitem(uid="2.25.proc1"):
    _evict_ups()
    from ups.workitem import UPSWorkitem
    return UPSWorkitem(
        study_uid="1.2.3",
        series_uids=["1.2.3.1"],
        wado_rs_retrieval=[
            {"retrieval_url": "http://x/studies/1.2.3", "study_uid": "1.2.3", "series_uid": "1.2.3.1"}
        ],
        priority="MEDIUM",
        workitem_uid=uid,
    )


@pytest.fixture(autouse=True)
def _router_path_guard() -> Iterator[None]:
    """Ensure router/ is at sys.path[0] for each test; restore after."""
    saved = list(sys.path)
    _ensure_router_path()
    yield
    sys.path[:] = saved


@pytest.fixture
def proc() -> ModuleType:
    _evict_ups()
    import ups.processor as p
    return p


@pytest.fixture
def fake_workitem() -> Any:
    return _make_workitem()


# ---------------------------------------------------------------------------
# notify_subscriber
# ---------------------------------------------------------------------------

def test_notify_subscriber_happy_path(proc, fake_workitem):
    """Successful POST returns 200; no exception raised."""
    mock_response = SimpleNamespace(status_code=200)
    with mock.patch('ups.processor.requests.post', return_value=mock_response) as m:
        proc.notify_subscriber(fake_workitem, "http://viewer:8042")
    m.assert_called_once()
    call_kwargs = m.call_args
    # URL should include workitem UID
    url_arg = call_kwargs[0][0] if call_kwargs[0] else call_kwargs[1].get('url', '')
    assert fake_workitem.workitem_uid in url_arg


def test_notify_subscriber_non_200_does_not_raise(proc, fake_workitem):
    """Non-200 response is logged but does not raise."""
    mock_response = SimpleNamespace(status_code=503)
    with mock.patch('ups.processor.requests.post', return_value=mock_response):
        proc.notify_subscriber(fake_workitem, "http://viewer:8042")  # must not raise


def test_notify_subscriber_network_error_swallowed(proc, fake_workitem):
    """Network exception is swallowed (subscriber failure must not crash workflow)."""
    import requests as req_lib
    with mock.patch('ups.processor.requests.post', side_effect=req_lib.exceptions.ConnectionError("refused")):
        proc.notify_subscriber(fake_workitem, "http://unreachable:9999")  # must not raise


def test_notify_subscriber_posts_to_correct_url(proc, fake_workitem):
    """POST URL should be subscriber_url/ups-rs/workitems/{uid}."""
    captured = {}
    def fake_post(url, **kwargs):
        captured['url'] = url
        return SimpleNamespace(status_code=200)
    with mock.patch('ups.processor.requests.post', side_effect=fake_post):
        proc.notify_subscriber(fake_workitem, "http://viewer:8042")
    assert captured['url'] == f"http://viewer:8042/ups-rs/workitems/{fake_workitem.workitem_uid}"


def test_notify_subscriber_sends_json_content_type(proc, fake_workitem):
    """POST should include application/dicom+json Content-Type header."""
    captured = {}
    def fake_post(url, **kwargs):
        captured['headers'] = kwargs.get('headers', {})
        return SimpleNamespace(status_code=200)
    with mock.patch('ups.processor.requests.post', side_effect=fake_post):
        proc.notify_subscriber(fake_workitem, "http://viewer:8042")
    assert captured['headers'].get('Content-Type') == 'application/dicom+json'


# ---------------------------------------------------------------------------
# notify_all_subscribers
# ---------------------------------------------------------------------------

def test_notify_all_subscribers_calls_notify_for_each_subscriber(proc, fake_workitem):
    """notify_all_subscribers should call notify_subscriber for every registered subscriber."""
    from ups.subscription_storage import subscription_storage
    subscription_storage.add_subscription(fake_workitem.workitem_uid, "http://viewer-a:8042")
    subscription_storage.add_subscription(fake_workitem.workitem_uid, "http://viewer-b:8042")

    notified = []
    with mock.patch.object(proc, 'notify_subscriber', side_effect=lambda wi, url: notified.append(url)):
        proc.notify_all_subscribers(fake_workitem)

    assert "http://viewer-a:8042" in notified
    assert "http://viewer-b:8042" in notified
    assert len(notified) == 2


def test_notify_all_subscribers_no_subscribers_does_not_raise(proc, fake_workitem):
    """No subscribers registered: should not raise, just log."""
    with mock.patch('ups.processor.requests.post') as m:
        proc.notify_all_subscribers(fake_workitem)
    m.assert_not_called()


def test_notify_all_subscribers_includes_global_subscriber(proc, fake_workitem):
    """Global subscriptions should be included in notifications."""
    from ups.subscription_storage import subscription_storage
    subscription_storage.add_global_subscription("http://global-viewer:8042")

    notified = []
    with mock.patch.object(proc, 'notify_subscriber', side_effect=lambda wi, url: notified.append(url)):
        proc.notify_all_subscribers(fake_workitem)

    assert "http://global-viewer:8042" in notified


# ---------------------------------------------------------------------------
# process_workitem (smoke)
# ---------------------------------------------------------------------------

def test_process_workitem_transitions_state_to_in_progress(proc, fake_workitem, monkeypatch):
    """
    With a 500 model mock, process_workitem must move SCHEDULED -> IN_PROGRESS -> CANCELED.
    Captures the state sequence via a wrapped store_workitem so we can pin the transition.
    """
    mock_model_resp = SimpleNamespace(
        status_code=500,
        text="mocked model error",
        json=lambda: {},
    )
    monkeypatch.setattr('ups.processor.requests.post', lambda *a, **kw: mock_model_resp)
    monkeypatch.setattr(proc, 'notify_all_subscribers', lambda *a, **kw: None)

    # Patch via proc.ups_storage (NOT a fresh `from ups.storage import ...` — that
    # would re-import after `_make_workitem` evicted `ups.*` from sys.modules,
    # producing a different singleton from the one process_workitem actually uses).
    # The post-call `assert states` below catches any regression where the patch
    # fails to take effect (e.g. if process_workitem starts using a fresh import).
    states = []
    real_store = proc.ups_storage.store_workitem

    def _capture_then_store(wi):
        states.append(wi.get_state())
        real_store(wi)

    monkeypatch.setattr(proc.ups_storage, 'store_workitem', _capture_then_store)

    assert fake_workitem.get_state() == 'SCHEDULED'

    try:
        proc.process_workitem(fake_workitem)
    except Exception:
        pass  # processor may bail out on model error; that's expected

    # The state sequence must show IN_PROGRESS before terminating in CANCELED.
    # Captures any silent regression that skips the IN_PROGRESS transition.
    assert states, "store_workitem was never invoked — process_workitem returned without storing"
    assert 'IN_PROGRESS' in states, f"expected IN_PROGRESS in transitions, got {states}"
    assert states[-1] == 'CANCELED', f"expected final state CANCELED, got {states[-1]!r}"


def test_process_workitem_cancels_on_model_network_error(proc, fake_workitem, monkeypatch):
    """
    When the model backend call raises a network exception,
    process_workitem should catch it and set state to CANCELED.
    """
    import requests as req_lib

    monkeypatch.setattr(
        'ups.processor.requests.post',
        mock.Mock(side_effect=req_lib.exceptions.ConnectionError("refused"))
    )
    monkeypatch.setattr(proc, 'notify_all_subscribers', lambda *a, **kw: None)

    try:
        proc.process_workitem(fake_workitem)
    except Exception:
        pass

    final_state = fake_workitem.get_state()
    assert final_state == 'CANCELED'


def test_process_workitem_stores_updated_workitem(proc, fake_workitem, monkeypatch):
    """process_workitem must persist state changes to ups_storage."""
    mock_model_resp = SimpleNamespace(
        status_code=500,
        text="mocked error",
        json=lambda: {},
    )
    monkeypatch.setattr('ups.processor.requests.post', lambda *a, **kw: mock_model_resp)
    monkeypatch.setattr(proc, 'notify_subscriber', lambda *a, **kw: None)

    try:
        proc.process_workitem(fake_workitem)
    except Exception:
        pass

    from ups.storage import ups_storage
    stored = ups_storage.get_workitem(fake_workitem.workitem_uid)
    assert stored is not None
    assert stored.get_state() == 'CANCELED'


# ---------------------------------------------------------------------------
# R1, R2: process_workitem happy path (bilateral and bilateral_with_heatmap)
# ---------------------------------------------------------------------------

def _make_minimal_metadata():
    """Minimal DICOM-JSON metadata for retrieve_series_metadata_sorted's first instance."""
    return (
        {
            "00200032": {"Value": [0.0, 0.0, 0.0]},
            "00200037": {"Value": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]},
            "00200052": {"Value": ["1.2.3.frame"]},
            "00100010": {"Value": [{"Alphabetic": "TEST^PATIENT"}]},
            "00100020": {"Value": ["P001"]},
            "0020000D": {"Value": ["1.2.3"]},
            "00080016": {"Value": ["1.2.840.10008.5.1.4.1.1.4"]},
            "00080018": {"Value": ["1.2.3.4.1"]},
        },
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        1.0,
    )


def _install_fake_server(monkeypatch, response_format="bilateral"):
    """Stub the lazy `from server import (...)` in process_workitem step 4."""
    import sys, types
    fake = types.ModuleType("server")
    fake.detect_response_format = lambda results: response_format
    fake.create_bilateral_sr = lambda original_dicom, results: (b"SR_BYTES", "20260101", "120000.000", "1.2.3.sr.uid")
    fake.create_multiframe_attention_sc = lambda *a, **kw: b"SC_BYTES"
    monkeypatch.setitem(sys.modules, "server", fake)
    return fake


def test_process_workitem_completes_on_bilateral_response(proc, fake_workitem, monkeypatch):
    """Happy path: 200 model + 200 upload -> final state COMPLETED."""
    states = []
    real_store = proc.ups_storage.store_workitem
    def _capture(wi):
        states.append(wi.get_state())
        real_store(wi)
    monkeypatch.setattr(proc.ups_storage, 'store_workitem', _capture)
    monkeypatch.setattr(proc, 'notify_all_subscribers', lambda *a, **kw: None)

    model_resp = SimpleNamespace(
        status_code=200, text="ok",
        json=lambda: {"left": {"prediction": "Benign", "confidence": 95.0},
                       "right": {"prediction": "Malignant", "confidence": 80.0}},
    )
    upload_resp = SimpleNamespace(status_code=200, text="ok", json=lambda: {})
    posts = []
    def _post(url, **kw):
        posts.append((url, kw))
        if "/instances" in url:
            return upload_resp
        return model_resp
    monkeypatch.setattr('ups.processor.requests.post', _post)
    monkeypatch.setattr(proc, 'retrieve_series_metadata_sorted',
                        lambda urls: _make_minimal_metadata())
    _install_fake_server(monkeypatch, response_format="bilateral")

    proc.process_workitem(fake_workitem)

    assert states[-1] == 'COMPLETED', f"expected final COMPLETED, got states={states}"
    # Bilateral -> exactly one upload (SR only), no SC.
    upload_calls = [u for u in posts if "/instances" in u[0]]
    assert len(upload_calls) == 1
    assert upload_calls[0][1].get("data") == b"SR_BYTES"


def test_process_workitem_bilateral_with_heatmap_uploads_sr_and_sc(proc, fake_workitem, monkeypatch):
    """bilateral_with_heatmap branch: when attention_maps.data is set, upload BOTH SR and SC."""
    monkeypatch.setattr(proc.ups_storage, 'store_workitem', lambda wi: None)
    monkeypatch.setattr(proc, 'notify_all_subscribers', lambda *a, **kw: None)

    model_resp = SimpleNamespace(
        status_code=200, text="ok",
        json=lambda: {
            "left": {"prediction": "Benign", "confidence": 95.0},
            "right": {"prediction": "Malignant", "confidence": 80.0},
            "attention_maps": {"data": "BASE64DATA", "shape": [5, 16, 16, 3], "dtype": "uint8"},
        },
    )
    upload_resp = SimpleNamespace(status_code=200, text="ok", json=lambda: {})
    posts = []
    def _post(url, **kw):
        posts.append((url, kw))
        if "/instances" in url:
            return upload_resp
        return model_resp
    monkeypatch.setattr('ups.processor.requests.post', _post)
    monkeypatch.setattr(proc, 'retrieve_series_metadata_sorted',
                        lambda urls: _make_minimal_metadata())
    _install_fake_server(monkeypatch, response_format="bilateral_with_heatmap")

    proc.process_workitem(fake_workitem)

    # H7: explicitly assert one POST hit the configured model URL.
    model_calls = [u for u in posts if "/analyze/mri" in u[0]]
    assert len(model_calls) == 1, f"expected exactly one model POST, saw URLs: {[u[0] for u in posts]}"
    upload_calls = [u for u in posts if "/instances" in u[0]]
    upload_payloads = [u[1].get("data") for u in upload_calls]
    assert b"SR_BYTES" in upload_payloads
    assert b"SC_BYTES" in upload_payloads
    assert len(upload_calls) == 2
