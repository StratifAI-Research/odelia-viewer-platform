"""Unit tests for router/ups/routes.py — 6 UPS-RS REST handlers.

Covers:
  - register_ups_routes(): all 6 URIs registered
  - CreateWorkitem: POST happy path, wrong method, missing study_uid
  - GetWorkitem: GET happy path, wrong method, missing uid, not found
  - UpdateWorkitemState: PUT happy path, wrong method, missing uid, missing state, not found
  - QueryWorkitems: GET happy path, wrong method, state filter
  - SubscribeToWorkitem: POST happy path, wrong method, missing uid, missing subscriber_url, workitem not found
  - UnsubscribeFromWorkitem: DELETE happy path, wrong method, missing uid
  - ServeManifest: GET happy path, GET missing file, wrong method
"""
import json
import sys
from types import ModuleType
from typing import Iterator
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------

import os as _os
_ROUTER_DIR = _os.path.abspath(
    _os.path.join(_os.path.dirname(__file__), '..', '..', '..', 'router')
)


def _ensure_router_path():
    """Ensure router/ is at front of sys.path so ups.routes resolves to router/ups/routes."""
    if _ROUTER_DIR in sys.path:
        sys.path.remove(_ROUTER_DIR)
    sys.path.insert(0, _ROUTER_DIR)


@pytest.fixture(autouse=True)
def _router_path_guard() -> Iterator[None]:
    """Ensure router/ is at front of sys.path for each test; restore after.

    Both viewer and router conftest files run at collection time. By test time
    sys.path is typically [viewer/, router/, ...]. This fixture temporarily moves
    router/ to front, and restores the saved path after the test so viewer-side
    tests that follow still find viewer/ups/ first.
    """
    saved = list(sys.path)
    _ensure_router_path()
    yield
    sys.path[:] = saved


def _load_ups_routes():
    """Evict and re-import router/ups/routes so module-level state is fresh."""
    _ensure_router_path()
    for key in [k for k in sys.modules if k == "ups" or k.startswith("ups.")]:
        del sys.modules[key]
    from ups import routes
    return routes


@pytest.fixture
def routes() -> ModuleType:
    return _load_ups_routes()


def _make_workitem():
    for key in [k for k in sys.modules if k == "ups" or k.startswith("ups.")]:
        del sys.modules[key]
    from ups.workitem import UPSWorkitem
    return UPSWorkitem(
        study_uid="1.2.3",
        series_uids=["1.2.3.1"],
        wado_rs_retrieval=[
            {"retrieval_url": "http://x/studies/1.2.3", "study_uid": "1.2.3", "series_uid": "1.2.3.1"}
        ],
        priority="MEDIUM",
        workitem_uid="2.25.777",
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_register_ups_routes_registers_all_six_uris(routes):
    import orthanc
    orthanc._rest_callbacks.clear()
    routes.register_ups_routes()
    uris = [u for u, _ in orthanc._rest_callbacks]
    assert '/ups-rs/workitems$' in uris
    assert '/ups-rs/workitems/([0-9.]+)$' in uris
    assert '/ups-rs/workitems/([0-9.]+)/state$' in uris
    assert '/ups-rs/workitems/([0-9.]+)/subscribers$' in uris
    assert '/ups-rs/workitems/([0-9.]+)/subscribers/(.+)$' in uris
    assert '/manifest$' in uris


# ---------------------------------------------------------------------------
# CreateWorkitem
# ---------------------------------------------------------------------------

def test_create_workitem_wrong_method_returns_405(out, routes):
    with mock.patch('ups.routes.process_workitem'):
        routes.CreateWorkitem(out, '/ups-rs/workitems', method='GET', body=b'{}', groups=[])
    assert out.status == 405


def test_create_workitem_missing_study_uid_returns_400(out, routes):
    body = json.dumps({"series_uids": ["1.2.3"]}).encode()
    routes.CreateWorkitem(out, '/ups-rs/workitems', method='POST', body=body, groups=[])
    assert out.status == 400


def test_create_workitem_happy_path_returns_200(out, routes):
    body = json.dumps({
        "study_uid": "1.2.3",
        "series_uids": ["1.2.3.1"],
        "wado_rs_base": "http://viewer:8042/dicom-web",
    }).encode()
    with mock.patch('ups.routes.process_workitem'):
        routes.CreateWorkitem(out, '/ups-rs/workitems', method='POST', body=body, groups=[])
    assert out.status == 200
    result = json.loads(out.body)
    # The created workitem should have state SCHEDULED
    assert "00741000" in result  # Procedure Step State tag
    state_value = result["00741000"]["Value"][0]
    assert state_value == "SCHEDULED"


def test_create_workitem_starts_background_thread(out, routes):
    """Verify Thread is constructed with daemon=True, target invokes process_workitem, start() is called."""
    body = json.dumps({
        "study_uid": "1.2.3",
        "series_uids": ["1.2.3.1"],
    }).encode()
    calls = []
    with mock.patch('ups.routes.process_workitem',
                    side_effect=lambda w: calls.append(w.workitem_uid)), \
         mock.patch('threading.Thread') as mock_thread:
        mock_thread.return_value.start = mock.MagicMock()
        routes.CreateWorkitem(out, '/ups-rs/workitems', method='POST', body=body, groups=[])

        mock_thread.assert_called_once()
        assert mock_thread.call_args.kwargs.get('daemon') is True
        mock_thread.return_value.start.assert_called_once()

        # The captured target callable must invoke process_workitem.
        target = mock_thread.call_args.kwargs['target']
        target()
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# GetWorkitem
# ---------------------------------------------------------------------------

def test_get_workitem_wrong_method_returns_405(out, routes):
    routes.GetWorkitem(out, '/ups-rs/workitems/2.25.1', method='POST', body=b'', groups=['2.25.1'])
    assert out.status == 405


def test_get_workitem_missing_uid_returns_400(out, routes):
    routes.GetWorkitem(out, '/ups-rs/workitems/', method='GET', body=b'', groups=[])
    assert out.status == 400


def test_get_workitem_not_found_returns_404(out, routes):
    routes.GetWorkitem(out, '/ups-rs/workitems/2.25.nope', method='GET', body=b'', groups=['2.25.nope'])
    assert out.status == 404


def test_get_workitem_happy_path(out, routes):
    wi = _make_workitem()
    from ups.storage import ups_storage
    ups_storage.store_workitem(wi)
    routes.GetWorkitem(out, f'/ups-rs/workitems/{wi.workitem_uid}', method='GET', body=b'', groups=[wi.workitem_uid])
    assert out.status == 200
    data = json.loads(out.body)
    # The returned data should contain the workitem UID
    assert "00080018" in data
    assert data["00080018"]["Value"][0] == wi.workitem_uid


# ---------------------------------------------------------------------------
# UpdateWorkitemState
# ---------------------------------------------------------------------------

def test_update_workitem_state_wrong_method_returns_405(out, routes):
    routes.UpdateWorkitemState(out, '/ups-rs/workitems/2.25.1/state', method='GET', body=b'', groups=['2.25.1'])
    assert out.status == 405


def test_update_workitem_state_missing_uid_returns_400(out, routes):
    body = json.dumps({"state": "COMPLETED"}).encode()
    routes.UpdateWorkitemState(out, '/ups-rs/workitems//state', method='PUT', body=body, groups=[])
    assert out.status == 400


def test_update_workitem_state_missing_state_returns_400(out, routes):
    wi = _make_workitem()
    from ups.storage import ups_storage
    ups_storage.store_workitem(wi)
    body = json.dumps({}).encode()
    routes.UpdateWorkitemState(out, f'/ups-rs/workitems/{wi.workitem_uid}/state', method='PUT', body=body, groups=[wi.workitem_uid])
    assert out.status == 400


def test_update_workitem_state_not_found_returns_404(out, routes):
    body = json.dumps({"state": "COMPLETED"}).encode()
    routes.UpdateWorkitemState(out, '/ups-rs/workitems/9.9.9/state', method='PUT', body=body, groups=['9.9.9'])
    assert out.status == 404


def test_update_workitem_state_happy_path(out, routes):
    wi = _make_workitem()
    from ups.storage import ups_storage
    ups_storage.store_workitem(wi)
    body = json.dumps({"state": "IN_PROGRESS", "progress_info": "Running"}).encode()
    routes.UpdateWorkitemState(out, f'/ups-rs/workitems/{wi.workitem_uid}/state', method='PUT', body=body, groups=[wi.workitem_uid])
    assert out.status == 200
    # Verify state was actually updated in storage
    retrieved = ups_storage.get_workitem(wi.workitem_uid)
    assert retrieved.get_state() == "IN_PROGRESS"


# ---------------------------------------------------------------------------
# QueryWorkitems
# ---------------------------------------------------------------------------

def test_query_workitems_wrong_method_returns_405(out, routes):
    routes.QueryWorkitems(out, '/ups-rs/workitems', method='POST', body=b'', get={})
    assert out.status == 405


def test_query_workitems_returns_all_when_no_filter(out, routes):
    wi = _make_workitem()
    from ups.storage import ups_storage
    ups_storage.store_workitem(wi)
    routes.QueryWorkitems(out, '/ups-rs/workitems', method='GET', body=b'', get={})
    assert out.status == 200
    result = json.loads(out.body)
    assert isinstance(result, list)
    assert len(result) == 1


def test_query_workitems_state_filter(out, routes):
    wi = _make_workitem()
    from ups.storage import ups_storage
    ups_storage.store_workitem(wi)
    routes.QueryWorkitems(
        out, '/ups-rs/workitems', method='GET', body=b'',
        get={"state": ["SCHEDULED"]}
    )
    assert out.status == 200
    result = json.loads(out.body)
    assert len(result) == 1

    # Filter for non-matching state
    out2_class = type(out)
    out2 = out2_class()
    routes.QueryWorkitems(
        out2, '/ups-rs/workitems', method='GET', body=b'',
        get={"state": ["COMPLETED"]}
    )
    assert out2.status == 200
    result2 = json.loads(out2.body)
    assert result2 == []


def test_query_workitems_empty_returns_empty_list(out, routes):
    routes.QueryWorkitems(out, '/ups-rs/workitems', method='GET', body=b'', get={})
    assert out.status == 200
    assert json.loads(out.body) == []


# ---------------------------------------------------------------------------
# SubscribeToWorkitem
# ---------------------------------------------------------------------------

def test_subscribe_wrong_method_returns_405(out, routes):
    routes.SubscribeToWorkitem(out, '/ups-rs/workitems/2.25.1/subscribers', method='GET', body=b'', groups=['2.25.1'])
    assert out.status == 405


def test_subscribe_missing_uid_returns_400(out, routes):
    body = json.dumps({"subscriber_url": "http://viewer:8042"}).encode()
    routes.SubscribeToWorkitem(out, '/ups-rs/workitems//subscribers', method='POST', body=body, groups=[])
    assert out.status == 400


def test_subscribe_missing_subscriber_url_returns_400(out, routes):
    wi = _make_workitem()
    from ups.storage import ups_storage
    ups_storage.store_workitem(wi)
    body = json.dumps({}).encode()
    routes.SubscribeToWorkitem(out, f'/ups-rs/workitems/{wi.workitem_uid}/subscribers', method='POST', body=body, groups=[wi.workitem_uid])
    assert out.status == 400


def test_subscribe_workitem_not_found_returns_404(out, routes):
    body = json.dumps({"subscriber_url": "http://viewer:8042"}).encode()
    routes.SubscribeToWorkitem(out, '/ups-rs/workitems/9.9.9/subscribers', method='POST', body=body, groups=['9.9.9'])
    assert out.status == 404


def test_subscribe_happy_path(out, routes):
    wi = _make_workitem()
    from ups.storage import ups_storage
    ups_storage.store_workitem(wi)
    body = json.dumps({"subscriber_url": "http://viewer:8042"}).encode()
    with mock.patch('ups.processor.notify_subscriber'):
        routes.SubscribeToWorkitem(
            out, f'/ups-rs/workitems/{wi.workitem_uid}/subscribers',
            method='POST', body=body, groups=[wi.workitem_uid]
        )
    assert out.status == 200
    result = json.loads(out.body)
    assert result.get("status") == "subscribed"


# ---------------------------------------------------------------------------
# UnsubscribeFromWorkitem
# ---------------------------------------------------------------------------

def test_unsubscribe_wrong_method_returns_405(out, routes):
    routes.UnsubscribeFromWorkitem(out, '/ups-rs/workitems/2.25.1/subscribers/x', method='GET', body=b'', groups=['2.25.1', 'http://x'])
    assert out.status == 405


def test_unsubscribe_missing_uid_returns_400(out, routes):
    routes.UnsubscribeFromWorkitem(out, '/ups-rs/workitems//subscribers/', method='DELETE', body=b'', groups=[])
    assert out.status == 400


def test_unsubscribe_happy_path(out, routes):
    wi = _make_workitem()
    from ups.storage import ups_storage
    ups_storage.store_workitem(wi)
    from ups.subscription_storage import subscription_storage
    subscription_storage.add_subscription(wi.workitem_uid, "http://viewer:8042")

    routes.UnsubscribeFromWorkitem(
        out, f'/ups-rs/workitems/{wi.workitem_uid}/subscribers/http://viewer:8042',
        method='DELETE', body=b'', groups=[wi.workitem_uid, 'http://viewer:8042']
    )
    assert out.status == 200
    result = json.loads(out.body)
    assert result.get("status") == "unsubscribed"


# ---------------------------------------------------------------------------
# ServeManifest
# ---------------------------------------------------------------------------

def test_serve_manifest_wrong_method_returns_405(out, routes):
    routes.ServeManifest(out, '/manifest', method='POST', body=b'', groups=[])
    assert out.status == 405


def test_serve_manifest_missing_file_returns_404(out, routes, tmp_path):
    from ups import routes as r
    with mock.patch.object(r, 'MANIFEST_PATH', str(tmp_path / "no_manifest.json")):
        r.ServeManifest(out, '/manifest', method='GET', body=b'', groups=[])
    assert out.status == 404


def test_serve_manifest_returns_json(out, routes, tmp_path):
    manifest_file = tmp_path / "manifest.json"
    manifest_data = {"models": ["breast-cancer"]}
    manifest_file.write_text(json.dumps(manifest_data))
    from ups import routes as r
    with mock.patch.object(r, 'MANIFEST_PATH', str(manifest_file)):
        r.ServeManifest(out, '/manifest', method='GET', body=b'', groups=[])
    assert out.status == 200
    assert json.loads(out.body) == manifest_data
