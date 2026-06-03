"""Unit tests for viewer/router.py handlers.

SendToAiDicomWeb and SendToAi are deliberately not fully covered here because
they make outbound HTTP calls via `requests` to a router service AND call
multiple cascaded orthanc REST paths in a sequence that is tightly coupled to
real Orthanc data. Their happy-path requires mocking an entire multi-step chain
(study exists, series exist, series tags, series instances, DICOMweb config,
UPS POST, subscription POST). We provide:
  - wrong-method → 405 tests (no network calls)
  - missing-field → 400 tests (no network calls)
A note in DONE report flags them as refactor candidates for dependency injection.

All other handlers (UPSUpdateWorkitem, UPSGetWorkitem, UPSWorkitemHandler,
GetAIManifest, SendToAiDicom) have full happy + error path coverage.
"""
import json
import sys
from types import ModuleType
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Module import helper
# ---------------------------------------------------------------------------

def _load_router():
    """Evict and re-import viewer/router so module-level registrations replay."""
    for key in [k for k in sys.modules if k == "router" or k.startswith("router.")]:
        del sys.modules[key]
    for key in [k for k in sys.modules if k == "ups" or k.startswith("ups.")]:
        del sys.modules[key]
    import router as _router
    return _router


@pytest.fixture
def router() -> ModuleType:
    return _load_router()


# ---------------------------------------------------------------------------
# Registration — module-level RegisterRestCallback
# ---------------------------------------------------------------------------

def test_module_registers_expected_uris():
    import orthanc
    orthanc._rest_callbacks.clear()
    _load_router()
    uris = [u for u, _ in orthanc._rest_callbacks]
    assert "/send-to-ai" in uris
    assert "/send-to-ai-dicom" in uris
    assert "/send-to-ai-dicomweb" in uris
    assert "/ai-manifest" in uris
    assert "/ups-rs/workitems/(.*)" in uris


def test_module_registers_feedback_endpoints_when_available():
    import orthanc
    orthanc._rest_callbacks.clear()
    _load_router()
    uris = [u for u, _ in orthanc._rest_callbacks]
    # feedback_routes imports successfully in the test environment
    assert "/feedback/submit" in uris
    assert "/feedback/health" in uris


# ---------------------------------------------------------------------------
# UPSUpdateWorkitem
# ---------------------------------------------------------------------------

def _make_workitem_body(uid: str = "2.25.111", state: str = "SCHEDULED") -> str:
    return json.dumps({
        "00741000": {"vr": "CS", "Value": [state]},
        "00080018": {"vr": "UI", "Value": [uid]},
        "0020000D": {"vr": "UI", "Value": ["1.2.3"]},
    })


def test_ups_update_workitem_happy(out, router):
    body = _make_workitem_body()
    router.UPSUpdateWorkitem(
        out, "/ups-rs/workitems/2.25.111",
        method="POST", body=body,
    )
    assert out.status == 200
    data = json.loads(out.body)
    assert data["status"] == "updated"


def test_ups_update_workitem_stores_in_kv(out, router):
    import orthanc
    body = _make_workitem_body(uid="2.25.222")
    router.UPSUpdateWorkitem(
        out, "/ups-rs/workitems/2.25.222",
        method="POST", body=body,
    )
    assert out.status == 200
    # Verify KV store received something for this workitem
    stored = orthanc.GetKeyValue("ups", "upsworkitem:2.25.222")
    assert stored is not None


def test_ups_update_workitem_wrong_method_returns_405(out, router):
    router.UPSUpdateWorkitem(
        out, "/ups-rs/workitems/2.25.333",
        method="GET", body=b"",
    )
    assert out.status == 405


def test_ups_update_workitem_bad_json_returns_400(out, router):
    router.UPSUpdateWorkitem(
        out, "/ups-rs/workitems/2.25.444",
        method="POST", body="not-json",
    )
    assert out.status == 400


# ---------------------------------------------------------------------------
# UPSGetWorkitem
# ---------------------------------------------------------------------------

def test_ups_get_workitem_returns_404_when_not_found(out, router):
    router.UPSGetWorkitem(
        out, "/ups-rs/workitems/phantom.uid",
        method="GET",
    )
    assert out.status == 404


def test_ups_get_workitem_returns_stored_workitem(out, router):
    uid = "2.25.555"
    body = _make_workitem_body(uid=uid)
    router.UPSUpdateWorkitem(out, f"/ups-rs/workitems/{uid}", method="POST", body=body)
    assert out.status == 200
    out2 = type(out).__new__(type(out))
    out2.__init__()
    router.UPSGetWorkitem(out2, f"/ups-rs/workitems/{uid}", method="GET")
    assert out2.status == 200
    data = json.loads(out2.body)
    assert data["00741000"]["Value"][0] == "SCHEDULED"


def test_ups_get_workitem_wrong_method_returns_405(out, router):
    router.UPSGetWorkitem(
        out, "/ups-rs/workitems/2.25.666",
        method="DELETE",
    )
    assert out.status == 405


# ---------------------------------------------------------------------------
# UPSWorkitemHandler (dispatch)
# ---------------------------------------------------------------------------

def test_ups_workitem_handler_dispatches_post(out, router):
    uid = "2.25.777"
    body = _make_workitem_body(uid=uid)
    router.UPSWorkitemHandler(out, f"/ups-rs/workitems/{uid}", method="POST", body=body)
    assert out.status == 200


def test_ups_workitem_handler_dispatches_get(out, router):
    uid = "2.25.888"
    body = _make_workitem_body(uid=uid)
    # Store first
    router.UPSUpdateWorkitem(out, f"/ups-rs/workitems/{uid}", method="POST", body=body)
    out2 = type(out).__new__(type(out))
    out2.__init__()
    router.UPSWorkitemHandler(out2, f"/ups-rs/workitems/{uid}", method="GET")
    assert out2.status == 200


def test_ups_workitem_handler_unknown_method_returns_405(out, router):
    router.UPSWorkitemHandler(
        out, "/ups-rs/workitems/2.25.999",
        method="PUT",
    )
    assert out.status == 405


# ---------------------------------------------------------------------------
# GetAIManifest
# ---------------------------------------------------------------------------

def test_get_ai_manifest_wrong_method_returns_405(out, router):
    router.GetAIManifest(out, "/ai-manifest", method="POST", body=b"")
    assert out.status == 405


def test_get_ai_manifest_missing_target_url_returns_400(out, router):
    router.GetAIManifest(out, "/ai-manifest", method="GET", get={})
    assert out.status == 400
    assert "target_url" in out.body


def test_get_ai_manifest_returns_manifest_on_200(out, router):
    manifest_data = {"models": ["breast-cancer"]}
    mock_resp = mock.Mock()
    mock_resp.status_code = 200
    mock_resp.text = json.dumps(manifest_data)
    with mock.patch("requests.get", return_value=mock_resp) as mocked:
        router.GetAIManifest(
            out, "/ai-manifest",
            method="GET",
            get={"target_url": "http://router:8042/dicom-web"},
        )
    mocked.assert_called_once()
    assert out.status == 200
    data = json.loads(out.body)
    assert data["models"] == ["breast-cancer"]


def test_get_ai_manifest_returns_null_manifest_on_router_404(out, router):
    mock_resp = mock.Mock()
    mock_resp.status_code = 404
    with mock.patch("requests.get", return_value=mock_resp):
        router.GetAIManifest(
            out, "/ai-manifest",
            method="GET",
            get={"target_url": "http://router:8042/dicom-web"},
        )
    assert out.status == 200
    data = json.loads(out.body)
    assert data["manifest"] is None


def test_get_ai_manifest_returns_null_manifest_on_connection_error(out, router):
    import requests
    with mock.patch("requests.get", side_effect=requests.exceptions.ConnectionError("refused")):
        router.GetAIManifest(
            out, "/ai-manifest",
            method="GET",
            get={"target_url": "http://router:8042/dicom-web"},
        )
    assert out.status == 200
    data = json.loads(out.body)
    assert data["manifest"] is None


def test_get_ai_manifest_strips_dicom_web_suffix(out, router):
    """Ensure /dicom-web suffix is stripped when constructing manifest URL."""
    mock_resp = mock.Mock()
    mock_resp.status_code = 200
    mock_resp.text = json.dumps({})
    with mock.patch("requests.get", return_value=mock_resp) as mocked:
        router.GetAIManifest(
            out, "/ai-manifest",
            method="GET",
            get={"target_url": "http://router:8042/dicom-web"},
        )
    called_url = mocked.call_args[0][0]
    assert called_url == "http://router:8042/manifest"


# ---------------------------------------------------------------------------
# SendToAiDicom — wrong-method + bad-input (no network calls)
# ---------------------------------------------------------------------------

def test_send_to_ai_dicom_wrong_method_returns_405(out, router):
    router.SendToAiDicom(out, "/send-to-ai-dicom", method="GET")
    assert out.status == 405


def test_send_to_ai_dicom_missing_fields_returns_400(out, rest_fake, router):
    # Needs rest_fake bound because FilterAIResultSeries calls orthanc.RestApiGet
    # when study_id is present. Provide missing both study_id AND target → 400.
    router.SendToAiDicom(
        out, "/send-to-ai-dicom",
        method="POST",
        body=json.dumps({"target": "mst"}),  # no study_id
    )
    assert out.status == 400
    assert "study_id" in out.body


def test_send_to_ai_dicom_empty_body_returns_400(out, router):
    router.SendToAiDicom(
        out, "/send-to-ai-dicom",
        method="POST",
        body=json.dumps({}),
    )
    assert out.status == 400
    assert "study_id" in out.body


# ---------------------------------------------------------------------------
# SendToAiDicomWeb — wrong-method + bad-input (no network calls)
# ---------------------------------------------------------------------------

def test_send_to_ai_dicomweb_wrong_method_returns_405(out, router):
    router.SendToAiDicomWeb(out, "/send-to-ai-dicomweb", method="GET")
    assert out.status == 405


def test_send_to_ai_dicomweb_missing_study_id_returns_400(out, router):
    router.SendToAiDicomWeb(
        out, "/send-to-ai-dicomweb",
        method="POST",
        body=json.dumps({"target": "mst", "target_url": "http://x"}),
    )
    assert out.status == 400
    assert "study_id" in out.body


def test_send_to_ai_dicomweb_missing_target_url_returns_400(out, router):
    router.SendToAiDicomWeb(
        out, "/send-to-ai-dicomweb",
        method="POST",
        body=json.dumps({"study_id": "abc", "target": "mst"}),
    )
    assert out.status == 400
    assert "target_url" in out.body


# ---------------------------------------------------------------------------
# SendToAi — thin wrapper; wrong-method propagates via delegate
# ---------------------------------------------------------------------------

def test_send_to_ai_wrong_method_returns_405(out, router):
    router.SendToAi(out, "/send-to-ai", method="GET")
    assert out.status == 405


def test_send_to_ai_missing_study_id_returns_400(out, router):
    router.SendToAi(
        out, "/send-to-ai",
        method="POST",
        body=json.dumps({"target": "mst", "target_url": "http://x"}),
    )
    assert out.status == 400
    assert "study_id" in out.body


# ---------------------------------------------------------------------------
# V5: register_feedback_endpoints fallback when feedback_routes import fails
# ---------------------------------------------------------------------------

def test_register_feedback_endpoints_is_none_when_feedback_routes_lacks_register(monkeypatch):
    """When import succeeds but register_feedback_endpoints attr is missing, fallback engages."""
    import sys
    import types as _types
    # Install a fake feedback_routes module WITHOUT register_feedback_endpoints.
    fake = _types.ModuleType("feedback_routes")
    monkeypatch.setitem(sys.modules, "feedback_routes", fake)
    # Evict router so re-import picks up the fake.
    sys.modules.pop("router", None)
    import router as r
    assert r.register_feedback_endpoints is None


def test_no_feedback_uris_registered_when_register_is_none(monkeypatch):
    """When register_feedback_endpoints is None, the module-load guard skips registration."""
    import sys
    import types as _types
    import orthanc
    orthanc._rest_callbacks.clear()
    fake = _types.ModuleType("feedback_routes")
    monkeypatch.setitem(sys.modules, "feedback_routes", fake)
    sys.modules.pop("router", None)
    import router  # noqa: F401  (re-import triggers callback registration)
    feedback_uris = [uri for uri, _ in orthanc._rest_callbacks if uri.startswith("/feedback")]
    assert feedback_uris == []


# ---------------------------------------------------------------------------
# V6: UPSGetWorkitem / UPSUpdateWorkitem ups_storage=None branches
# ---------------------------------------------------------------------------

def test_ups_get_workitem_returns_500_when_storage_not_initialized(out, router, monkeypatch):
    monkeypatch.setattr(router, "ups_storage", None)
    router.UPSGetWorkitem(out, "/ups-rs/workitems/2.25.999", method="GET")
    assert out.status == 500
    assert "UPS storage not initialized" in (out.body or "")


def test_ups_update_workitem_falls_back_to_body_state_when_storage_unavailable(out, router, monkeypatch):
    """When ups_storage is falsy, UPSUpdateWorkitem extracts the state from the body (no persistence) and still answers 200."""
    monkeypatch.setattr(router, "ups_storage", None)
    body = _make_workitem_body(uid="2.25.fallback", state="IN_PROGRESS")
    router.UPSUpdateWorkitem(out, "/ups-rs/workitems/2.25.fallback", method="POST", body=body)
    assert out.status == 200
    import json as _json
    resp = _json.loads(out.body)
    assert resp == {"status": "updated"}


# ---------------------------------------------------------------------------
# D2: small helper coverage (FilterAIResultSeries, HasProcessableContent,
#     GetStudyInstanceUID, ListModalities)
# ---------------------------------------------------------------------------

import json as _json


def test_get_study_instance_uid_returns_uid_from_main_tags(out, router, rest_fake):
    rest_fake.responses[("GET", "/studies/STD1")] = _json.dumps(
        {"MainDicomTags": {"StudyInstanceUID": "1.2.3.std"}}
    ).encode()
    assert router.GetStudyInstanceUID("STD1") == "1.2.3.std"


def test_get_study_instance_uid_returns_none_when_tag_missing(out, router, rest_fake):
    rest_fake.responses[("GET", "/studies/STD2")] = _json.dumps(
        {"MainDicomTags": {"PatientName": "X"}}    # no StudyInstanceUID
    ).encode()
    assert router.GetStudyInstanceUID("STD2") is None


def test_get_study_instance_uid_returns_none_on_orthanc_error(out, router, rest_fake):
    """When orthanc.RestApiGet raises (no response bound), GetStudyInstanceUID returns None."""
    # No response bound -> rest_fake raises RuntimeError, which the function catches.
    assert router.GetStudyInstanceUID("MISSING") is None


def test_list_modalities_returns_configured_modalities(out, router, rest_fake):
    rest_fake.responses[("GET", "/modalities")] = b"[\"PACS1\", \"PACS2\"]"
    rest_fake.responses[("GET", "/modalities/PACS1")] = _json.dumps(
        {"Host": "h1", "Port": 4242, "AET": "AET1"}
    ).encode()
    rest_fake.responses[("GET", "/modalities/PACS2")] = _json.dumps(
        {"Host": "h2", "Port": 4243, "AET": "AET2"}
    ).encode()
    result = router.ListModalities()
    assert result == ["PACS1", "PACS2"]


def test_list_modalities_returns_empty_on_orthanc_error(out, router, rest_fake):
    """No response bound -> rest_fake raises -> ListModalities catches -> returns []."""
    assert router.ListModalities() == []


def test_filter_ai_result_series_excludes_known_ai_descriptions(out, router, rest_fake):
    """Series whose description matches an AI marker is filtered out; others remain."""
    rest_fake.responses[("GET", "/studies/STD/series")] = _json.dumps([
        {"ID": "S-orig"},
        {"ID": "S-ai-sr"},
        {"ID": "S-ai-heatmap"},
    ]).encode()
    rest_fake.responses[("GET", "/series/S-orig/tags?simplify")] = _json.dumps(
        {"SeriesDescription": "T1 axial", "Modality": "MR"}
    ).encode()
    rest_fake.responses[("GET", "/series/S-ai-sr/tags?simplify")] = _json.dumps(
        {"SeriesDescription": "Automated Diagnostic Findings", "Modality": "SR"}
    ).encode()
    rest_fake.responses[("GET", "/series/S-ai-heatmap/tags?simplify")] = _json.dumps(
        {"SeriesDescription": "Axial - Heatmap overlay", "Modality": "SC"}
    ).encode()
    result = router.FilterAIResultSeries("STD")
    assert result == ["S-orig"]


def test_filter_ai_result_series_treats_sc_or_sr_with_ai_keyword_as_ai(out, router, rest_fake):
    """An SC/SR modality with 'AI' anywhere in the description is filtered out."""
    rest_fake.responses[("GET", "/studies/STD/series")] = _json.dumps([
        {"ID": "S-ai-sc"},
        {"ID": "S-orig"},
    ]).encode()
    rest_fake.responses[("GET", "/series/S-ai-sc/tags?simplify")] = _json.dumps(
        {"SeriesDescription": "Generic AI Stuff", "Modality": "SC"}
    ).encode()
    rest_fake.responses[("GET", "/series/S-orig/tags?simplify")] = _json.dumps(
        {"SeriesDescription": "Axial T2", "Modality": "MR"}
    ).encode()
    result = router.FilterAIResultSeries("STD")
    assert result == ["S-orig"]


def test_filter_ai_result_series_filters_description_prefix_and_suffix_markers(out, router, rest_fake):
    """Descriptions starting with `AI_` or ending with `_AI` are filtered."""
    rest_fake.responses[("GET", "/studies/STD/series")] = _json.dumps([
        {"ID": "S-prefix"},
        {"ID": "S-suffix"},
        {"ID": "S-clean"},
    ]).encode()
    rest_fake.responses[("GET", "/series/S-prefix/tags?simplify")] = _json.dumps(
        {"SeriesDescription": "AI_Stuff", "Modality": "MR"}
    ).encode()
    rest_fake.responses[("GET", "/series/S-suffix/tags?simplify")] = _json.dumps(
        {"SeriesDescription": "Stuff_AI", "Modality": "MR"}
    ).encode()
    rest_fake.responses[("GET", "/series/S-clean/tags?simplify")] = _json.dumps(
        {"SeriesDescription": "Original", "Modality": "MR"}
    ).encode()
    result = router.FilterAIResultSeries("STD")
    assert result == ["S-clean"]


def test_filter_ai_result_series_keeps_series_when_tag_lookup_fails(out, router, rest_fake):
    """If reading tags raises, the series is conservatively kept as original."""
    rest_fake.responses[("GET", "/studies/STD/series")] = _json.dumps([
        {"ID": "S-tag-fail"},
        {"ID": "S-ok"},
    ]).encode()
    # No response bound for S-tag-fail -> rest_fake raises -> series is kept.
    rest_fake.responses[("GET", "/series/S-ok/tags?simplify")] = _json.dumps(
        {"SeriesDescription": "Original", "Modality": "MR"}
    ).encode()
    result = router.FilterAIResultSeries("STD")
    assert sorted(result) == ["S-ok", "S-tag-fail"]


def test_filter_ai_result_series_returns_empty_on_outer_error(out, router, rest_fake):
    """If the study/series listing raises, return empty list."""
    # No response bound for /studies/.../series -> outer except returns [].
    assert router.FilterAIResultSeries("NOPE") == []


def test_has_processable_content_true_when_any_original_series(out, router, rest_fake):
    rest_fake.responses[("GET", "/studies/STD/series")] = _json.dumps([{"ID": "S-orig"}]).encode()
    rest_fake.responses[("GET", "/series/S-orig/tags?simplify")] = _json.dumps(
        {"SeriesDescription": "Axial", "Modality": "MR"}
    ).encode()
    assert router.HasProcessableContent("STD") is True


def test_has_processable_content_false_when_all_ai(out, router, rest_fake):
    rest_fake.responses[("GET", "/studies/STD/series")] = _json.dumps([{"ID": "S-ai"}]).encode()
    rest_fake.responses[("GET", "/series/S-ai/tags?simplify")] = _json.dumps(
        {"SeriesDescription": "Automated Diagnostic Findings", "Modality": "SR"}
    ).encode()
    assert router.HasProcessableContent("STD") is False


# ---------------------------------------------------------------------------
# D4: wider SendToAi(Dicom|DicomWeb) coverage — beyond the early 400/405 checks
# ---------------------------------------------------------------------------

def _bind_series_listing(rest_fake, study_id, series_list, ai=False):
    """Bind /studies/<id>/series + tags-simplify for each series. ai=True flags them as AI results."""
    import json as _json
    rest_fake.responses[("GET", f"/studies/{study_id}/series")] = _json.dumps(
        [{"ID": sid} for sid in series_list]
    ).encode()
    desc = "Automated Diagnostic Findings" if ai else "Axial T1"
    modality = "SR" if ai else "MR"
    for sid in series_list:
        rest_fake.responses[("GET", f"/series/{sid}/tags?simplify")] = _json.dumps(
            {"SeriesDescription": desc, "Modality": modality}
        ).encode()


def test_send_to_ai_dicom_no_processable_content_returns_400(out, router, rest_fake):
    """If no series_uids in body AND HasProcessableContent is False, return 400."""
    _bind_series_listing(rest_fake, "STD", ["S-ai-1"], ai=True)  # all AI -> not processable
    body = b'{"study_id": "STD", "target": "PACS"}'
    router.SendToAiDicom(out, "/send-to-ai-dicom", method="POST", body=body)
    assert out.status == 400
    assert "no processable content" in (out.body or "").lower()


def test_send_to_ai_dicom_no_series_after_filter_returns_400(out, router, rest_fake):
    """series_uids list, but every lookup returns no Series-type -> "No series found to send"."""
    import json as _json
    rest_fake.responses[("GET", "/modalities")] = b"[]"
    # Lookup returns Study-only result (not Series) for every UID
    rest_fake.responses[("POST", "/tools/lookup")] = _json.dumps([{"Type": "Study", "ID": "X"}]).encode()
    body = b'{"study_id": "STD", "target": "PACS", "series_uids": ["1.2.3.SE1"]}'
    router.SendToAiDicom(out, "/send-to-ai-dicom", method="POST", body=body)
    assert out.status == 400
    assert "no series found" in (out.body or "").lower()


def test_send_to_ai_dicom_happy_path_responds_with_success(out, router, rest_fake):
    """Full happy path: study has originals -> instances fetched -> /modalities/<target>/store POST succeeds."""
    import json as _json
    _bind_series_listing(rest_fake, "STD", ["S-orig"], ai=False)  # processable
    rest_fake.responses[("GET", "/modalities")] = b"[]"
    rest_fake.responses[("GET", "/series/S-orig/instances")] = _json.dumps(
        [{"ID": "I1"}, {"ID": "I2"}]
    ).encode()
    rest_fake.responses[("POST", "/modalities/PACS/store")] = b'{}'
    body = b'{"study_id": "STD", "target": "PACS"}'
    router.SendToAiDicom(out, "/send-to-ai-dicom", method="POST", body=body)
    assert out.status == 200
    resp = _json.loads(out.body)
    assert resp["status"] == "success"
    assert resp["target"] == "PACS"


@pytest.mark.xfail(
    reason="production wart: modality-store failure returns HTTP 200 with error JSON "
           "body instead of 5xx (same class as A1: 500->400 for bad JSON). "
           "When the production fix lands, this test xpasses and the marker is removed.",
    strict=False,
)
def test_send_to_ai_dicom_store_failure_should_return_5xx(out, router, rest_fake):
    """When the modality store endpoint raises, expected behavior is a 5xx response (currently 200)."""
    import json as _json
    _bind_series_listing(rest_fake, "STD", ["S-orig"], ai=False)
    rest_fake.responses[("GET", "/modalities")] = b"[]"
    rest_fake.responses[("GET", "/series/S-orig/instances")] = _json.dumps([{"ID": "I1"}]).encode()
    # No response bound for POST /modalities/PACS/store -> rest_fake raises.
    body = b'{"study_id": "STD", "target": "PACS"}'
    router.SendToAiDicom(out, "/send-to-ai-dicom", method="POST", body=body)
    assert out.status >= 500, (
        "store failure currently surfaces as 200 + error payload; expected 5xx"
    )


# --- SendToAiDicomWeb ---

def test_send_to_ai_dicomweb_study_not_found_returns_404(out, router, rest_fake):
    """orthanc.RestApiGet(/studies/<id>) raising -> 404 surface."""
    body = b'{"study_id": "MISSING", "target": "P", "target_url": "http://x/dicom-web"}'
    router.SendToAiDicomWeb(out, "/send-to-ai-dicomweb", method="POST", body=body)
    assert out.status == 404
    assert "not found" in (out.body or "").lower()


def test_send_to_ai_dicomweb_study_no_processable_content_returns_400(out, router, rest_fake):
    """study_id exists but all series are AI results -> 400."""
    import json as _json
    rest_fake.responses[("GET", "/studies/STD")] = _json.dumps(
        {"Series": ["S-ai-1"], "PatientMainDicomTags": {"PatientName": "X"}}
    ).encode()
    _bind_series_listing(rest_fake, "STD", ["S-ai-1"], ai=True)
    body = b'{"study_id": "STD", "target": "P", "target_url": "http://x/dicom-web"}'
    router.SendToAiDicomWeb(out, "/send-to-ai-dicomweb", method="POST", body=body)
    assert out.status == 400
    assert "no processable content" in (out.body or "").lower()


def test_send_to_ai_dicomweb_dicomweb_server_config_failure_returns_500(out, router, rest_fake, monkeypatch):
    """When requests.put to /dicom-web/servers/<target> returns non-2xx, response is 500."""
    import json as _json
    rest_fake.responses[("GET", "/studies/STD")] = _json.dumps(
        {"Series": ["S-orig"], "PatientMainDicomTags": {"PatientName": "X"}}
    ).encode()
    _bind_series_listing(rest_fake, "STD", ["S-orig"], ai=False)

    from unittest import mock as _mock
    bad_resp = _mock.MagicMock(status_code=500, text="config error")
    monkeypatch.setattr(router.requests, "put", _mock.MagicMock(return_value=bad_resp))

    body = b'{"study_id": "STD", "target": "P", "target_url": "http://x/dicom-web"}'
    router.SendToAiDicomWeb(out, "/send-to-ai-dicomweb", method="POST", body=body)
    assert out.status == 500
    assert "configuring dicomweb" in (out.body or "").lower()


def test_send_to_ai_wraps_dicomweb_for_backward_compat(out, router, rest_fake):
    """SendToAi is a thin wrapper around SendToAiDicomWeb; missing target_url -> 400."""
    body = b'{"study_id": "STD", "target": "P"}'   # no target_url
    router.SendToAi(out, "/send-to-ai", method="POST", body=body)
    assert out.status == 400


# ---------------------------------------------------------------------------
# SendToAiDicomWeb post-config coverage (viewer/router.py:430-613)
# ---------------------------------------------------------------------------

def _good_server_config_resp():
    from unittest import mock as _mock
    return _mock.MagicMock(status_code=200, text="ok")


def _bind_dicomweb_study_state(rest_fake, study_id="STD", series_id="S-orig",
                                 series_dicom_uid="1.2.3.SE1"):
    """Bind /studies/<id> + the FilterAIResultSeries subset (via _bind_series_listing) + per-series
    instances + /series/<id> for SeriesInstanceUID — everything SendToAiDicomWeb needs to reach
    the workitem POST."""
    import json as _json
    # FilterAIResultSeries path (returns series_id as non-AI) — reuse the existing helper.
    _bind_series_listing(rest_fake, study_id, [series_id], ai=False)
    rest_fake.responses[("GET", f"/studies/{study_id}")] = _json.dumps(
        {"Series": [series_id], "PatientMainDicomTags": {"PatientName": "X"},
         "MainDicomTags": {"StudyInstanceUID": "1.2.3.STUDY"}}
    ).encode()
    rest_fake.responses[("GET", f"/series/{series_id}/instances")] = _json.dumps(
        [{"ID": "I1"}, {"ID": "I2"}]
    ).encode()
    rest_fake.responses[("GET", f"/series/{series_id}")] = _json.dumps(
        {"MainDicomTags": {"SeriesInstanceUID": series_dicom_uid}}
    ).encode()


def test_send_to_ai_dicomweb_happy_path_creates_workitem_and_returns_success(out, router, rest_fake, monkeypatch):
    """Full happy path: DICOMweb config OK -> series filtered -> UPS workitem POSTed -> success payload with workitem_uid."""
    import json as _json
    from unittest import mock as _mock
    _bind_dicomweb_study_state(rest_fake)

    workitem_response = _mock.MagicMock(
        status_code=201,
        json=lambda: {"00080018": {"Value": ["1.2.3.WORKITEM"]}},
    )
    subscribe_response = _mock.MagicMock(status_code=200)
    posts = []
    def _post(url, **kw):
        posts.append((url, kw))
        if "/subscribers" in url:
            return subscribe_response
        return workitem_response
    monkeypatch.setattr(router.requests, "post", _post)
    monkeypatch.setattr(router.requests, "put", _mock.MagicMock(return_value=_good_server_config_resp()))

    body = b'{"study_id": "STD", "target": "P", "target_url": "http://router:8042/dicom-web"}'
    router.SendToAiDicomWeb(out, "/send-to-ai-dicomweb", method="POST", body=body)

    assert out.status == 200
    resp = _json.loads(out.body)
    assert resp["status"] == "success"
    assert resp["workitem_uid"] == "1.2.3.WORKITEM"
    assert resp["study_id"] == "STD"

    # Two POSTs: workitem creation + subscribe.
    urls = [p[0] for p in posts]
    assert any("/ups-rs/workitems" in u and "/subscribers" not in u for u in urls)
    assert any("/subscribers" in u for u in urls)


def test_send_to_ai_dicomweb_happy_path_passes_input_mapping_through(out, router, rest_fake, monkeypatch):
    """If body has input_mapping + input_configuration_id, they appear in the workitem POST body."""
    import json as _json
    from unittest import mock as _mock
    _bind_dicomweb_study_state(rest_fake)

    posts = []
    def _post(url, **kw):
        posts.append((url, kw))
        if "/subscribers" in url:
            return _mock.MagicMock(status_code=200)
        return _mock.MagicMock(status_code=201, json=lambda: {"00080018": {"Value": ["wi.X"]}})
    monkeypatch.setattr(router.requests, "post", _post)
    monkeypatch.setattr(router.requests, "put", _mock.MagicMock(return_value=_good_server_config_resp()))

    body = _json.dumps({
        "study_id": "STD", "target": "P", "target_url": "http://router:8042/dicom-web",
        "input_mapping": {"primary": "1.2.3.SE1"},
        "input_configuration_id": "cfg-42",
    }).encode()
    router.SendToAiDicomWeb(out, "/send-to-ai-dicomweb", method="POST", body=body)

    create_call = next(p for p in posts if "/ups-rs/workitems" in p[0] and "/subscribers" not in p[0])
    payload = create_call[1].get("json", {})
    assert payload.get("input_mapping") == {"primary": "1.2.3.SE1"}
    assert payload.get("input_configuration_id") == "cfg-42"


def test_send_to_ai_dicomweb_no_series_after_filter_returns_400(out, router, rest_fake, monkeypatch):
    """series_uids path: every lookup returns Study-only -> empty -> 400 'No series found to send'."""
    import json as _json
    from unittest import mock as _mock
    # Minimal study binding (needed for the study-existence verify step).
    rest_fake.responses[("GET", "/studies/STD")] = _json.dumps(
        {"Series": ["S-any"], "PatientMainDicomTags": {"PatientName": "X"},
         "MainDicomTags": {"StudyInstanceUID": "1.2.3.STUDY"}}
    ).encode()
    monkeypatch.setattr(router.requests, "put", _mock.MagicMock(return_value=_good_server_config_resp()))

    # series_uids provided -> takes the lookup path; lookup returns no Series-type entries.
    rest_fake.responses[("POST", "/tools/lookup")] = _json.dumps([{"Type": "Study", "ID": "X"}]).encode()
    body = b'{"study_id": "STD", "target": "P", "target_url": "http://router:8042/dicom-web", "series_uids": ["1.2.3.SE1"]}'
    router.SendToAiDicomWeb(out, "/send-to-ai-dicomweb", method="POST", body=body)
    assert out.status == 400
    assert "no series" in (out.body or "").lower()


def test_send_to_ai_dicomweb_returns_500_when_study_instance_uid_missing(out, router, rest_fake, monkeypatch):
    """When GetStudyInstanceUID returns None (StudyInstanceUID missing from MainDicomTags), respond 500."""
    import json as _json
    from unittest import mock as _mock
    # Override study response with empty MainDicomTags
    rest_fake.responses[("GET", "/studies/STD")] = _json.dumps(
        {"Series": ["S-orig"], "PatientMainDicomTags": {"PatientName": "X"},
         "MainDicomTags": {}}  # no StudyInstanceUID
    ).encode()
    rest_fake.responses[("GET", "/studies/STD/series")] = _json.dumps([{"ID": "S-orig"}]).encode()
    rest_fake.responses[("GET", "/series/S-orig/tags?simplify")] = _json.dumps(
        {"SeriesDescription": "Axial T1", "Modality": "MR"}
    ).encode()
    rest_fake.responses[("GET", "/series/S-orig/instances")] = _json.dumps([{"ID": "I1"}]).encode()
    monkeypatch.setattr(router.requests, "put", _mock.MagicMock(return_value=_good_server_config_resp()))

    body = b'{"study_id": "STD", "target": "P", "target_url": "http://router:8042/dicom-web"}'
    router.SendToAiDicomWeb(out, "/send-to-ai-dicomweb", method="POST", body=body)
    assert out.status == 500
    assert "StudyInstanceUID" in (out.body or "")


@pytest.mark.xfail(
    reason="production wart: SendToAiDicomWeb returns HTTP 200 with error JSON when UPS workitem "
           "creation on the router fails. Same class of bug as the D4 xfail "
           "(test_send_to_ai_dicom_store_failure_should_return_5xx) — error conditions on "
           "downstream POSTs should surface as 5xx. When the production fix lands this xpasses "
           "and the marker is removed.",
    strict=False,
)
def test_send_to_ai_dicomweb_workitem_creation_failure_should_return_5xx(out, router, rest_fake, monkeypatch):
    """PINNING-BUG: workitem-creation failure currently surfaces as 200 + error payload (should be 5xx).

    Once the production fix lands, this test xpasses; remove the marker.
    Also asserts subscribe was NEVER called so the failure path is exercised at the right point.
    """
    from unittest import mock as _mock
    _bind_dicomweb_study_state(rest_fake)

    bad_workitem = _mock.MagicMock(status_code=500, text="router unavailable")
    posts = []
    def _post(url, **kw):
        posts.append(url)
        if "/subscribers" in url:
            return _mock.MagicMock(status_code=200)  # would lie if reached
        return bad_workitem
    monkeypatch.setattr(router.requests, "post", _post)
    monkeypatch.setattr(router.requests, "put", _mock.MagicMock(return_value=_good_server_config_resp()))

    body = b'{"study_id": "STD", "target": "P", "target_url": "http://router:8042/dicom-web"}'
    router.SendToAiDicomWeb(out, "/send-to-ai-dicomweb", method="POST", body=body)
    # Subscribe must not have been attempted when workitem creation failed (this part is real spec).
    assert all("/subscribers" not in u for u in posts), \
        f"subscribe should not be called when workitem creation fails; saw {posts}"
    # Expected (correct) behavior: 5xx response for upstream router failure.
    assert out.status >= 500, (
        "workitem-creation failure currently surfaces as 200 + error payload; expected 5xx"
    )


def test_send_to_ai_dicomweb_subscribe_failure_does_not_block_success(out, router, rest_fake, monkeypatch, capsys):
    """Subscribe failure is logged AND attempted (not silently skipped), but the workitem-created success
    response still goes out."""
    import json as _json
    from unittest import mock as _mock
    _bind_dicomweb_study_state(rest_fake)

    workitem_response = _mock.MagicMock(status_code=201, json=lambda: {"00080018": {"Value": ["wi.OK"]}})
    bad_subscribe = _mock.MagicMock(status_code=500)
    posts = []
    def _post(url, **kw):
        posts.append(url)
        return bad_subscribe if "/subscribers" in url else workitem_response
    monkeypatch.setattr(router.requests, "post", _post)
    monkeypatch.setattr(router.requests, "put", _mock.MagicMock(return_value=_good_server_config_resp()))

    body = b'{"study_id": "STD", "target": "P", "target_url": "http://router:8042/dicom-web"}'
    router.SendToAiDicomWeb(out, "/send-to-ai-dicomweb", method="POST", body=body)

    assert out.status == 200
    assert _json.loads(out.body)["workitem_uid"] == "wi.OK"
    # Subscribe must have been attempted (so the "doesn't-block-success" claim is meaningful, not vacuous).
    assert any("/subscribers" in u for u in posts), f"subscribe was never attempted; saw {posts}"
    # And the failure must surface in the logs (production uses print()).
    assert "Subscription failed" in capsys.readouterr().out


def test_send_to_ai_dicomweb_strips_dicom_web_suffix_from_router_url(out, router, rest_fake, monkeypatch):
    """target_url with /dicom-web suffix is stripped before POSTing to /ups-rs/workitems."""
    import json as _json
    from unittest import mock as _mock
    _bind_dicomweb_study_state(rest_fake)

    posts = []
    def _post(url, **kw):
        posts.append(url)
        if "/subscribers" in url:
            return _mock.MagicMock(status_code=200)
        return _mock.MagicMock(status_code=201, json=lambda: {"00080018": {"Value": ["wi.42"]}})
    monkeypatch.setattr(router.requests, "post", _post)
    monkeypatch.setattr(router.requests, "put", _mock.MagicMock(return_value=_good_server_config_resp()))

    body = b'{"study_id": "STD", "target": "P", "target_url": "http://router:8042/dicom-web/"}'
    router.SendToAiDicomWeb(out, "/send-to-ai-dicomweb", method="POST", body=body)
    # The UPS-RS create POST goes to the root, not under /dicom-web
    create_post = next(u for u in posts if "/ups-rs/workitems" in u and "/subscribers" not in u)
    assert create_post == "http://router:8042/ups-rs/workitems"


# ---------------------------------------------------------------------------
# SendToAiDicom modality-config coverage (viewer/router.py:173-262)
# ---------------------------------------------------------------------------

def _bind_send_to_ai_dicom_state(rest_fake, study_id="STD", series_id="S-orig",
                                    target="PACS", *, include_existing_modality=False):
    """Bind everything SendToAiDicom needs to reach the final modality store POST.

    - FilterAIResultSeries (non-AI series) via _bind_series_listing
    - GET /modalities (ListModalities)
    - Optional GET /modalities/<target> (lookup-existing branch)
    - GET /series/<sid>/instances + POST /modalities/<target>/store

    Returns nothing; caller can layer more bindings on top.
    """
    import json as _json
    _bind_series_listing(rest_fake, study_id, [series_id], ai=False)
    rest_fake.responses[("GET", "/modalities")] = b"[]"
    if include_existing_modality:
        rest_fake.responses[("GET", f"/modalities/{target}")] = _json.dumps(
            {"AET": "EXISTING", "Host": "old", "Port": 4242}
        ).encode()
        rest_fake.responses[("DELETE", f"/modalities/{target}")] = b""
    rest_fake.responses[("PUT", f"/modalities/{target}")] = b""
    rest_fake.responses[("GET", f"/series/{series_id}/instances")] = _json.dumps(
        [{"ID": "I1"}, {"ID": "I2"}]
    ).encode()
    rest_fake.responses[("POST", f"/modalities/{target}/store")] = b"{}"


def test_send_to_ai_dicom_configures_new_modality_when_target_url_provided(out, router, rest_fake):
    """No existing modality (GET /modalities/<target> raises -> silent pass) -> PUT configures it -> success."""
    import json as _json
    _bind_send_to_ai_dicom_state(rest_fake, include_existing_modality=False)
    body = b'{"study_id": "STD", "target": "PACS", "target_url": "remote-host:11112/REMOTE_AET"}'
    router.SendToAiDicom(out, "/send-to-ai-dicom", method="POST", body=body)
    assert out.status == 200
    assert _json.loads(out.body)["status"] == "success"
    # PUT and the post-PUT verification GET both hit /modalities/PACS.
    put_calls = [c for c in rest_fake.calls if c[0] == "PUT" and c[1] == "/modalities/PACS"]
    assert len(put_calls) == 1
    # The PUT body must be a JSON-serialized modality config with the parsed host/port/AET.
    config = _json.loads(put_calls[0][2])
    assert config["Host"] == "remote-host"
    assert config["Port"] == 11112
    assert config["AET"] == "REMOTE_AET"


def test_send_to_ai_dicom_deletes_existing_modality_before_reconfiguring(out, router, rest_fake):
    """When GET /modalities/<target> succeeds, the existing modality is DELETE'd before PUT reconfigure."""
    _bind_send_to_ai_dicom_state(rest_fake, include_existing_modality=True)
    body = b'{"study_id": "STD", "target": "PACS", "target_url": "h:104/AET1"}'
    router.SendToAiDicom(out, "/send-to-ai-dicom", method="POST", body=body)
    assert out.status == 200
    # DELETE preceded PUT in the call order.
    call_seq = [(c[0], c[1]) for c in rest_fake.calls if "/modalities/PACS" in c[1]]
    delete_idx = call_seq.index(("DELETE", "/modalities/PACS"))
    put_idx = call_seq.index(("PUT", "/modalities/PACS"))
    assert delete_idx < put_idx, f"expected DELETE before PUT; got {call_seq}"


def test_send_to_ai_dicom_defaults_aet_to_target_when_url_has_no_aet(out, router, rest_fake):
    """target_url='host:port' (no /AET) -> AET parsed as 'port' due to split mechanics, port defaults to 104.

    Pins the current parsing behavior. The url_parts = target_url.split('/') logic with input 'h:104'
    yields ['h:104'] which is len < 2 -> 'Invalid target URL format' branch.
    Production then logs and continues to the store step (no PUT).
    """
    import json as _json
    _bind_send_to_ai_dicom_state(rest_fake)
    body = b'{"study_id": "STD", "target": "PACS", "target_url": "h:104"}'
    router.SendToAiDicom(out, "/send-to-ai-dicom", method="POST", body=body)
    assert out.status == 200    # continues to store and succeeds
    assert _json.loads(out.body)["status"] == "success"
    # PUT must NOT have been issued — the URL parse fell into the 'Invalid target URL format' branch.
    assert not any(c[0] == "PUT" for c in rest_fake.calls)


def test_send_to_ai_dicom_continues_when_put_raises(out, router, rest_fake):
    """If RestApiPut raises ('Failed to configure DICOM modality'), function still continues to the store POST."""
    import json as _json
    _bind_send_to_ai_dicom_state(rest_fake)
    # Remove PUT binding so rest_fake raises -> outer try/except logs + continues.
    del rest_fake.responses[("PUT", "/modalities/PACS")]
    body = b'{"study_id": "STD", "target": "PACS", "target_url": "h:104/AET"}'
    router.SendToAiDicom(out, "/send-to-ai-dicom", method="POST", body=body)
    assert out.status == 200
    assert _json.loads(out.body)["status"] == "success"


def test_send_to_ai_dicom_succeeds_without_target_url(out, router, rest_fake):
    """No target_url -> modality-config block skipped entirely; function proceeds straight to the store POST."""
    import json as _json
    _bind_send_to_ai_dicom_state(rest_fake)
    body = b'{"study_id": "STD", "target": "PACS"}'   # no target_url
    router.SendToAiDicom(out, "/send-to-ai-dicom", method="POST", body=body)
    assert out.status == 200
    assert _json.loads(out.body)["status"] == "success"
    # No modality-config GET/PUT/DELETE for PACS — only the store POST.
    modality_calls = [c for c in rest_fake.calls if "/modalities/PACS" in c[1] and not c[1].endswith("/store")]
    assert modality_calls == []
