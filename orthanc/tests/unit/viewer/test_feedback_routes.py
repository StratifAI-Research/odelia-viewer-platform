"""Unit tests for viewer/feedback_routes.py — REST handler functions."""
import importlib
import json
from types import ModuleType

import pytest


@pytest.fixture
def fb_routes(tmp_path, monkeypatch) -> ModuleType:
    """Reload feedback_db and feedback_routes with isolated SQLite DB."""
    monkeypatch.setenv("ORTHANC_FEEDBACK_DB_DIR", str(tmp_path))
    monkeypatch.setenv("ORTHANC_FEEDBACK_DB_PATH", str(tmp_path / "fb.sqlite"))
    import feedback_db
    importlib.reload(feedback_db)
    import feedback_routes
    importlib.reload(feedback_routes)
    return feedback_routes


def _valid_submit():
    return {
        "study_uid": "1.2.3",
        "model_name": "M",
        "model_version": "1",
        "result_ts": "2026-01-01T00:00:00Z",
        "user_id": "u1",
        "verdict_L": 1,
        "verdict_R": -1,
    }


# ---------------------------------------------------------------------------
# FeedbackHealth
# ---------------------------------------------------------------------------

def test_health_get_returns_200(out, fb_routes):
    fb_routes.FeedbackHealth(out, "/feedback/health", method="GET")
    assert out.status == 200
    data = json.loads(out.body)
    assert data["db_ready"] is True


def test_health_post_returns_405(out, fb_routes):
    fb_routes.FeedbackHealth(out, "/feedback/health", method="POST")
    assert out.status == 405


# ---------------------------------------------------------------------------
# FeedbackSubmit
# ---------------------------------------------------------------------------

def test_submit_happy_path_returns_201(out, fb_routes):
    fb_routes.FeedbackSubmit(
        out, "/feedback/submit",
        method="POST",
        body=json.dumps(_valid_submit()),
    )
    assert out.status == 201
    data = json.loads(out.body)
    assert data["submission_kind"] == "initial"
    assert data["verdict_L"] == 1


def test_submit_invalid_json_returns_400(out, fb_routes):
    fb_routes.FeedbackSubmit(
        out, "/feedback/submit",
        method="POST",
        body="not-json",
    )
    assert out.status == 400


def test_submit_missing_fields_returns_400(out, fb_routes):
    fb_routes.FeedbackSubmit(
        out, "/feedback/submit",
        method="POST",
        body=json.dumps({"study_uid": "x"}),
    )
    assert out.status == 400
    assert "Missing fields" in out.body


def test_submit_wrong_method_returns_405(out, fb_routes):
    fb_routes.FeedbackSubmit(out, "/feedback/submit", method="GET", body=b"")
    assert out.status == 405


def test_submit_duplicate_returns_409(out, fb_routes):
    payload = json.dumps(_valid_submit())
    fb_routes.FeedbackSubmit(out, "/feedback/submit", method="POST", body=payload)
    assert out.status == 201
    out2 = type(out).__new__(type(out))
    out2.__init__()
    fb_routes.FeedbackSubmit(out2, "/feedback/submit", method="POST", body=payload)
    assert out2.status == 409


def test_submit_invalid_verdict_returns_400(out, fb_routes):
    p = dict(_valid_submit(), verdict_L=99)
    fb_routes.FeedbackSubmit(
        out, "/feedback/submit",
        method="POST",
        body=json.dumps(p),
    )
    assert out.status == 400


# ---------------------------------------------------------------------------
# FeedbackRegisterResult
# ---------------------------------------------------------------------------

def test_register_result_happy_path_returns_201(out, fb_routes):
    fb_routes.FeedbackRegisterResult(
        out, "/feedback/register-result",
        method="POST",
        body=json.dumps({
            "study_uid": "1.2.3",
            "model_name": "M",
            "model_version": "1",
            "result_ts": "2026-01-01T00:00:00Z",
        }),
    )
    assert out.status == 201
    data = json.loads(out.body)
    assert data["created"] is True


def test_register_result_idempotent_returns_200(out, fb_routes):
    body = json.dumps({
        "study_uid": "1.2.3",
        "model_name": "M",
        "model_version": "1",
        "result_ts": "2026-01-01T00:00:00Z",
    })
    fb_routes.FeedbackRegisterResult(out, "/feedback/register-result", method="POST", body=body)
    out2 = type(out).__new__(type(out))
    out2.__init__()
    fb_routes.FeedbackRegisterResult(out2, "/feedback/register-result", method="POST", body=body)
    assert out2.status == 200
    assert json.loads(out2.body)["created"] is False


def test_register_result_missing_fields_returns_400(out, fb_routes):
    fb_routes.FeedbackRegisterResult(
        out, "/feedback/register-result",
        method="POST",
        body=json.dumps({"study_uid": "only-one-field"}),
    )
    assert out.status == 400


def test_register_result_wrong_method_returns_405(out, fb_routes):
    fb_routes.FeedbackRegisterResult(
        out, "/feedback/register-result",
        method="GET", body=b"",
    )
    assert out.status == 405


def test_register_result_invalid_json_returns_400(out, fb_routes):
    fb_routes.FeedbackRegisterResult(
        out, "/feedback/register-result",
        method="POST", body="bad json",
    )
    assert out.status == 400


# ---------------------------------------------------------------------------
# FeedbackRead (GET /feedback)
# ---------------------------------------------------------------------------

def test_feedback_read_returns_200_with_data(out, fb_routes):
    # First submit something so there is data
    fb_routes.FeedbackSubmit(
        out, "/feedback/submit",
        method="POST",
        body=json.dumps(_valid_submit()),
    )
    out2 = type(out).__new__(type(out))
    out2.__init__()
    fb_routes.FeedbackRead(
        out2, "/feedback",
        method="GET",
        get={
            "study_uid": "1.2.3",
            "model_name": "M",
            "model_version": "1",
            "result_ts": "2026-01-01T00:00:00Z",
        },
    )
    assert out2.status == 200
    data = json.loads(out2.body)
    assert "n_submissions" in data


def test_feedback_read_missing_params_returns_400(out, fb_routes):
    fb_routes.FeedbackRead(
        out, "/feedback",
        method="GET",
        get={"study_uid": "1.2.3"},
    )
    assert out.status == 400


def test_feedback_read_wrong_method_returns_405(out, fb_routes):
    fb_routes.FeedbackRead(out, "/feedback", method="POST", body=b"")
    assert out.status == 405


# ---------------------------------------------------------------------------
# FeedbackExportNdjson
# ---------------------------------------------------------------------------

def test_export_ndjson_returns_200(out, fb_routes):
    fb_routes.FeedbackSubmit(
        out, "/feedback/submit",
        method="POST",
        body=json.dumps(_valid_submit()),
    )
    out2 = type(out).__new__(type(out))
    out2.__init__()
    fb_routes.FeedbackExportNdjson(out2, "/feedback/export.ndjson", method="GET", get={})
    assert out2.status == 200


def test_export_ndjson_wrong_method_returns_405(out, fb_routes):
    fb_routes.FeedbackExportNdjson(out, "/feedback/export.ndjson", method="POST", body=b"")
    assert out.status == 405


# ---------------------------------------------------------------------------
# FeedbackExportCsv
# ---------------------------------------------------------------------------

def test_export_csv_returns_200(out, fb_routes):
    fb_routes.FeedbackSubmit(
        out, "/feedback/submit",
        method="POST",
        body=json.dumps(_valid_submit()),
    )
    out2 = type(out).__new__(type(out))
    out2.__init__()
    fb_routes.FeedbackExportCsv(out2, "/feedback/export.csv", method="GET", get={})
    assert out2.status == 200
    assert "submission_kind" in out2.body


def test_export_csv_wrong_method_returns_405(out, fb_routes):
    fb_routes.FeedbackExportCsv(out, "/feedback/export.csv", method="DELETE", body=b"")
    assert out.status == 405


# ---------------------------------------------------------------------------
# register_feedback_endpoints — covers all 6 routes
# ---------------------------------------------------------------------------

def test_register_feedback_endpoints_registers_6_routes(fb_routes):
    import orthanc
    orthanc._rest_callbacks.clear()
    fb_routes.register_feedback_endpoints()
    uris = [u for u, _ in orthanc._rest_callbacks]
    assert "/feedback/submit" in uris
    assert "/feedback" in uris
    assert "/feedback/register-result" in uris
    assert "/feedback/export.ndjson" in uris
    assert "/feedback/export.csv" in uris
    assert "/feedback/health" in uris
    assert len(uris) == 6


# ---------------------------------------------------------------------------
# V4: 409 body shape on duplicate submit (frontend depends on {code, message})
# ---------------------------------------------------------------------------

def test_submit_duplicate_409_body_has_expected_shape(out, fb_routes):
    fb_routes.FeedbackSubmit(out, "/feedback/submit", method="POST", body=json.dumps(_valid_submit()))
    out2 = type(out)()
    fb_routes.FeedbackSubmit(out2, "/feedback/submit", method="POST", body=json.dumps(_valid_submit()))
    assert out2.status == 409
    body = json.loads(out2.body)
    assert body == {"code": 409, "message": "Already submitted; use edit flow"}


# ---------------------------------------------------------------------------
# V2: FeedbackRead forwards includeUsers/includeHistory query params
# ---------------------------------------------------------------------------

def test_feedback_read_include_users_propagates_to_response(out, fb_routes):
    fb_routes.FeedbackSubmit(out, "/feedback/submit", method="POST", body=json.dumps(_valid_submit()))
    out2 = type(out)()
    fb_routes.FeedbackRead(
        out2, "/feedback", method="GET",
        get={
            "study_uid": "1.2.3", "model_name": "M", "model_version": "1",
            "result_ts": "2026-01-01T00:00:00Z", "includeUsers": "true",
        },
    )
    assert out2.status == 200
    data = json.loads(out2.body)
    assert "users" in data
    assert len(data["users"]) == 1


def test_feedback_read_include_history_propagates_to_response(out, fb_routes):
    fb_routes.FeedbackSubmit(out, "/feedback/submit", method="POST", body=json.dumps(_valid_submit()))
    p2 = {**_valid_submit(), "verdict_L": -1, "edited": True}
    out2 = type(out)()
    fb_routes.FeedbackSubmit(out2, "/feedback/submit", method="POST", body=json.dumps(p2))
    out3 = type(out)()
    fb_routes.FeedbackRead(
        out3, "/feedback", method="GET",
        get={
            "study_uid": "1.2.3", "model_name": "M", "model_version": "1",
            "result_ts": "2026-01-01T00:00:00Z", "includeHistory": "true",
        },
    )
    assert out3.status == 200
    data = json.loads(out3.body)
    assert "history" in data
    assert len(data["history"]) == 2


def test_export_ndjson_scope_current_returns_current_row_only(out, fb_routes):
    """Exercise the SQL-rewrite path through the route handler."""
    fb_routes.FeedbackSubmit(out, "/feedback/submit", method="POST", body=json.dumps(_valid_submit()))
    p2 = {**_valid_submit(), "verdict_L": -1, "edited": True}
    out2 = type(out)()
    fb_routes.FeedbackSubmit(out2, "/feedback/submit", method="POST", body=json.dumps(p2))
    out3 = type(out)()
    fb_routes.FeedbackExportNdjson(out3, "/feedback/export.ndjson", method="GET",
                                    get={"scope": "current"})
    assert out3.status == 200
    lines = [l for l in out3.body.split("\n") if l]
    assert len(lines) == 1                       # current view: only latest event
    obj = json.loads(lines[0])
    assert obj["verdict_L"] == -1
    assert obj["submission_kind"] == "edit"


def test_export_csv_model_name_filter_excludes_other(out, fb_routes):
    """model_name query param is forwarded into the SQL WHERE clause."""
    fb_routes.FeedbackSubmit(out, "/feedback/submit", method="POST", body=json.dumps(_valid_submit()))
    other = {**_valid_submit(), "study_uid": "9.9.9", "model_name": "OTHER"}
    out2 = type(out)()
    fb_routes.FeedbackSubmit(out2, "/feedback/submit", method="POST", body=json.dumps(other))
    out3 = type(out)()
    fb_routes.FeedbackExportCsv(out3, "/feedback/export.csv", method="GET",
                                 get={"model_name": "M"})
    assert out3.status == 200
    # Header + 1 data row for model M; "OTHER" model excluded.
    lines = [l for l in out3.body.split("\n") if l]
    assert len(lines) == 2
    assert "M," in lines[1]
    assert "OTHER," not in lines[1]
