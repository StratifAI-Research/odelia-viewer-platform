"""Unit tests for viewer/feedback_db.py — sqlite-backed feedback storage."""
import importlib
from types import ModuleType

import pytest


@pytest.fixture
def fb(tmp_path, monkeypatch) -> ModuleType:
    # Reset module-level singleton state so each test gets a fresh DB.
    monkeypatch.setenv('ORTHANC_FEEDBACK_DB_DIR', str(tmp_path))
    monkeypatch.setenv('ORTHANC_FEEDBACK_DB_PATH', str(tmp_path / 'fb.sqlite'))
    import feedback_db
    importlib.reload(feedback_db)
    return feedback_db


def _payload(study='1.2.3', user='u1', verdict_L=1, verdict_R=-1, **extra):
    return {
        'study_uid': study,
        'model_name': 'M',
        'model_version': '1',
        'result_ts': '2026-01-01T00:00:00Z',
        'user_id': user,
        'verdict_L': verdict_L,
        'verdict_R': verdict_R,
        **extra,
    }


def test_health_reports_db_ready(fb):
    info = fb.health()
    assert info['db_ready'] is True
    assert 'sqlite_version' in info


def test_register_result_idempotent(fb):
    r1 = fb.register_result('1.2.3', 'M', '1', '2026-01-01T00:00:00Z', None)
    r2 = fb.register_result('1.2.3', 'M', '1', '2026-01-01T00:00:00Z', None)
    assert r1['created'] is True
    assert r2['created'] is False
    assert r1['id'] == r2['id']


def test_submit_happy_path_returns_initial(fb):
    saved = fb.submit_feedback(_payload())
    assert saved['submission_kind'] == 'initial'
    assert saved['verdict_L'] == 1
    assert saved['verdict_R'] == -1
    assert saved['study_uid'] == '1.2.3'


def test_submit_duplicate_without_edited_raises_conflict(fb):
    fb.submit_feedback(_payload())
    with pytest.raises(fb.ConflictError):
        fb.submit_feedback(_payload())


def test_submit_edited_replaces_verdict(fb):
    fb.submit_feedback(_payload(verdict_L=1, verdict_R=0))
    saved = fb.submit_feedback(_payload(verdict_L=-1, verdict_R=1, edited=True))
    assert saved['submission_kind'] == 'edit'
    assert saved['verdict_L'] == -1
    assert saved['verdict_R'] == 1


def test_read_feedback_aggregates(fb):
    fb.submit_feedback(_payload(user='alice', verdict_L=1, verdict_R=-1))
    fb.submit_feedback(_payload(user='bob', verdict_L=1, verdict_R=1))
    agg = fb.read_feedback(
        study_uid='1.2.3', model_name='M', model_version='1',
        result_ts='2026-01-01T00:00:00Z',
        include_users=False, include_history=False,
    )
    assert agg['n_submissions'] >= 2
    assert agg['aggregate']['L']['agree'] >= 2


def test_read_feedback_no_submissions_returns_zeros(fb):
    agg = fb.read_feedback(
        study_uid='9.9.9', model_name='M', model_version='1',
        result_ts='2026-01-01T00:00:00Z',
    )
    assert agg['n_submissions'] == 0
    assert agg['aggregate']['L']['agree'] == 0


def test_export_ndjson_includes_submitted_row(fb):
    fb.submit_feedback(_payload())
    # export_rows_ndjson yields dicts (not strings)
    rows = list(fb.export_rows_ndjson(model_name='M', model_version='1', scope='current'))
    assert len(rows) == 1
    row = rows[0]
    assert row['user_id'] == 'u1'
    assert row['verdict_L'] == 1
    assert row['verdict_R'] == -1
    assert row['submission_kind'] == 'initial'


def test_export_csv_header_and_rows(fb):
    fb.submit_feedback(_payload())
    # export_rows_csv returns (header_str, iterable_of_tuples)
    header, rows_iter = fb.export_rows_csv(model_name='M', model_version='1', scope='current')
    assert 'submission_kind' in header
    assert 'study_uid' in header
    data_rows = list(rows_iter)
    assert len(data_rows) == 1
    # Each row is a tuple; the last column is submission_kind
    assert data_rows[0][-1] == 'initial'


# ---------------------------------------------------------------------------
# Edit-replace round-trip via read_feedback (v_feedback_current view contract)
# ---------------------------------------------------------------------------

def test_submit_edited_round_trip_aggregate_reflects_latest_only(fb):
    """After an edit, read_feedback's aggregate reflects ONLY the latest verdict.
    Pins the v_feedback_current view contract (current-row-per-user semantics)."""
    fb.submit_feedback(_payload(verdict_L=1, verdict_R=0))          # initial: agree-L, unsure-R
    fb.submit_feedback(_payload(verdict_L=-1, verdict_R=1, edited=True))  # edit:   disagree-L, agree-R
    agg = fb.read_feedback("1.2.3", "M", "1", "2026-01-01T00:00:00Z")["aggregate"]
    assert agg["L"] == {"agree": 0, "unsure": 0, "disagree": 1}
    assert agg["R"] == {"agree": 1, "unsure": 0, "disagree": 0}


def test_read_feedback_history_includes_both_initial_and_edit(fb):
    """include_history=True returns the full event log; both initial + edit appear in order."""
    fb.submit_feedback(_payload(verdict_L=1, verdict_R=0))
    fb.submit_feedback(_payload(verdict_L=-1, verdict_R=1, edited=True))
    data = fb.read_feedback("1.2.3", "M", "1", "2026-01-01T00:00:00Z", False, True)
    assert "history" in data
    assert len(data["history"]) == 2
    kinds = [h["submission_kind"] for h in data["history"]]
    assert kinds == ["initial", "edit"]


def test_read_feedback_include_users_lists_current_submitters(fb):
    """include_users=True returns one row per user (current view)."""
    fb.submit_feedback(_payload(user="alice", verdict_L=1, verdict_R=1))
    fb.submit_feedback(_payload(user="bob", verdict_L=0, verdict_R=-1))
    data = fb.read_feedback("1.2.3", "M", "1", "2026-01-01T00:00:00Z", True, False)
    assert "users" in data
    user_ids = sorted(u["user_id"] for u in data["users"])
    assert user_ids == ["alice", "bob"]


# ---------------------------------------------------------------------------
# Export filter coverage (V2 + V3: scope='current' SQL rewrite path)
# ---------------------------------------------------------------------------

def test_export_ndjson_model_name_filter_excludes_other_model(fb):
    fb.submit_feedback(_payload())
    fb.submit_feedback(_payload(study="9.9.9", model_name_override="OTHER") if False else {
        "study_uid": "9.9.9",
        "model_name": "OTHER",
        "model_version": "1",
        "result_ts": "2026-01-01T00:00:00Z",
        "user_id": "u1",
        "verdict_L": 1,
        "verdict_R": -1,
    })
    rows = list(fb.export_rows_ndjson(None, None, "M", None, "history"))
    assert all(r["model_name"] == "M" for r in rows)
    assert len(rows) == 1


def test_export_csv_since_future_returns_no_rows(fb):
    """since filter binds to e.created_at via parameter — far-future date excludes all."""
    fb.submit_feedback(_payload())
    header, rows_iter = fb.export_rows_csv(since="2099-01-01T00:00:00Z", scope="history")
    rows = list(rows_iter)
    assert header.startswith("study_uid,")
    assert rows == []


def test_export_csv_since_past_returns_all_rows(fb):
    """since filter with a date earlier than all rows returns everything."""
    fb.submit_feedback(_payload())
    header, rows_iter = fb.export_rows_csv(since="1970-01-01T00:00:00Z", scope="history")
    rows = list(rows_iter)
    assert len(rows) == 1


def test_export_ndjson_scope_current_returns_only_current_after_edit(fb):
    """scope='current' exercises the where.replace('e.','c.') SQL rewrite path."""
    fb.submit_feedback(_payload(verdict_L=1, verdict_R=0))            # initial
    fb.submit_feedback(_payload(verdict_L=-1, verdict_R=1, edited=True))  # edit
    rows_current = list(fb.export_rows_ndjson(None, None, None, None, "current"))
    rows_history = list(fb.export_rows_ndjson(None, None, None, None, "history"))
    assert len(rows_current) == 1                      # current view: latest only
    assert rows_current[0]["verdict_L"] == -1
    assert rows_current[0]["submission_kind"] == "edit"
    assert len(rows_history) == 2                      # history: both events


def test_export_ndjson_scope_current_with_since_filter(fb):
    """scope='current' + since filter exercises the rewritten 'c.created_at >= ?' clause."""
    fb.submit_feedback(_payload())
    rows = list(fb.export_rows_ndjson("1970-01-01T00:00:00Z", None, None, None, "current"))
    assert len(rows) == 1


def test_read_feedback_with_both_include_users_and_include_history(fb):
    """Both flags True -> returns both users[] (current view) and history[] (all events)."""
    fb.submit_feedback(_payload(user="alice", verdict_L=1, verdict_R=0))
    fb.submit_feedback(_payload(user="alice", verdict_L=-1, verdict_R=1, edited=True))
    fb.submit_feedback(_payload(user="bob", verdict_L=0, verdict_R=0))
    data = fb.read_feedback("1.2.3", "M", "1", "2026-01-01T00:00:00Z", True, True)
    assert "users" in data
    assert "history" in data
    user_ids = sorted(u["user_id"] for u in data["users"])
    assert user_ids == ["alice", "bob"]            # current view -> 2 users
    assert len(data["history"]) == 3                # full event log -> 3 events
