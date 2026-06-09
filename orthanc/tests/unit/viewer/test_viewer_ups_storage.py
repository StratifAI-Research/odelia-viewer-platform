"""Unit tests for viewer/ups/storage.py — KV-backed UPS workitem storage."""
import sys
from typing import Any

import pytest


@pytest.fixture
def storage() -> Any:
    # Evict cached module so KV-backed state is fresh per test
    for key in [k for k in sys.modules if k == "ups" or k.startswith("ups.")]:
        del sys.modules[key]
    from ups.storage import UPSStorage
    return UPSStorage()


@pytest.fixture
def workitem() -> Any:
    """Return a minimal UPSWorkitem for use in storage tests."""
    for key in [k for k in sys.modules if k == "ups" or k.startswith("ups.")]:
        del sys.modules[key]
    from ups.workitem import UPSWorkitem
    return UPSWorkitem(
        study_uid="1.2.3.4",
        series_uids=["1.2.3.4.1"],
        wado_rs_retrieval=[
            {
                "retrieval_url": "http://orthanc:8042/wado-rs/studies/1.2.3.4",
                "study_uid": "1.2.3.4",
                "series_uid": "1.2.3.4.1",
            }
        ],
        priority="MEDIUM",
        workitem_uid="2.25.99999",
    )


# ---------------------------------------------------------------------------
# Round-trip: store then get
# ---------------------------------------------------------------------------

def test_store_and_get_returns_workitem(storage, workitem):
    storage.store_workitem(workitem)
    retrieved = storage.get_workitem(workitem.workitem_uid)
    assert retrieved is not None
    assert retrieved.workitem_uid == workitem.workitem_uid


def test_get_workitem_preserves_study_uid(storage, workitem):
    storage.store_workitem(workitem)
    retrieved = storage.get_workitem(workitem.workitem_uid)
    assert retrieved.get_study_uid() == "1.2.3.4"


def test_get_workitem_preserves_state(storage, workitem):
    storage.store_workitem(workitem)
    retrieved = storage.get_workitem(workitem.workitem_uid)
    assert retrieved.get_state() == "SCHEDULED"


def test_get_workitem_missing_key_returns_none(storage):
    result = storage.get_workitem("does.not.exist")
    assert result is None


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

def test_delete_removes_from_get(storage, workitem):
    storage.store_workitem(workitem)
    storage.delete_workitem(workitem.workitem_uid)
    assert storage.get_workitem(workitem.workitem_uid) is None


def test_delete_removes_from_list(storage, workitem):
    storage.store_workitem(workitem)
    storage.delete_workitem(workitem.workitem_uid)
    items = storage.list_workitems()
    assert workitem.workitem_uid not in [w.workitem_uid for w in items]


def test_delete_nonexistent_does_not_raise(storage):
    storage.delete_workitem("phantom.uid")


# ---------------------------------------------------------------------------
# list_workitems
# ---------------------------------------------------------------------------

def test_list_workitems_returns_stored(storage, workitem):
    storage.store_workitem(workitem)
    items = storage.list_workitems()
    assert len(items) == 1
    assert items[0].workitem_uid == workitem.workitem_uid


def test_list_workitems_empty_when_nothing_stored(storage):
    assert storage.list_workitems() == []


def test_list_workitems_state_filter_matches(storage, workitem):
    storage.store_workitem(workitem)
    items = storage.list_workitems(state="SCHEDULED")
    assert len(items) == 1
    assert items[0].workitem_uid == workitem.workitem_uid


def test_list_workitems_state_filter_excludes_other(storage, workitem):
    storage.store_workitem(workitem)
    items = storage.list_workitems(state="COMPLETED")
    assert items == []


def test_list_workitems_multiple_entries(storage):
    for key in [k for k in sys.modules if k == "ups" or k.startswith("ups.")]:
        del sys.modules[key]
    from ups.workitem import UPSWorkitem
    for uid in ["2.25.1", "2.25.2", "2.25.3"]:
        w = UPSWorkitem(
            study_uid="1.1.1",
            series_uids=["1.1.1.1"],
            wado_rs_retrieval=[
                {"retrieval_url": "http://x", "study_uid": "1.1.1", "series_uid": "1.1.1.1"}
            ],
            priority="LOW",
            workitem_uid=uid,
        )
        storage.store_workitem(w)
    items = storage.list_workitems()
    assert len(items) == 3


# ---------------------------------------------------------------------------
# Coverage fill: get_workitem malformed-bytes / delete swallow / _get_index corrupt (viewer side)
# ---------------------------------------------------------------------------

def test_get_workitem_returns_none_on_malformed_kv_bytes():
    import orthanc
    from ups.storage import ups_storage, UPSStorage
    orthanc.StoreKeyValue(UPSStorage.BUCKET, f"{UPSStorage.KEY_PREFIX}garbled.uid", b"not-valid-json")
    assert ups_storage.get_workitem("garbled.uid") is None


def test_delete_workitem_swallows_exception_and_leaves_state_intact(capsys):
    """Same contract as the router twin: DeleteKeyValue failure must not corrupt state or raise."""
    import orthanc
    from ups.storage import ups_storage, UPSStorage
    orthanc.StoreKeyValue(UPSStorage.BUCKET, f"{UPSStorage.KEY_PREFIX}survivor.uid", b'{"data": "x"}')
    orthanc.StoreKeyValue(UPSStorage.BUCKET, UPSStorage.INDEX_KEY, b'["survivor.uid"]')
    before_index = sorted(w.workitem_uid for w in ups_storage.list_workitems())

    original = orthanc.DeleteKeyValue
    def _raise(*a, **kw):
        raise RuntimeError("simulated DeleteKeyValue failure")
    orthanc.DeleteKeyValue = _raise
    try:
        ups_storage.delete_workitem("survivor.uid")
    finally:
        orthanc.DeleteKeyValue = original

    captured = capsys.readouterr().out
    assert "Error deleting" in captured, f"expected error log, got: {captured!r}"
    assert sorted(w.workitem_uid for w in ups_storage.list_workitems()) == before_index


def test_get_index_returns_empty_list_when_index_value_is_corrupt():
    import orthanc
    from ups.storage import ups_storage, UPSStorage
    orthanc.StoreKeyValue(UPSStorage.BUCKET, UPSStorage.INDEX_KEY, b"\xff\xfe\xff{not-json")
    assert ups_storage.list_workitems() == []
