"""Unit tests for router/ups/storage.py — KV-backed UPS workitem storage.

The router variant uses no-colon key prefixes (KEY_PREFIX="upsworkitem",
INDEX_KEY="upsworkitemindex") unlike the viewer variant which uses colons.
All behaviour is otherwise identical to the viewer-side storage.
"""
import sys
import os
from typing import Any, Iterator

import pytest

_ROUTER_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'router')
)


def _ensure_router_path():
    """Ensure router/ is at front of sys.path so ups.* resolves to router/ups/."""
    if _ROUTER_DIR in sys.path:
        sys.path.remove(_ROUTER_DIR)
    sys.path.insert(0, _ROUTER_DIR)


@pytest.fixture(autouse=True)
def _router_path_guard() -> Iterator[None]:
    """Ensure router/ is at sys.path[0] for each test; restore after."""
    saved = list(sys.path)
    _ensure_router_path()
    yield
    sys.path[:] = saved


@pytest.fixture
def storage() -> Any:
    _ensure_router_path()
    for key in [k for k in sys.modules if k == "ups" or k.startswith("ups.")]:
        del sys.modules[key]
    from ups.storage import UPSStorage
    return UPSStorage()


@pytest.fixture
def workitem() -> Any:
    _ensure_router_path()
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
# Key-prefix differences: router uses bare strings without colons
# ---------------------------------------------------------------------------

def test_router_key_prefix_has_no_colon(storage):
    assert ":" not in storage.KEY_PREFIX
    assert ":" not in storage.INDEX_KEY


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
    _ensure_router_path()
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
# State-update round-trip
# ---------------------------------------------------------------------------

def test_state_update_persists_after_re_store(storage, workitem):
    storage.store_workitem(workitem)
    retrieved = storage.get_workitem(workitem.workitem_uid)
    retrieved.update_state("IN_PROGRESS", progress_percent=50)
    storage.store_workitem(retrieved)
    retrieved2 = storage.get_workitem(workitem.workitem_uid)
    assert retrieved2.get_state() == "IN_PROGRESS"


# ---------------------------------------------------------------------------
# Index integrity: no duplicate entries
# ---------------------------------------------------------------------------

def test_store_twice_no_duplicate_in_list(storage, workitem):
    storage.store_workitem(workitem)
    storage.store_workitem(workitem)
    items = storage.list_workitems()
    uids = [w.workitem_uid for w in items]
    assert uids.count(workitem.workitem_uid) == 1


# ---------------------------------------------------------------------------
# Coverage fill: get_workitem returns None when stored bytes are malformed (lines 56-58 in storage.py).
# delete_workitem swallows error when the key doesn't exist after we've cleared the stub's lenient _del.
# _get_index falls back to [] when the index value is corrupt JSON (line 103-104).
# ---------------------------------------------------------------------------

def test_get_workitem_returns_none_on_malformed_kv_bytes():
    """Stored value cannot be decoded as a valid workitem JSON -> get_workitem returns None."""
    import orthanc
    from ups.storage import ups_storage, UPSStorage
    # Put garbage bytes at the expected key.
    orthanc.StoreKeyValue(UPSStorage.BUCKET, f"{UPSStorage.KEY_PREFIX}garbled.uid", b"not-valid-json-bytes")
    assert ups_storage.get_workitem("garbled.uid") is None


def test_delete_workitem_swallows_exception_and_leaves_state_intact(capsys):
    """If orthanc.DeleteKeyValue raises (real Orthanc semantics), the wrapper logs and
    leaves index + KV state intact. Pins the no-corruption guarantee."""
    import orthanc
    from ups.storage import ups_storage, UPSStorage
    # Seed an existing item so we can prove its index entry survives a failed delete.
    orthanc.StoreKeyValue(UPSStorage.BUCKET, f"{UPSStorage.KEY_PREFIX}survivor.uid", b'{"data": "x"}')
    orthanc.StoreKeyValue(UPSStorage.BUCKET, UPSStorage.INDEX_KEY,
                          b'["survivor.uid"]')
    before_index = sorted(w.workitem_uid for w in ups_storage.list_workitems())

    original = orthanc.DeleteKeyValue
    def _raise(*a, **kw):
        raise RuntimeError("simulated orthanc DeleteKeyValue failure")
    orthanc.DeleteKeyValue = _raise
    try:
        ups_storage.delete_workitem("survivor.uid")  # must NOT raise
    finally:
        orthanc.DeleteKeyValue = original

    # Production uses print(); capsys captures it.
    captured = capsys.readouterr().out
    assert "Error deleting" in captured, f"expected error log, got: {captured!r}"
    # Index entry must survive — the function MUST NOT call _remove_from_index when the KV delete failed.
    assert sorted(w.workitem_uid for w in ups_storage.list_workitems()) == before_index


def test_get_index_returns_empty_list_when_index_value_is_corrupt():
    """Corrupt index bytes -> bare except returns []; list_workitems then returns []."""
    import orthanc
    from ups.storage import ups_storage, UPSStorage
    orthanc.StoreKeyValue(UPSStorage.BUCKET, UPSStorage.INDEX_KEY, b"\xff\xfe\xff{not-json")
    assert ups_storage.list_workitems() == []
