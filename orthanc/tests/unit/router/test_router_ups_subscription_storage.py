"""Unit tests for router/ups/subscription_storage.py — KV subscription registry.

Router and viewer subscription_storage.py are structurally identical (diff shows
no differences). Tests mirror the viewer-side suite to confirm the router copy
behaves identically under the router conftest's sys.path.
"""
import json
import os
import sys
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
def sub_storage() -> Any:
    _ensure_router_path()
    for key in [k for k in sys.modules if k == "ups" or k.startswith("ups.")]:
        del sys.modules[key]
    from ups.subscription_storage import UPSSubscriptionStorage
    return UPSSubscriptionStorage()


# ---------------------------------------------------------------------------
# add_subscription / get_subscribers (per-workitem)
# ---------------------------------------------------------------------------

def test_add_subscription_and_get_subscribers(sub_storage):
    sub_storage.add_subscription("wuid.1", "http://sub-a")
    subs = sub_storage.get_subscribers("wuid.1")
    assert "http://sub-a" in subs


def test_get_subscribers_unknown_workitem_returns_empty(sub_storage):
    subs = sub_storage.get_subscribers("wuid.missing")
    assert subs == []


def test_add_subscription_idempotent(sub_storage):
    sub_storage.add_subscription("wuid.2", "http://sub-b")
    sub_storage.add_subscription("wuid.2", "http://sub-b")
    subs = sub_storage.get_subscribers("wuid.2")
    # Dedup: subscriber should appear only once
    assert subs.count("http://sub-b") == 1


def test_add_multiple_subscribers_for_same_workitem(sub_storage):
    sub_storage.add_subscription("wuid.3", "http://sub-c")
    sub_storage.add_subscription("wuid.3", "http://sub-d")
    subs = sub_storage.get_subscribers("wuid.3")
    assert "http://sub-c" in subs
    assert "http://sub-d" in subs


def test_subscribers_are_isolated_per_workitem(sub_storage):
    sub_storage.add_subscription("wuid.4", "http://sub-e")
    subs = sub_storage.get_subscribers("wuid.5")
    assert "http://sub-e" not in subs


# ---------------------------------------------------------------------------
# remove_subscription
# ---------------------------------------------------------------------------

def test_remove_subscription_removes_subscriber(sub_storage):
    sub_storage.add_subscription("wuid.6", "http://sub-f")
    sub_storage.remove_subscription("wuid.6", "http://sub-f")
    subs = sub_storage.get_subscribers("wuid.6")
    assert "http://sub-f" not in subs


def test_remove_nonexistent_subscription_does_not_raise(sub_storage):
    sub_storage.remove_subscription("wuid.999", "http://phantom")


# ---------------------------------------------------------------------------
# add_global_subscription / get_subscribers includes global
# ---------------------------------------------------------------------------

def test_global_subscription_appears_for_any_workitem(sub_storage):
    sub_storage.add_global_subscription("http://global-sub")
    subs_a = sub_storage.get_subscribers("wuid.a")
    subs_b = sub_storage.get_subscribers("wuid.b")
    assert "http://global-sub" in subs_a
    assert "http://global-sub" in subs_b


def test_global_subscription_idempotent(sub_storage):
    sub_storage.add_global_subscription("http://global-x")
    sub_storage.add_global_subscription("http://global-x")
    subs = sub_storage.get_subscribers("wuid.any")
    assert subs.count("http://global-x") == 1


def test_global_and_local_subscription_merged(sub_storage):
    sub_storage.add_subscription("wuid.m", "http://local-sub")
    sub_storage.add_global_subscription("http://global-sub")
    subs = sub_storage.get_subscribers("wuid.m")
    assert "http://local-sub" in subs
    assert "http://global-sub" in subs


def test_deletion_lock_stored(sub_storage):
    sub_storage.add_subscription("wuid.lock", "http://sub-lock", deletion_lock=True)
    import orthanc
    it = orthanc.CreateKeysValuesIterator("ups_subscriptions")
    lock_record = None
    while it.Next():
        key = it.GetKey()
        if "wuid.lock" in key and "http://sub-lock" in key:
            lock_record = json.loads(it.GetValue().decode("utf-8"))
            break
    assert lock_record is not None, "subscription record for wuid.lock not found in KV store"
    assert lock_record["deletion_lock"] is True


def test_local_sub_not_in_unrelated_workitem(sub_storage):
    sub_storage.add_subscription("wuid.p", "http://sub-p")
    sub_storage.add_subscription("wuid.q", "http://sub-q")
    subs_p = sub_storage.get_subscribers("wuid.p")
    assert "http://sub-q" not in subs_p
