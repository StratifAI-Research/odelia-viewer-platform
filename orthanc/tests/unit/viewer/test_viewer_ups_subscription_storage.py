"""Unit tests for viewer/ups/subscription_storage.py — KV-backed subscription registry."""
import sys
from typing import Any

import pytest


@pytest.fixture
def sub_storage() -> Any:
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


def test_add_subscription_same_pair_overwrites(sub_storage):
    sub_storage.add_subscription("wuid.2", "http://sub-b")
    sub_storage.add_subscription("wuid.2", "http://sub-b")
    subs = sub_storage.get_subscribers("wuid.2")
    # Same (uid, url) pair overwrites the existing KV entry (key collision)
    assert subs.count("http://sub-b") == 1


def test_add_multiple_subscribers_for_same_workitem(sub_storage):
    sub_storage.add_subscription("wuid.3", "http://sub-c")
    sub_storage.add_subscription("wuid.3", "http://sub-d")
    subs = sub_storage.get_subscribers("wuid.3")
    assert "http://sub-c" in subs
    assert "http://sub-d" in subs


def test_subscribers_are_isolated_per_workitem(sub_storage):
    sub_storage.add_subscription("wuid.4", "http://sub-e")
    # wuid.5 should have no subscribers
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
    # global list should not contain duplicates
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
    import json
    import orthanc
    # Use the public Orthanc iterator API to find the stored record and verify
    # the deletion_lock field — avoids coupling to the stub's internal _kv key tuple.
    it = orthanc.CreateKeysValuesIterator("ups_subscriptions")
    lock_record = None
    while it.Next():
        key = it.GetKey()
        if "wuid.lock" in key and "http://sub-lock" in key:
            lock_record = json.loads(it.GetValue().decode("utf-8"))
            break
    assert lock_record is not None, "subscription record for wuid.lock not found in KV store"
    assert lock_record["deletion_lock"] is True
