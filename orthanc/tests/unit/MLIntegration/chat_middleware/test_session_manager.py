"""Tests for chat-middleware/session_manager.py — in-memory chat sessions with asyncio primitives."""
import asyncio
from datetime import datetime, timedelta

import pytest


def test_create_session_with_explicit_id():
    from session_manager import SessionManager
    sm = SessionManager()
    s = sm.create_session("sess-1")
    assert s.session_id == "sess-1"
    assert sm.get_session("sess-1") is s
    assert s.conversation_history == []


def test_create_session_without_id_generates_uuid():
    from session_manager import SessionManager
    sm = SessionManager()
    s = sm.create_session()
    # UUID4 string format: 36 chars with 4 hyphens
    assert len(s.session_id) == 36
    assert s.session_id.count("-") == 4


def test_get_session_returns_none_for_unknown_id():
    from session_manager import SessionManager
    sm = SessionManager()
    assert sm.get_session("nope") is None


def test_get_or_create_session_new_keyword_generates_id():
    """The literal session_id 'new' triggers a fresh UUID, not an entry with id='new'."""
    from session_manager import SessionManager
    sm = SessionManager()
    s = sm.get_or_create_session("new")
    assert s.session_id != "new"
    assert sm.get_session("new") is None       # 'new' was not stored
    assert sm.get_session(s.session_id) is s   # but the new session was


def test_get_or_create_session_returns_existing_for_known_id():
    from session_manager import SessionManager
    sm = SessionManager()
    first = sm.create_session("sess-x")
    again = sm.get_or_create_session("sess-x")
    assert again is first


def test_get_or_create_session_creates_with_provided_id_when_missing():
    """Unknown session_id (not 'new') -> create one with that ID."""
    from session_manager import SessionManager
    sm = SessionManager()
    s = sm.get_or_create_session("brand-new-id")
    assert s.session_id == "brand-new-id"
    assert sm.get_session("brand-new-id") is s


def test_append_message_updates_history_and_last_activity():
    from session_manager import SessionManager
    sm = SessionManager()
    s = sm.create_session("sess-1")
    old_ts = s.last_activity
    sm.append_message("sess-1", "user", "hello")
    assert sm.get_history("sess-1") == [{"role": "user", "content": "hello"}]
    assert sm.get_session("sess-1").last_activity >= old_ts


def test_append_message_supports_array_content_for_images():
    """User messages may carry list-of-dict content (interleaved images)."""
    from session_manager import SessionManager
    sm = SessionManager()
    sm.create_session("s")
    array_content = [
        {"type": "image_url", "image_url": {"url": "data:.."}},
        {"type": "text", "text": "SLICE 1"},
    ]
    sm.append_message("s", "user", array_content)
    assert sm.get_history("s")[0]["content"] == array_content


def test_append_message_to_unknown_session_is_silent_noop():
    from session_manager import SessionManager
    sm = SessionManager()
    # Must NOT raise.
    sm.append_message("nonexistent", "user", "ignored")
    assert sm.get_session("nonexistent") is None


def test_get_history_returns_empty_list_for_unknown_session():
    from session_manager import SessionManager
    sm = SessionManager()
    assert sm.get_history("nope") == []


def test_remove_session_returns_true_when_existed():
    from session_manager import SessionManager
    sm = SessionManager()
    sm.create_session("s")
    assert sm.remove_session("s") is True
    assert sm.get_session("s") is None


def test_remove_session_returns_false_when_not_existed():
    from session_manager import SessionManager
    sm = SessionManager()
    assert sm.remove_session("never") is False


def test_list_sessions_returns_iso_timestamps_and_counts():
    from session_manager import SessionManager
    sm = SessionManager()
    sm.create_session("a")
    sm.append_message("a", "user", "hi")
    sm.append_message("a", "assistant", "hello")
    sm.create_session("b")
    info = sm.list_sessions()
    by_id = {entry["session_id"]: entry for entry in info}
    assert by_id["a"]["message_count"] == 2
    assert by_id["b"]["message_count"] == 0
    # ISO timestamps should be string-typed and parseable
    datetime.fromisoformat(by_id["a"]["created_at"])
    datetime.fromisoformat(by_id["a"]["last_activity"])


def test_cleanup_stale_removes_only_old_sessions():
    """Sessions whose last_activity is older than max_age_minutes get removed."""
    from session_manager import SessionManager
    sm = SessionManager()
    fresh = sm.create_session("fresh")
    stale = sm.create_session("stale")
    stale.last_activity = datetime.now() - timedelta(minutes=120)
    removed = sm.cleanup_stale(max_age_minutes=60)
    assert removed == 1
    assert sm.get_session("fresh") is fresh
    assert sm.get_session("stale") is None


def test_cleanup_stale_returns_zero_when_none_old_enough():
    from session_manager import SessionManager
    sm = SessionManager()
    sm.create_session("s1")
    sm.create_session("s2")
    assert sm.cleanup_stale(max_age_minutes=60) == 0


def test_get_session_manager_singleton():
    import session_manager as sm_mod
    sm_mod._session_manager = None
    a = sm_mod.get_session_manager()
    b = sm_mod.get_session_manager()
    assert a is b


# ---------- async surface ----------

async def test_cancel_active_generation_with_no_task_is_noop():
    """Session with active_task=None: cancel just clears the event, no error."""
    from session_manager import SessionManager
    sm = SessionManager()
    s = sm.create_session("s")
    s.cancel_event.set()
    await s.cancel_active_generation()
    assert not s.cancel_event.is_set()


async def test_cancel_active_generation_signals_event_and_lets_task_drain():
    """A cooperative task that observes cancel_event finishes within the timeout window."""
    from session_manager import SessionManager
    sm = SessionManager()
    s = sm.create_session("s")

    async def cooperative_task():
        # Drain until cancel_event is set.
        while not s.cancel_event.is_set():
            await asyncio.sleep(0.005)
        return "stopped"

    s.active_task = asyncio.create_task(cooperative_task())
    await s.cancel_active_generation()
    assert s.active_task is None
    assert not s.cancel_event.is_set()


async def test_cancel_active_generation_force_cancels_uncooperative_task():
    """A task that ignores cancel_event past the 0.5s wait gets force-cancelled."""
    from session_manager import SessionManager
    sm = SessionManager()
    s = sm.create_session("s")

    async def uncooperative_task():
        # Sleeps past the 0.5s shield window without checking cancel_event.
        # 0.6s is enough to exceed the shield while keeping suite runtime tight.
        await asyncio.sleep(0.6)
        return "never"

    s.active_task = asyncio.create_task(uncooperative_task())
    await s.cancel_active_generation()
    assert s.active_task is None
