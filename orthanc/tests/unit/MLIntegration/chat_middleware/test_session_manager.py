"""Tests for chat-middleware/session_manager.py — in-memory chat sessions with asyncio primitives."""
import asyncio
import contextlib
from datetime import datetime, timedelta, timezone

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


async def test_remove_session_returns_true_when_existed():
    from session_manager import SessionManager
    sm = SessionManager()
    sm.create_session("s")
    assert await sm.remove_session("s") is True
    assert sm.get_session("s") is None


async def test_remove_session_cancels_an_active_generation():
    """A removed session must not leave work running against it.

    Its closing append_message would find no session and drop the turn, so the
    generation would keep retrieving and preprocessing images to produce an
    answer nothing records.
    """
    from session_manager import SessionManager
    sm = SessionManager()
    session = sm.create_session("s")

    observed_cancel = asyncio.Event()

    async def generation():
        await session.cancel_event.wait()
        observed_cancel.set()

    session.active_task = asyncio.create_task(generation())
    await asyncio.sleep(0)

    assert await sm.remove_session("s") is True
    assert observed_cancel.is_set()
    assert session.active_task is None


async def test_remove_session_cancels_a_generation_that_ignores_the_event():
    """The graceful wait is bounded, so a task that never checks is force-cancelled."""
    from session_manager import SessionManager
    sm = SessionManager()
    session = sm.create_session("s")

    async def deaf_generation():
        await asyncio.sleep(3600)

    task = asyncio.create_task(deaf_generation())
    session.active_task = task
    await asyncio.sleep(0)

    assert await sm.remove_session("s") is True
    assert task.cancelled()


async def test_remove_session_marks_the_session_closed():
    """A WebSocket handler holds its Session, not its ID.

    It captured the object before its first await and never looks the ID up
    again, so popping the registry entry does not reach it. Without the flag it
    would go on serving a session the caller was told is gone.
    """
    from session_manager import SessionManager
    sm = SessionManager()
    session = sm.create_session("s")
    assert session.closed is False

    assert await sm.remove_session("s") is True
    assert session.closed is True


async def test_remove_session_survives_a_generation_that_raises_on_its_way_out():
    """The generation is stopped either way, which is all removal promises.

    Re-raising would report a delete that has already happened as a failure of
    the delete.
    """
    from session_manager import SessionManager
    sm = SessionManager()
    session = sm.create_session("s")

    async def failing_generation():
        await session.cancel_event.wait()
        raise RuntimeError("kaboom")

    session.active_task = asyncio.create_task(failing_generation())
    await asyncio.sleep(0)

    assert await sm.remove_session("s") is True
    assert sm.get_session("s") is None


async def test_cancellation_does_not_touch_a_generation_started_while_it_waited():
    """`active_task` is a mutable pointer and cancellation suspends.

    A handler still attached can replace the task mid-wait, and force-cancelling
    whatever the pointer holds at that moment would land on a generation that
    had done nothing wrong.
    """
    from session_manager import SessionManager
    sm = SessionManager()
    session = sm.create_session("s")

    async def deaf_generation():
        await asyncio.sleep(3600)

    first = asyncio.create_task(deaf_generation())
    session.active_task = first
    await asyncio.sleep(0)

    removal = asyncio.create_task(sm.remove_session("s"))
    await asyncio.sleep(0)

    # A handler that has not noticed the removal starts another generation.
    second = asyncio.create_task(deaf_generation())
    session.active_task = second

    try:
        # Bounded: a version that cancels the wrong task then awaits an
        # uncancelled one hangs, and a hang is a worse signal than a failure.
        assert await asyncio.wait_for(removal, timeout=5) is True
        assert first.cancelled()
        assert not second.done()
        # The pointer is left alone too, since it no longer names the task that
        # was cancelled.
        assert session.active_task is second
    finally:
        second.cancel()


async def test_a_generation_that_will_not_stop_is_reported_and_kept():
    """False is not "stopped", and the caller must be able to tell.

    The pointer is the only handle anything still has on such a task, and the
    cancel flag is the only thing still asking it to stop. Dropping either would
    let a second generation start beside it, on the same socket, with the loser
    of that race writing its turn into the history.
    """
    from session_manager import SessionManager
    sm = SessionManager()
    session = sm.create_session("s")

    async def unstoppable():
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            await asyncio.sleep(3600)

    task = asyncio.create_task(unstoppable())
    session.active_task = task
    await asyncio.sleep(0)

    try:
        assert await asyncio.wait_for(session.cancel_active_generation(), timeout=5) is False
        assert session.active_task is task
        assert session.cancel_event.is_set()
        assert not task.done()
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def test_cancelling_a_generation_that_has_already_finished_reports_success():
    from session_manager import SessionManager
    sm = SessionManager()
    session = sm.create_session("s")
    session.cancel_event.set()

    finished = asyncio.create_task(asyncio.sleep(0))
    session.active_task = finished
    await finished

    assert await session.cancel_active_generation() is True
    assert session.active_task is None
    assert not session.cancel_event.is_set()


async def test_remove_session_does_not_wait_forever_on_a_task_that_refuses_to_stop(caplog):
    """Both waits are bounded.

    The deletion has already taken effect by the time cancellation is attempted,
    so a task that swallows cancellation must not be able to hold the caller. At
    worst it outlives the session it was serving, unreachable.
    """
    import logging

    from session_manager import SessionManager
    sm = SessionManager()
    session = sm.create_session("s")

    async def unstoppable():
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            # Swallows the cancellation and carries on, which is the case the
            # bound exists for. Only once, so the test can still clean up.
            await asyncio.sleep(3600)

    task = asyncio.create_task(unstoppable())
    session.active_task = task
    await asyncio.sleep(0)

    try:
        with caplog.at_level(logging.WARNING, logger="session_manager"):
            assert await asyncio.wait_for(sm.remove_session("s"), timeout=5) is True
        assert "did not stop when cancelled" in caplog.text
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def test_cancelling_the_caller_does_not_look_like_the_generation_stopping():
    """Only the task's own ending is swallowed.

    Cancelling whoever asked for the removal must reach them, not be absorbed as
    "the generation stopped". The generation is stopped on the way out too: the
    session has already been dropped by then, so nothing else owns it, and
    leaving it running would be a task nobody can reach and nobody will stop.
    """
    from session_manager import SessionManager
    sm = SessionManager()
    session = sm.create_session("s")

    async def deaf_generation():
        await asyncio.sleep(3600)

    generation = asyncio.create_task(deaf_generation())
    session.active_task = generation
    await asyncio.sleep(0)

    removal = asyncio.create_task(sm.remove_session("s"))
    await asyncio.sleep(0)
    removal.cancel()

    with pytest.raises(asyncio.CancelledError):
        await removal
    assert removal.cancelled()
    # Bounded: a version that leaves the generation running hangs here, and a
    # hang is a worse signal than a failure.
    with contextlib.suppress(asyncio.CancelledError, TimeoutError):
        await asyncio.wait_for(asyncio.shield(generation), timeout=2)
    assert generation.cancelled()


async def test_cancelling_the_caller_in_the_same_turn_as_the_generation_still_reaches_it():
    """The two cancellations have to be told apart by who was asked, not by who finished.

    Asking whether the generation is done is unsound here: cancelled in the same
    turn as the caller, it is already done when the question is put, and the
    caller's own cancellation would vanish.
    """
    from session_manager import SessionManager
    sm = SessionManager()
    session = sm.create_session("s")

    async def deaf_generation():
        await asyncio.sleep(3600)

    generation = asyncio.create_task(deaf_generation())
    session.active_task = generation
    await asyncio.sleep(0)

    removal = asyncio.create_task(sm.remove_session("s"))
    await asyncio.sleep(0)
    # Same turn, generation first: it settles before the removal is asked about.
    generation.cancel()
    removal.cancel()

    with pytest.raises(asyncio.CancelledError):
        await removal
    assert removal.cancelled()


async def test_remove_session_is_unreachable_before_cancellation_finishes():
    """The pop precedes the await.

    Cancellation suspends, and anything that looked the session up while it was
    suspended would obtain a session already promised to the caller as gone.
    """
    from session_manager import SessionManager
    sm = SessionManager()
    session = sm.create_session("s")

    async def deaf_generation():
        await asyncio.sleep(3600)

    session.active_task = asyncio.create_task(deaf_generation())
    await asyncio.sleep(0)

    removal = asyncio.create_task(sm.remove_session("s"))
    await asyncio.sleep(0)  # far enough for the pop, not for the cancellation

    assert not removal.done()
    assert sm.get_session("s") is None
    assert await removal is True


async def test_a_closed_session_keeps_its_cancel_flag_set():
    """Nothing will start another generation on it.

    Clearing the flag would tell a task that swallowed its cancellation that it
    is allowed to carry on -- and it is the one kind of task that would take the
    permission.
    """
    from session_manager import SessionManager
    sm = SessionManager()
    session = sm.create_session("s")

    async def generation():
        await session.cancel_event.wait()

    session.active_task = asyncio.create_task(generation())
    await asyncio.sleep(0)

    assert await sm.remove_session("s") is True
    assert session.cancel_event.is_set()


async def test_cleanup_stale_leaves_a_session_that_is_still_generating():
    """`last_activity` does not advance while a response streams.

    A long generation therefore looks stale mid-flight, and a sweep for the
    abandoned must not cancel a response someone is watching arrive.
    """
    from session_manager import SessionManager
    sm = SessionManager()
    busy = sm.create_session("busy")
    busy.last_activity -= timedelta(minutes=120)

    async def generation():
        await asyncio.sleep(3600)

    task = asyncio.create_task(generation())
    busy.active_task = task
    await asyncio.sleep(0)

    try:
        assert await sm.cleanup_stale(max_age_minutes=60) == 0
        assert sm.get_session("busy") is busy
        assert not task.done()
    finally:
        task.cancel()


async def test_remove_session_returns_false_when_not_existed():
    from session_manager import SessionManager
    sm = SessionManager()
    assert await sm.remove_session("never") is False


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


async def test_cleanup_stale_removes_only_old_sessions():
    """Sessions whose last_activity is older than max_age_minutes get removed."""
    from session_manager import SessionManager
    sm = SessionManager()
    fresh = sm.create_session("fresh")
    stale = sm.create_session("stale")
    stale.last_activity = datetime.now(timezone.utc) - timedelta(minutes=120)
    removed = await sm.cleanup_stale(max_age_minutes=60)
    assert removed == 1
    assert sm.get_session("fresh") is fresh
    assert sm.get_session("stale") is None


def test_session_timestamps_are_tz_aware():
    from session_manager import SessionManager
    sm = SessionManager()
    s = sm.create_session("s")
    assert s.created_at.tzinfo is not None
    assert s.last_activity.tzinfo is not None


def test_append_message_sets_tz_aware_last_activity():
    from session_manager import SessionManager
    sm = SessionManager()
    sm.create_session("s")
    sm.append_message("s", "user", "hi")
    assert sm.get_session("s").last_activity.tzinfo is not None


async def test_cleanup_stale_works_with_aware_last_activity():
    """Expiry comparison must not raise on aware datetimes (naive vs aware -> TypeError)."""
    from session_manager import SessionManager
    sm = SessionManager()
    stale = sm.create_session("stale")
    stale.last_activity = datetime.now(timezone.utc) - timedelta(minutes=120)
    assert await sm.cleanup_stale(max_age_minutes=60) == 1


async def test_cleanup_stale_returns_zero_when_none_old_enough():
    from session_manager import SessionManager
    sm = SessionManager()
    sm.create_session("s1")
    sm.create_session("s2")
    assert await sm.cleanup_stale(max_age_minutes=60) == 0


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
