"""Tests for chat-middleware/websocket_handler.py — handle_chat + handle_websocket dispatcher."""
import asyncio
import asyncio as _asyncio
import contextlib
from datetime import datetime

import pytest


def _reset_singletons(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGE_FOLDER", str(tmp_path / "ws-img"))
    monkeypatch.setenv("MAX_CACHE_ENTRIES", "10")
    import config; config.config = None
    import runtime_config; runtime_config._runtime_config = None
    import session_manager; session_manager._session_manager = None
    import image_cache; image_cache._image_cache = None
    import ollama_client
    ollama_client._ollama_client = None
    import sys
    sys.modules.pop("websocket_handler", None)
    sys.modules.pop("prompt_builder", None)


# ---------------------------------------------------------------------------
# Fake WebSocket — captures send_json messages, drives iter_json from a queue.
# Async-compatible (each method is a coroutine).
# ---------------------------------------------------------------------------

class _FakeWebSocket:
    def __init__(self, incoming=None):
        self.sent = []
        self.accepted = False
        self._incoming = list(incoming or [])

    async def accept(self):
        self.accepted = True

    async def send_json(self, payload):
        self.sent.append(payload)

    async def iter_json(self):
        for msg in self._incoming:
            yield msg


class _AwaitableOllama:
    """Async ollama fake that yields chunks with optional asyncio.sleep gaps.

    Allows tests to interleave a second CHAT request while the first is still streaming."""
    def __init__(self, chunks, per_chunk_sleep_s=0.05):
        self._chunks = chunks
        self._sleep = per_chunk_sleep_s
        self.model = "test"
        self.calls = []

    async def chat_stream(self, messages, cancel_event=None, runtime_options=None):
        self.calls.append({"messages": messages})
        for chunk in self._chunks:
            if cancel_event is not None and cancel_event.is_set():
                return
            await _asyncio.sleep(self._sleep)
            yield chunk

    async def health_check(self):
        return True

    async def list_models(self):
        return ["test"]


class _DeafOllama(_AwaitableOllama):
    """Never looks at cancel_event, so a wait for it to stop actually waits."""

    async def chat_stream(self, messages, cancel_event=None, runtime_options=None):
        self.calls.append({"messages": messages})
        for chunk in self._chunks:
            await _asyncio.sleep(self._sleep)
            yield chunk


class _CountingOllama(_AwaitableOllama):
    """Records the high-water mark of generations running at the same time."""

    def __init__(self, chunks, per_chunk_sleep_s=0.05):
        super().__init__(chunks, per_chunk_sleep_s)
        self.in_flight = 0
        self.max_in_flight = 0

    async def chat_stream(self, messages, cancel_event=None, runtime_options=None):
        self.calls.append({"messages": messages})
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            for chunk in self._chunks:
                await _asyncio.sleep(self._sleep)
                yield chunk
        finally:
            self.in_flight -= 1


class _UnstoppableOllama(_AwaitableOllama):
    """Swallows its cancellation once, so stopping it cannot succeed in time."""

    async def chat_stream(self, messages, cancel_event=None, runtime_options=None):
        self.calls.append({"messages": messages})
        try:
            await _asyncio.sleep(3600)
        except _asyncio.CancelledError:
            await _asyncio.sleep(3600)
        yield {"type": "content", "text": "never reached"}


# ---------- send_message ----------

async def test_send_message_wraps_payload_with_type_field(tmp_path, monkeypatch):
    _reset_singletons(tmp_path, monkeypatch)
    import websocket_handler as wh
    from models import ServerMessageType
    ws = _FakeWebSocket()
    await wh.send_message(ws, ServerMessageType.TOKEN, content="hi")
    assert ws.sent == [{"type": "token", "content": "hi"}]


# ---------- handle_chat happy path ----------

class _FakeOllamaForChat:
    def __init__(self, chunks):
        self._chunks = chunks
        self.model = "default"
        self.calls = []

    async def chat_stream(self, messages, cancel_event=None, runtime_options=None):
        self.calls.append({"messages": messages, "runtime_options": runtime_options})
        for ch in self._chunks:
            yield ch


async def test_handle_chat_streams_tokens_and_signals_done(tmp_path, monkeypatch):
    _reset_singletons(tmp_path, monkeypatch)
    import websocket_handler as wh

    fake = _FakeOllamaForChat([
        {"type": "content", "text": "Hello "},
        {"type": "content", "text": "world"},
    ])
    monkeypatch.setattr(wh, "get_client_for_provider", lambda *a, **kw:fake)

    from session_manager import get_session_manager
    sm = get_session_manager()
    s = sm.create_session("S1")

    ws = _FakeWebSocket()
    await wh.handle_chat(ws, s, content="describe", study_uid="STD", series_uids=[])
    # Token messages + a final DONE
    msg_types = [m["type"] for m in ws.sent]
    assert "token" in msg_types
    assert msg_types[-1] == "done"
    # History persisted
    history = sm.get_history("S1")
    assert history[-1]["content"] == "Hello world"


async def test_handle_chat_with_series_emits_preprocessing_messages(tmp_path, monkeypatch):
    _reset_singletons(tmp_path, monkeypatch)
    import websocket_handler as wh
    fake = _FakeOllamaForChat([{"type": "content", "text": "ok"}])
    monkeypatch.setattr(wh, "get_client_for_provider", lambda *a, **kw:fake)

    async def _fake_preprocess(series_uid, study_uid, params, wado_url, image_folder,
                               selection=None):
        return ["data:image/png;base64,IMG1"]
    monkeypatch.setattr(wh, "preprocess_series", _fake_preprocess)

    from session_manager import get_session_manager
    sm = get_session_manager()
    s = sm.create_session("S2")
    ws = _FakeWebSocket()
    await wh.handle_chat(ws, s, content="q", study_uid="STD", series_uids=["SE1", "SE2"])

    # Expect at least 2 preprocessing messages (one per uncached series + completion)
    pp = [m for m in ws.sent if m["type"] == "preprocessing"]
    assert len(pp) >= 2
    assert pp[-1]["progress"] == 1.0


async def test_handle_chat_uses_cache_when_series_present(tmp_path, monkeypatch):
    """If image_cache already has the series, preprocess_series is NOT called again."""
    _reset_singletons(tmp_path, monkeypatch)
    import websocket_handler as wh
    fake = _FakeOllamaForChat([{"type": "content", "text": "x"}])
    monkeypatch.setattr(wh, "get_client_for_provider", lambda *a, **kw:fake)

    pp_calls = []
    async def _fake_preprocess(*a, **kw):
        pp_calls.append(a)
        return ["img"]
    monkeypatch.setattr(wh, "preprocess_series", _fake_preprocess)

    # Seeded under the real key: series PLUS the recipe in force. Seeding the bare
    # series UID would miss, which is the whole point of the recipe-aware key.
    from image_cache import get_image_cache, make_cache_key, CachedSeries
    from preprocessing import recipe_signature
    from runtime_config import get_runtime_config
    cache = get_image_cache()
    key = make_cache_key("SE-cached", recipe_signature(get_runtime_config().preprocessing))
    cache.put(key, CachedSeries(series_uid="SE-cached", base64_images=["cached-img"]))

    from session_manager import get_session_manager
    s = get_session_manager().create_session("S3")
    ws = _FakeWebSocket()
    await wh.handle_chat(ws, s, content="q", study_uid="STD", series_uids=["SE-cached"])
    assert pp_calls == []                      # cache hit, no preprocess


async def test_handle_chat_emits_error_when_preprocess_fails(tmp_path, monkeypatch):
    _reset_singletons(tmp_path, monkeypatch)
    import websocket_handler as wh
    fake = _FakeOllamaForChat([])
    monkeypatch.setattr(wh, "get_client_for_provider", lambda *a, **kw:fake)

    async def _fake_preprocess(*a, **kw):
        raise RuntimeError("WADO unreachable")
    monkeypatch.setattr(wh, "preprocess_series", _fake_preprocess)

    from session_manager import get_session_manager
    s = get_session_manager().create_session("S4")
    ws = _FakeWebSocket()
    await wh.handle_chat(ws, s, content="q", study_uid="STD", series_uids=["SE-bad"])
    error_msgs = [m for m in ws.sent if m["type"] == "error"]
    assert len(error_msgs) == 1
    assert "WADO unreachable" in error_msgs[0]["content"]


async def test_handle_chat_emits_thinking_token_for_reasoning_chunks(tmp_path, monkeypatch):
    _reset_singletons(tmp_path, monkeypatch)
    import websocket_handler as wh
    fake = _FakeOllamaForChat([
        {"type": "thinking", "text": "let me think"},
        {"type": "content", "text": "answer"},
    ])
    monkeypatch.setattr(wh, "get_client_for_provider", lambda *a, **kw:fake)
    from session_manager import get_session_manager
    s = get_session_manager().create_session("S5")
    ws = _FakeWebSocket()
    await wh.handle_chat(ws, s, content="x", study_uid="", series_uids=[])
    types = [m["type"] for m in ws.sent]
    assert "thinking_token" in types
    assert "token" in types


async def test_handle_chat_skips_when_cancel_event_already_set(tmp_path, monkeypatch):
    """If cancel_event is set before handle_chat starts, function returns immediately."""
    _reset_singletons(tmp_path, monkeypatch)
    import websocket_handler as wh
    fake = _FakeOllamaForChat([{"type": "content", "text": "should not see"}])
    monkeypatch.setattr(wh, "get_client_for_provider", lambda *a, **kw:fake)
    from session_manager import get_session_manager
    s = get_session_manager().create_session("S6")
    s.cancel_event.set()
    ws = _FakeWebSocket()
    await wh.handle_chat(ws, s, content="x", study_uid="", series_uids=[])
    assert ws.sent == []                       # nothing sent, function returned early


async def test_handle_chat_emits_error_when_ollama_stream_raises(tmp_path, monkeypatch):
    _reset_singletons(tmp_path, monkeypatch)
    import websocket_handler as wh

    class _Boom:
        model = "x"
        async def chat_stream(self, *a, **kw):
            raise RuntimeError("ollama down")
            yield   # unreachable but makes it an async generator
    monkeypatch.setattr(wh, "get_client_for_provider", lambda *a, **kw:_Boom())

    from session_manager import get_session_manager
    s = get_session_manager().create_session("S7")
    ws = _FakeWebSocket()
    await wh.handle_chat(ws, s, content="x", study_uid="", series_uids=[])
    errs = [m for m in ws.sent if m["type"] == "error"]
    assert len(errs) == 1
    assert "ollama down" in errs[0]["content"]


async def test_handle_chat_does_not_persist_history_when_cancelled_mid_stream(tmp_path, monkeypatch):
    """If cancel_event is set during the stream, no history is appended AND a DONE is still sent.

    H4 fix: also assert no `assistant`-role entry landed in history — a refactor that swaps
    the cancel-check order would slip past a pure history-emptiness assertion."""
    _reset_singletons(tmp_path, monkeypatch)
    import websocket_handler as wh
    from session_manager import get_session_manager
    s = get_session_manager().create_session("S8")

    class _CancellingClient:
        model = "x"
        async def chat_stream(self, messages, cancel_event=None, runtime_options=None):
            yield {"type": "content", "text": "partial"}
            cancel_event.set()
            yield {"type": "content", "text": "should be ignored after cancel"}
    monkeypatch.setattr(wh, "get_client_for_provider", lambda *a, **kw:_CancellingClient())

    ws = _FakeWebSocket()
    await wh.handle_chat(ws, s, content="x", study_uid="", series_uids=[])
    # History must NOT contain an assistant entry (the partial response).
    history = get_session_manager().get_history("S8")
    assert all(m.get("role") != "assistant" for m in history), \
        f"assistant entry leaked into history after cancel: {history}"
    # Specifically no entry at all (cancel happens before user message append too).
    assert history == []


# ---------- make_task_done_callback ----------

async def test_task_done_callback_releases_the_session_hold(tmp_path, monkeypatch):
    _reset_singletons(tmp_path, monkeypatch)
    import websocket_handler as wh
    from session_manager import Session

    session = Session(session_id="s")
    task = asyncio.create_task(asyncio.sleep(0))
    session.active_task = task
    await task
    wh.make_task_done_callback(session)(task)

    assert session.active_task is None


async def test_task_done_callback_does_not_raise_on_a_cancelled_task(
    tmp_path, monkeypatch, caplog
):
    """`Task.exception()` re-raises on a cancelled task.

    Cancellation has to be answered before asking, or the cleanup callback
    itself raises inside the event loop, where nothing is waiting to handle it.
    Reachable through the CANCEL message, and through deleting a session that
    is still generating.
    """
    import logging

    _reset_singletons(tmp_path, monkeypatch)
    import websocket_handler as wh
    from session_manager import Session

    session = Session(session_id="s")
    task = asyncio.create_task(asyncio.sleep(3600))
    session.active_task = task

    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Called directly rather than through add_done_callback: asyncio hands a
    # callback's exception to the loop's exception handler instead of raising it
    # into the test, so a scheduled call would pass whether or not the guard is
    # there.
    with caplog.at_level(logging.INFO, logger="websocket_handler"):
        wh.make_task_done_callback(session)(task)

    assert session.active_task is None
    assert "Chat task failed" not in caplog.text


async def test_task_done_callback_reports_a_failure(tmp_path, monkeypatch, caplog):
    import logging

    _reset_singletons(tmp_path, monkeypatch)
    import websocket_handler as wh
    from session_manager import Session

    async def boom():
        raise RuntimeError("kaboom")

    session = Session(session_id="s")
    task = asyncio.create_task(boom())
    session.active_task = task

    with pytest.raises(RuntimeError):
        await task
    with caplog.at_level(logging.ERROR, logger="websocket_handler"):
        wh.make_task_done_callback(session)(task)

    assert "Chat task failed: kaboom" in caplog.text


# ---------- handle_websocket dispatcher ----------

async def test_handle_websocket_sends_connected_with_session_id(tmp_path, monkeypatch):
    _reset_singletons(tmp_path, monkeypatch)
    import websocket_handler as wh
    ws = _FakeWebSocket(incoming=[])           # no client messages
    await wh.handle_websocket(ws, "explicit-sess")
    assert ws.accepted
    first = ws.sent[0]
    assert first["type"] == "connected"
    assert first["session_id"] == "explicit-sess"


async def test_handle_websocket_invalid_message_emits_error(tmp_path, monkeypatch):
    """A message that fails ClientMessage(**message) validation is wrapped as an error event."""
    _reset_singletons(tmp_path, monkeypatch)
    import websocket_handler as wh
    ws = _FakeWebSocket(incoming=[{"type": "not-a-real-type"}])
    await wh.handle_websocket(ws, "S")
    err = [m for m in ws.sent if m["type"] == "error"]
    assert len(err) == 1


async def test_handle_websocket_non_dict_payload_emits_error_and_continues(tmp_path, monkeypatch):
    """A non-object JSON payload (list) must not crash on ClientMessage(**message);
    a structured error frame is sent and the loop continues to the next (valid) message."""
    _reset_singletons(tmp_path, monkeypatch)
    import websocket_handler as wh
    ws = _FakeWebSocket(incoming=[["not", "a", "dict"], {"type": "cancel"}])
    await wh.handle_websocket(ws, "S")
    err = [m for m in ws.sent if m["type"] == "error"]
    assert len(err) == 1                       # one structured error for the bad payload
    # Loop survived: the subsequent cancel (no active task) produced no further error/done.
    assert [m["type"] for m in ws.sent].count("error") == 1


async def test_handle_websocket_stops_serving_a_deleted_session(tmp_path, monkeypatch):
    """A handler holds its Session from before its first await.

    Popping the registry entry never reaches it, so a session deleted while a
    connection was open would go on being served -- accumulating history, and
    with it references to the base64 slices of every turn.
    """
    _reset_singletons(tmp_path, monkeypatch)
    import websocket_handler as wh
    from session_manager import get_session_manager

    # A malformed frame, because the dispatcher answers one synchronously: a
    # handler that is still serving replies with an error, so silence here is
    # evidence the loop stopped rather than evidence the fake never got that far.
    ws = _FakeWebSocket([{"type": "nonsense"}])
    session = get_session_manager().create_session("S-deleted")
    # Closed as `remove_session` closes it, with the handler already holding the
    # object -- which is the case the flag exists for. Popping alone would leave
    # this handler serving a session the caller was told is gone.
    session.closed = True

    await wh.handle_websocket(ws, "S-deleted")

    assert [m["type"] for m in ws.sent] == ["connected"]


async def test_a_disconnecting_socket_does_not_cancel_another_ones_generation(
    tmp_path, monkeypatch
):
    """`active_task` is not a record of what a handler is responsible for.

    The same session object is handed to anyone who asks for the ID, so a
    handler that cancels whatever is running when it goes away can end an answer
    a different reader is watching arrive -- a conversation cut off by a
    disconnection somewhere else entirely.
    """
    _reset_singletons(tmp_path, monkeypatch)
    import websocket_handler as wh
    from session_manager import get_session_manager

    fake = _CountingOllama(chunks=[{"type": "content", "text": "x"}] * 40, per_chunk_sleep_s=0.02)
    monkeypatch.setattr(wh, "get_client_for_provider", lambda *a, **kw: fake)

    hold_open = asyncio.Event()

    class _GeneratingWebSocket(_FakeWebSocket):
        async def iter_json(self):
            yield {"type": "chat", "content": "hi", "study_uid": "", "series_uids": []}
            await hold_open.wait()

    class _IdleWebSocket(_FakeWebSocket):
        """Joins the same session, asks nothing, and leaves."""

        async def iter_json(self):
            return
            yield  # pragma: no cover - makes this an async generator

    generating = asyncio.create_task(
        wh.handle_websocket(_GeneratingWebSocket(), "S-shared-disconnect")
    )

    session = None
    for _ in range(400):
        await asyncio.sleep(0.005)
        session = get_session_manager().get_session("S-shared-disconnect")
        if session and session.active_task and fake.in_flight == 1:
            break
    assert session is not None and fake.in_flight == 1
    generation = session.active_task

    # The other socket joins and leaves without ever asking for anything.
    await wh.handle_websocket(_IdleWebSocket(), "S-shared-disconnect")

    try:
        assert not generation.done()
        assert not session.cancel_event.is_set()
        assert session.active_task is generation
    finally:
        hold_open.set()
        with contextlib.suppress(asyncio.CancelledError, TimeoutError):
            await asyncio.wait_for(generating, timeout=5)


async def test_two_sockets_on_one_session_do_not_generate_side_by_side(tmp_path, monkeypatch):
    """A session is not private to one socket.

    `get_or_create_session` hands the same object to anyone who asks for the ID,
    and replacing a generation is four steps with awaits between them. Two
    handlers waiting on the same previous generation both wake when it stops, and
    unserialized both go on to create one: two answers streaming into one
    conversation, with only the later referenced, so the other cannot even be
    cancelled afterwards.
    """
    _reset_singletons(tmp_path, monkeypatch)
    import websocket_handler as wh
    from session_manager import get_session_manager

    fake = _CountingOllama(chunks=[{"type": "content", "text": "x"}] * 40, per_chunk_sleep_s=0.02)
    monkeypatch.setattr(wh, "get_client_for_provider", lambda *a, **kw: fake)

    chat = {"type": "chat", "content": "hi", "study_uid": "", "series_uids": []}
    race = asyncio.Event()
    hold_open = asyncio.Event()

    class _RacingWebSocket(_FakeWebSocket):
        def __init__(self, first=False):
            super().__init__()
            self._first = first

        async def iter_json(self):
            if self._first:
                yield chat
            await race.wait()
            yield chat
            await hold_open.wait()

    first_ws, second_ws = _RacingWebSocket(first=True), _RacingWebSocket()
    handlers = [
        asyncio.create_task(wh.handle_websocket(first_ws, "S-shared")),
        asyncio.create_task(wh.handle_websocket(second_ws, "S-shared")),
    ]

    # One generation running, which both sockets will then try to replace.
    # Real sleeps, not `sleep(0)`: the generation's own chunk gaps are wall-clock,
    # so yielding without advancing time never lets it make progress.
    session = None
    for _ in range(400):
        await asyncio.sleep(0.005)
        session = get_session_manager().get_session("S-shared")
        if session and session.active_task and fake.in_flight == 1:
            break
    assert session is not None and fake.in_flight == 1

    race.set()
    for _ in range(400):
        await asyncio.sleep(0.005)
        if len(fake.calls) >= 3:
            break

    try:
        assert len(fake.calls) >= 2, "the second and third CHATs never raced"
        # Whatever the interleaving, one generation at a time.
        assert fake.max_in_flight == 1
    finally:
        hold_open.set()
        for handler in handlers:
            with contextlib.suppress(asyncio.CancelledError, TimeoutError):
                await asyncio.wait_for(handler, timeout=5)


async def test_handle_websocket_refuses_a_second_generation_while_the_first_will_not_stop(
    tmp_path, monkeypatch
):
    """Two generations on one socket would both stream to the same reader.

    And the loser of that race would still write its turn into the history. A
    cancellation that did not take is not a cancellation, and the dispatcher has
    to be told rather than assume.
    """
    _reset_singletons(tmp_path, monkeypatch)
    import websocket_handler as wh
    from session_manager import get_session_manager

    fake = _UnstoppableOllama(chunks=[], per_chunk_sleep_s=0)
    monkeypatch.setattr(wh, "get_client_for_provider", lambda *a, **kw: fake)

    second_chat = asyncio.Event()

    class _GatedWebSocket(_FakeWebSocket):
        async def iter_json(self):
            yield {"type": "chat", "content": "first", "study_uid": "", "series_uids": []}
            await second_chat.wait()
            yield {"type": "chat", "content": "second", "study_uid": "", "series_uids": []}

    ws = _GatedWebSocket()
    handler = asyncio.create_task(wh.handle_websocket(ws, "S-stuck"))

    session = None
    for _ in range(500):
        await asyncio.sleep(0)
        session = get_session_manager().get_session("S-stuck")
        if session and session.active_task and len(fake.calls) == 1:
            break
    assert session is not None
    first = session.active_task
    assert first is not None

    second_chat.set()
    await asyncio.wait_for(handler, timeout=10)

    try:
        # The second CHAT never became a generation, and the reader was told why
        # rather than left with two answers arriving at once.
        assert len(fake.calls) == 1
        assert "error" in [m["type"] for m in ws.sent]
        # The stuck generation keeps its pointer: it is the only handle left on it.
        assert session.active_task is first
        assert session.cancel_event.is_set()
    finally:
        first.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await first


async def test_handle_websocket_does_not_claim_a_cancellation_that_did_not_take(
    tmp_path, monkeypatch, caplog
):
    """Saying "Cancelled" over a generation still streaming is worse than not cancelling it.

    And no error frame either: the panel marks the turn cancelled the moment it
    asks, so a late error on top of that says three contradictory things about
    one message. The record belongs in the log.
    """
    import logging

    _reset_singletons(tmp_path, monkeypatch)
    import websocket_handler as wh
    from session_manager import get_session_manager

    fake = _UnstoppableOllama(chunks=[], per_chunk_sleep_s=0)
    monkeypatch.setattr(wh, "get_client_for_provider", lambda *a, **kw: fake)

    cancel_now = asyncio.Event()

    class _GatedWebSocket(_FakeWebSocket):
        async def iter_json(self):
            yield {"type": "chat", "content": "first", "study_uid": "", "series_uids": []}
            await cancel_now.wait()
            yield {"type": "cancel"}

    ws = _GatedWebSocket()
    handler = asyncio.create_task(wh.handle_websocket(ws, "S-stuck-cancel"))

    session = None
    for _ in range(500):
        await asyncio.sleep(0)
        session = get_session_manager().get_session("S-stuck-cancel")
        if session and session.active_task and len(fake.calls) == 1:
            break
    assert session is not None
    first = session.active_task
    assert first is not None

    cancel_now.set()
    with caplog.at_level(logging.WARNING, logger="websocket_handler"):
        await asyncio.wait_for(handler, timeout=10)

    try:
        sent = [(m["type"], m.get("content")) for m in ws.sent]
        assert ("done", "Cancelled") not in sent
        assert "error" not in [t for t, _ in sent]
        assert "did not take" in caplog.text
    finally:
        first.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await first


async def test_handle_websocket_does_not_start_a_generation_on_a_session_deleted_mid_wait(
    tmp_path, monkeypatch
):
    """The `closed` check has to be asked again after every await before a task starts.

    A second CHAT makes the dispatcher stop the first generation before starting
    another. A delete arriving during that leaves the handler about to start a
    generation on a session already reported gone -- and since reconnecting under
    the same ID makes a *fresh* session, that generation could write its turn
    into someone else's conversation.

    Driven by events rather than by sleeping. `cancel_event` is set by the
    dispatcher's second-CHAT branch and by nothing else before the delete, so
    waiting on it puts the delete inside that branch every time, however loaded
    the machine.
    """
    _reset_singletons(tmp_path, monkeypatch)
    import websocket_handler as wh
    from session_manager import get_session_manager

    # Deaf to the cancel event, so stopping it ends in a forced cancellation --
    # the branch that clears `active_task` and could find it already cleared.
    fake = _DeafOllama(chunks=[{"type": "content", "text": "x"}] * 20, per_chunk_sleep_s=1.5)
    monkeypatch.setattr(wh, "get_client_for_provider", lambda *a, **kw: fake)

    second_chat = asyncio.Event()
    hold_open = asyncio.Event()

    class _GatedWebSocket(_FakeWebSocket):
        async def iter_json(self):
            yield {"type": "chat", "content": "first", "study_uid": "", "series_uids": []}
            await second_chat.wait()
            yield {"type": "chat", "content": "second", "study_uid": "", "series_uids": []}
            # The connection stays open afterwards. The dispatcher does not await
            # the generation it starts, and ending the iteration would cancel it
            # in the `finally` before it ran -- so a handler that wrongly started
            # one would look identical to one that correctly did not.
            await hold_open.wait()

    ws = _GatedWebSocket()
    handler = asyncio.create_task(wh.handle_websocket(ws, "S-mid-wait"))

    # Let the first generation get going, so the second CHAT has something to
    # stop rather than sailing past.
    session = None
    for _ in range(500):
        await asyncio.sleep(0)
        session = get_session_manager().get_session("S-mid-wait")
        if session and session.active_task and len(fake.calls) == 1:
            break
    assert session is not None
    first = session.active_task
    assert first is not None

    second_chat.set()
    # Set as the dispatcher enters the stop, and by nothing else until then.
    await asyncio.wait_for(session.cancel_event.wait(), timeout=5)
    assert session.active_task is first

    assert await get_session_manager().remove_session("S-mid-wait") is True

    # A wrongly started second generation gets its chance to run before the
    # connection ends.
    for _ in range(500):
        await asyncio.sleep(0)
        if len(fake.calls) > 1:
            break
    hold_open.set()
    await asyncio.wait_for(handler, timeout=5)

    # The second CHAT never became a generation.
    assert len(fake.calls) == 1
    # And the deletion was not reported to the client as a failure: re-reading
    # `session.active_task` after the wait would find the None the removal left
    # there.
    assert "error" not in [m["type"] for m in ws.sent]


async def test_handle_chat_does_not_append_to_a_session_that_was_deleted(tmp_path, monkeypatch):
    """A generation that outlives its deletion must not write its turn anywhere.

    Checked on the session object, not by ID: reconnecting under a deleted ID
    produces a different session under the same name, and appending by ID would
    put this turn -- images and all -- into a stranger's conversation.
    """
    _reset_singletons(tmp_path, monkeypatch)
    import websocket_handler as wh
    from session_manager import get_session_manager

    fake = _AwaitableOllama(chunks=[{"type": "content", "text": "answer"}], per_chunk_sleep_s=0.0)
    monkeypatch.setattr(wh, "get_client_for_provider", lambda *a, **kw: fake)

    manager = get_session_manager()
    doomed = manager.create_session("S-gone")
    doomed.closed = True
    # A fresh session under the same ID, as a reconnect would produce.
    manager.sessions["S-gone"] = manager.create_session("S-gone")

    await wh.handle_chat(_FakeWebSocket(), doomed, "hi", "", [], [])

    assert doomed.conversation_history == []
    assert manager.get_session("S-gone").conversation_history == []


async def test_handle_websocket_cancel_without_active_task_is_silent(tmp_path, monkeypatch):
    """CANCEL with no active task: no error, no DONE."""
    _reset_singletons(tmp_path, monkeypatch)
    import websocket_handler as wh
    ws = _FakeWebSocket(incoming=[{"type": "cancel"}])
    await wh.handle_websocket(ws, "S")
    types = [m["type"] for m in ws.sent]
    assert "error" not in types
    assert types.count("done") == 0


# ---------------------------------------------------------------------------
# H1+H2: real WS TestClient (covers active-task cancel + task_done_callback
# + back-to-back CHAT concurrent-generation cancel) + WebSocketDisconnect path
# ---------------------------------------------------------------------------


def _build_app_with_ws(tmp_path, monkeypatch, fake_ollama):
    """Build a minimal FastAPI app exposing ONLY the /ws/chat/{sid} route + mount the real handler.

    Reset all chat-middleware singletons + inject the fake ollama before app import so the
    handler picks it up. Returns (app, TestClient).
    """
    monkeypatch.setenv("IMAGE_FOLDER", str(tmp_path / "ws-real-img"))
    monkeypatch.setenv("MAX_CACHE_ENTRIES", "5")
    import config; config.config = None
    import runtime_config; runtime_config._runtime_config = None
    import session_manager; session_manager._session_manager = None
    import image_cache; image_cache._image_cache = None
    import ollama_client
    ollama_client._ollama_client = None
    monkeypatch.setattr(ollama_client, "get_ollama_client", lambda *a, **kw: fake_ollama)

    import sys
    sys.modules.pop("websocket_handler", None)
    import websocket_handler as wh
    # Also reach the dispatcher's module-level alias. handle_chat resolves its client
    # through get_client_for_provider(provider, model), so that is the seam to patch.
    monkeypatch.setattr(wh, "get_client_for_provider", lambda *a, **kw: fake_ollama)

    from fastapi import FastAPI, WebSocket
    from fastapi.testclient import TestClient

    app = FastAPI()

    @app.websocket("/ws/chat/{session_id}")
    async def _ws(websocket: WebSocket, session_id: str):
        await wh.handle_websocket(websocket, session_id)

    return app, TestClient(app)


def test_real_ws_dispatcher_chat_message_streams_tokens_to_completion(tmp_path, monkeypatch):
    """Drive handle_websocket through a real TestClient.websocket_connect.
    Exercises accept() lifecycle + websocket.iter_json + asyncio.Task scheduling + task_done_callback."""
    fake = _AwaitableOllama(chunks=[
        {"type": "content", "text": "Hello "},
        {"type": "content", "text": "world"},
    ], per_chunk_sleep_s=0.0)
    _app, client = _build_app_with_ws(tmp_path, monkeypatch, fake)
    with client.websocket_connect("/ws/chat/sess-real-1") as ws:
        connected = ws.receive_json()
        assert connected["type"] == "connected"
        assert connected["session_id"] == "sess-real-1"

        ws.send_json({"type": "chat", "content": "hello", "study_uid": "", "series_uids": []})

        # Drain tokens + done. Expect 2 tokens + 1 done.
        received = []
        while True:
            msg = ws.receive_json()
            received.append(msg)
            if msg["type"] == "done":
                break
        token_msgs = [m for m in received if m["type"] == "token"]
        assert [m["content"] for m in token_msgs] == ["Hello ", "world"]


def test_real_ws_malformed_message_emits_error_then_keeps_connection_open(tmp_path, monkeypatch):
    """A malformed message yields a structured error frame; a subsequent valid CHAT still streams.

    Drives the real dispatcher so the background chat task actually runs."""
    fake = _AwaitableOllama(chunks=[{"type": "content", "text": "ok"}], per_chunk_sleep_s=0.0)
    _app, client = _build_app_with_ws(tmp_path, monkeypatch, fake)
    with client.websocket_connect("/ws/chat/sess-malformed") as ws:
        assert ws.receive_json()["type"] == "connected"

        # Malformed: unknown message type -> validation error frame.
        ws.send_json({"type": "not-a-real-type"})
        err = ws.receive_json()
        assert err["type"] == "error"
        assert err.get("content")              # structured, carries a message

        # Connection still open: a valid CHAT is processed normally.
        ws.send_json({"type": "chat", "content": "hi", "study_uid": "", "series_uids": []})
        received = []
        while True:
            msg = ws.receive_json()
            received.append(msg)
            if msg["type"] == "done":
                break
        assert [m["content"] for m in received if m["type"] == "token"] == ["ok"]


def test_real_ws_non_dict_payload_emits_error_and_keeps_connection_open(tmp_path, monkeypatch):
    """A non-object JSON payload (list) must not crash ClientMessage(**message)."""
    fake = _AwaitableOllama(chunks=[{"type": "content", "text": "ok"}], per_chunk_sleep_s=0.0)
    _app, client = _build_app_with_ws(tmp_path, monkeypatch, fake)
    with client.websocket_connect("/ws/chat/sess-nondict") as ws:
        assert ws.receive_json()["type"] == "connected"

        ws.send_json(["not", "a", "dict"])     # **message would raise TypeError
        err = ws.receive_json()
        assert err["type"] == "error"

        ws.send_json({"type": "chat", "content": "hi", "study_uid": "", "series_uids": []})
        received = []
        while True:
            msg = ws.receive_json()
            received.append(msg)
            if msg["type"] == "done":
                break
        assert [m["content"] for m in received if m["type"] == "token"] == ["ok"]


def test_real_ws_dispatcher_cancel_during_active_generation(tmp_path, monkeypatch):
    """CHAT then CANCEL: dispatcher must set cancel_event, await the active task, and emit DONE.

    Exercises the active_task / cancel_event / wait_for(..., 1.0) code path in handle_websocket
    that is otherwise unreachable through the fake-WS-only tests."""
    # Slow chunks so cancellation arrives mid-stream.
    fake = _AwaitableOllama(chunks=[{"type": "content", "text": "slow-chunk"}] * 30,
                              per_chunk_sleep_s=0.05)
    _app, client = _build_app_with_ws(tmp_path, monkeypatch, fake)

    with client.websocket_connect("/ws/chat/sess-cancel-test") as ws:
        ws.receive_json()         # connected
        ws.send_json({"type": "chat", "content": "long-generation",
                       "study_uid": "", "series_uids": []})
        # Drain the first token so we know the chat task is in-flight.
        first = ws.receive_json()
        assert first["type"] == "token"

        # Now CANCEL while the task is still streaming.
        ws.send_json({"type": "cancel"})

        # Dispatcher cancels active task -> chat task finishes (with DONE).
        # Then dispatcher sends its own DONE (content="Cancelled").
        # Drain until we see the cancellation DONE.
        cancel_done_seen = False
        while True:
            msg = ws.receive_json()
            if msg["type"] == "done" and msg.get("content") == "Cancelled":
                cancel_done_seen = True
                break
            # The chat task'''s own DONE (no content) — keep draining.
        assert cancel_done_seen


def test_handle_websocket_handles_websocketdisconnect_mid_iter(tmp_path, monkeypatch):
    """Fake WebSocket whose iter_json raises WebSocketDisconnect: the finally branch
    must cancel any active task and not propagate the exception."""
    _reset_singletons(tmp_path, monkeypatch)
    import websocket_handler as wh
    from fastapi import WebSocketDisconnect

    class _DisconnectingWS:
        def __init__(self):
            self.sent = []
            self.accepted = False
        async def accept(self):
            self.accepted = True
        async def send_json(self, payload):
            self.sent.append(payload)
        async def iter_json(self):
            # Yield nothing; just raise.
            raise WebSocketDisconnect()
            yield {}   # unreachable; needed to make iter_json an async generator

    ws = _DisconnectingWS()
    # Must NOT raise.
    import asyncio
    asyncio.run(wh.handle_websocket(ws, "S-disco"))
    assert ws.accepted
    # The connected event was sent before disconnect.
    assert ws.sent and ws.sent[0]["type"] == "connected"


# ---------- per-message slice selection ----------
#
# The panel shows a per-message provenance snapshot naming the slices it sent.
# These tests pin the two ways that claim could become a lie: the selection not
# reaching preprocessing, and the cache serving another message's pixels.

async def test_handle_chat_forwards_the_slice_selection_for_that_series(tmp_path, monkeypatch):
    _reset_singletons(tmp_path, monkeypatch)
    import websocket_handler as wh
    from models import SliceSelection
    fake = _FakeOllamaForChat([{"type": "content", "text": "ok"}])
    monkeypatch.setattr(wh, "get_client_for_provider", lambda *a, **kw: fake)

    seen = []
    async def _fake_preprocess(series_uid, study_uid, params, wado_url, image_folder,
                               selection=None):
        seen.append((series_uid, selection))
        return ["img"]
    monkeypatch.setattr(wh, "preprocess_series", _fake_preprocess)

    from session_manager import get_session_manager
    s = get_session_manager().create_session("SS1")
    ws = _FakeWebSocket()
    await wh.handle_chat(
        ws, s, content="q", study_uid="STD", series_uids=["SE1", "SE2"],
        slice_selections=[SliceSelection(series_uid="SE1", sop_instance_uids=["1.1", "1.2"])],
    )

    by_series = dict(seen)
    assert by_series["SE1"].sop_instance_uids == ["1.1", "1.2"]
    # SE2 had no selection: it falls back to the configured recipe rather than
    # inheriting SE1's slices.
    assert by_series["SE2"] is None


async def test_handle_chat_reruns_preprocessing_when_the_selection_changes(tmp_path, monkeypatch):
    """Two messages, same series, different slices -> two preprocess calls.

    Regression: with a series-only cache key the second message was answered with
    the first message's images while its snapshot claimed the new range.
    """
    _reset_singletons(tmp_path, monkeypatch)
    import websocket_handler as wh
    from models import SliceSelection
    fake = _FakeOllamaForChat([{"type": "content", "text": "ok"}])
    monkeypatch.setattr(wh, "get_client_for_provider", lambda *a, **kw: fake)

    calls = []
    async def _fake_preprocess(series_uid, study_uid, params, wado_url, image_folder,
                               selection=None):
        calls.append(list(selection.sop_instance_uids) if selection else None)
        return [f"img-for-{'-'.join(selection.sop_instance_uids)}"] if selection else ["img"]
    monkeypatch.setattr(wh, "preprocess_series", _fake_preprocess)

    from session_manager import get_session_manager
    s = get_session_manager().create_session("SS2")

    ws = _FakeWebSocket()
    await wh.handle_chat(ws, s, content="first", study_uid="STD", series_uids=["SE1"],
                         slice_selections=[SliceSelection(series_uid="SE1",
                                                          sop_instance_uids=["1.1"])])
    await wh.handle_chat(ws, s, content="second", study_uid="STD", series_uids=["SE1"],
                         slice_selections=[SliceSelection(series_uid="SE1",
                                                          sop_instance_uids=["1.9"])])
    assert calls == [["1.1"], ["1.9"]]

    # And the second call's images are the ones that reached the model.
    last_user_content = fake.calls[-1]["messages"][-1]["content"]
    urls = [p["image_url"]["url"] for p in last_user_content if p["type"] == "image_url"]
    assert urls == ["img-for-1.9"]


async def test_handle_chat_reuses_the_cache_for_an_identical_selection(tmp_path, monkeypatch):
    """Asking twice about the same slices must not re-retrieve the series."""
    _reset_singletons(tmp_path, monkeypatch)
    import websocket_handler as wh
    from models import SliceSelection
    fake = _FakeOllamaForChat([{"type": "content", "text": "ok"}])
    monkeypatch.setattr(wh, "get_client_for_provider", lambda *a, **kw: fake)

    calls = []
    async def _fake_preprocess(series_uid, study_uid, params, wado_url, image_folder,
                               selection=None):
        calls.append(series_uid)
        return ["img"]
    monkeypatch.setattr(wh, "preprocess_series", _fake_preprocess)

    from session_manager import get_session_manager
    s = get_session_manager().create_session("SS3")
    ws = _FakeWebSocket()
    for _ in range(2):
        await wh.handle_chat(ws, s, content="q", study_uid="STD", series_uids=["SE1"],
                             slice_selections=[SliceSelection(series_uid="SE1",
                                                              sop_instance_uids=["1.1", "1.2"])])
    assert calls == ["SE1"]


async def test_handle_chat_reports_an_unresolvable_selection_as_such(tmp_path, monkeypatch):
    """Retrieval succeeded; the slices are not in the series. Do not blame retrieval."""
    _reset_singletons(tmp_path, monkeypatch)
    import websocket_handler as wh
    from models import SliceSelection
    from preprocessing import SliceSelectionError
    fake = _FakeOllamaForChat([])
    monkeypatch.setattr(wh, "get_client_for_provider", lambda *a, **kw: fake)

    async def _fake_preprocess(*a, **kw):
        raise SliceSelectionError("2 of 5 selected slices are not part of the retrieved series.")
    monkeypatch.setattr(wh, "preprocess_series", _fake_preprocess)

    from session_manager import get_session_manager
    s = get_session_manager().create_session("SS4")
    ws = _FakeWebSocket()
    await wh.handle_chat(ws, s, content="q", study_uid="STD", series_uids=["SE1"],
                         slice_selections=[SliceSelection(series_uid="SE1",
                                                          sop_instance_uids=["1.1"])])
    errors = [m for m in ws.sent if m["type"] == "error"]
    assert len(errors) == 1
    assert "selected slices are not part" in errors[0]["content"]
    assert "Failed to retrieve" not in errors[0]["content"]


async def test_handle_chat_does_not_answer_when_the_selection_is_unresolvable(
    tmp_path, monkeypatch
):
    """An unresolvable selection must abort the turn, not fall back to other slices."""
    _reset_singletons(tmp_path, monkeypatch)
    import websocket_handler as wh
    from models import SliceSelection
    from preprocessing import SliceSelectionError
    fake = _FakeOllamaForChat([{"type": "content", "text": "should never run"}])
    monkeypatch.setattr(wh, "get_client_for_provider", lambda *a, **kw: fake)

    async def _fake_preprocess(*a, **kw):
        raise SliceSelectionError("nope")
    monkeypatch.setattr(wh, "preprocess_series", _fake_preprocess)

    from session_manager import get_session_manager
    sm = get_session_manager()
    s = sm.create_session("SS5")
    ws = _FakeWebSocket()
    await wh.handle_chat(ws, s, content="q", study_uid="STD", series_uids=["SE1"],
                         slice_selections=[SliceSelection(series_uid="SE1",
                                                          sop_instance_uids=["1.1"])])
    assert fake.calls == []
    assert sm.get_history("SS5") == []


def test_real_ws_carries_slice_selections_through_validation(tmp_path, monkeypatch):
    """The field survives ClientMessage validation and reaches preprocessing intact.

    Driven through the real route rather than handle_websocket directly: the
    dispatcher cancels its background task when the message stream ends, so a
    direct call races the very thing under test.
    """
    fake = _AwaitableOllama(chunks=[{"type": "content", "text": "ok"}], per_chunk_sleep_s=0.0)
    _app, client = _build_app_with_ws(tmp_path, monkeypatch, fake)
    import websocket_handler as wh

    seen = []
    async def _fake_preprocess(series_uid, study_uid, params, wado_url, image_folder,
                               selection=None):
        seen.append(selection)
        return ["img"]
    monkeypatch.setattr(wh, "preprocess_series", _fake_preprocess)

    with client.websocket_connect("/ws/chat/sess-sel-1") as ws:
        assert ws.receive_json()["type"] == "connected"
        ws.send_json({
            "type": "chat", "content": "q", "study_uid": "STD", "series_uids": ["SE1"],
            "slice_selections": [{
                "series_uid": "SE1", "sop_instance_uids": ["1.1", "1.2"],
                "range_start": 18, "range_end": 62, "total_slices": 103,
            }],
        })
        while ws.receive_json()["type"] != "done":
            pass

    assert len(seen) == 1
    assert seen[0].sop_instance_uids == ["1.1", "1.2"]
    assert (seen[0].range_start, seen[0].range_end, seen[0].total_slices) == (18, 62, 103)


def test_real_ws_rejects_an_oversized_slice_selection(tmp_path, monkeypatch):
    """A degenerate payload is refused by validation, never reaching retrieval."""
    fake = _AwaitableOllama(chunks=[{"type": "content", "text": "ok"}], per_chunk_sleep_s=0.0)
    _app, client = _build_app_with_ws(tmp_path, monkeypatch, fake)
    import websocket_handler as wh
    from models import MAX_SLICES_PER_SERIES

    called = []
    async def _fake_preprocess(*a, **kw):
        called.append(True)
        return ["img"]
    monkeypatch.setattr(wh, "preprocess_series", _fake_preprocess)

    with client.websocket_connect("/ws/chat/sess-sel-2") as ws:
        assert ws.receive_json()["type"] == "connected"
        ws.send_json({
            "type": "chat", "content": "q", "series_uids": ["SE1"],
            "slice_selections": [{
                "series_uid": "SE1",
                "sop_instance_uids": [f"1.{i}" for i in range(MAX_SLICES_PER_SERIES + 1)],
            }],
        })
        err = ws.receive_json()
        assert err["type"] == "error"

    assert called == []


# ---------- plan_series ----------
#
# OHIF can split one series into several display sets (one per instance for
# mammography and other single-image modalities). The panel sends a selection per
# display set, and those selections can legitimately differ — a region of interest
# on one and not the other. Collapsing them to one per series either drops a crop
# or applies it to images it was never drawn on.

def _sel(series_uid, uids, **kw):
    from models import SliceSelection
    return SliceSelection(series_uid=series_uid, sop_instance_uids=uids, **kw)


def _roi(**kw):
    from models import RegionOfInterest
    return RegionOfInterest(**{"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2, **kw})


def test_plan_series_keeps_one_entry_per_selection():
    import websocket_handler as wh
    plan = wh.plan_series(["SE1"], [_sel("SE1", ["1.1"]), _sel("SE1", ["1.2"])])
    assert [uids for _, s in plan for uids in [s.sop_instance_uids]] == [["1.1"], ["1.2"]]


def test_plan_series_keeps_a_crop_that_applies_to_only_one_display_set():
    """The regression: merging dropped this ROI or spread it over both."""
    import websocket_handler as wh
    plan = wh.plan_series(
        ["SE1"], [_sel("SE1", ["1.1"], roi=_roi()), _sel("SE1", ["1.2"])]
    )
    assert plan[0][1].roi is not None
    assert plan[1][1].roi is None


def test_plan_series_collapses_two_identical_requests():
    """Same series, same everything: one preprocessing pass, not two."""
    import websocket_handler as wh
    plan = wh.plan_series(["SE1"], [_sel("SE1", ["1.1"]), _sel("SE1", ["1.1"])])
    assert len(plan) == 1


def test_plan_series_treats_two_crops_of_one_series_as_distinct():
    import websocket_handler as wh
    plan = wh.plan_series(
        ["SE1"],
        [_sel("SE1", ["1.1"], roi=_roi()), _sel("SE1", ["1.1"], roi=_roi(x=0.5))],
    )
    assert len(plan) == 2


def test_plan_series_adds_series_that_carry_no_selection():
    import websocket_handler as wh
    plan = wh.plan_series(["SE1", "SE2"], [_sel("SE1", ["1.1"])])
    assert plan[-1] == ("SE2", None)


def test_plan_series_does_not_duplicate_a_series_listed_twice():
    """A series can appear twice when OHIF split it; retrieve it once."""
    import websocket_handler as wh
    plan = wh.plan_series(["SE1", "SE1"], [])
    assert plan == [("SE1", None)]


def test_plan_series_is_empty_for_a_message_with_no_images():
    import websocket_handler as wh
    assert wh.plan_series([], []) == []


async def test_handle_chat_preprocesses_each_display_set_of_a_split_series(
    tmp_path, monkeypatch
):
    """Both display sets' slices reach the model, each with its own crop."""
    _reset_singletons(tmp_path, monkeypatch)
    import websocket_handler as wh
    from models import RegionOfInterest, SliceSelection
    fake = _FakeOllamaForChat([{"type": "content", "text": "ok"}])
    monkeypatch.setattr(wh, "get_client_for_provider", lambda *a, **kw: fake)

    seen = []
    async def _fake_preprocess(series_uid, study_uid, params, wado_url, image_folder,
                               selection=None):
        seen.append((list(selection.sop_instance_uids), selection.roi is not None))
        return [f"img-{selection.sop_instance_uids[0]}"]
    monkeypatch.setattr(wh, "preprocess_series", _fake_preprocess)

    from session_manager import get_session_manager
    s = get_session_manager().create_session("SM2")
    ws = _FakeWebSocket()
    await wh.handle_chat(
        ws, s, content="q", study_uid="STD", series_uids=["SE1", "SE1"],
        slice_selections=[
            SliceSelection(series_uid="SE1", sop_instance_uids=["1.1"],
                           roi=RegionOfInterest(x=0.1, y=0.1, width=0.2, height=0.2)),
            SliceSelection(series_uid="SE1", sop_instance_uids=["1.2"]),
        ],
    )
    assert seen == [(["1.1"], True), (["1.2"], False)]

    # Both sets of images reached the model, not just the last.
    content_parts = fake.calls[-1]["messages"][-1]["content"]
    urls = [p["image_url"]["url"] for p in content_parts if p["type"] == "image_url"]
    assert urls == ["img-1.1", "img-1.2"]


async def test_handle_chat_reads_the_recipe_once_before_any_await(tmp_path, monkeypatch):
    """A config change mid-turn must not land this turn's images under the key
    computed from the old recipe."""
    _reset_singletons(tmp_path, monkeypatch)
    import websocket_handler as wh
    fake = _FakeOllamaForChat([{"type": "content", "text": "ok"}])
    monkeypatch.setattr(wh, "get_client_for_provider", lambda *a, **kw: fake)

    from runtime_config import get_runtime_config
    runtime = get_runtime_config()

    captured = {}
    async def _fake_preprocess(series_uid, study_uid, params, wado_url, image_folder,
                               selection=None):
        # Simulate a PUT landing while this turn is suspended.
        runtime.preprocessing.num_slices = 99
        captured["num_slices"] = params.num_slices
        return ["img"]
    monkeypatch.setattr(wh, "preprocess_series", _fake_preprocess)

    from session_manager import get_session_manager
    s = get_session_manager().create_session("SM3")
    ws = _FakeWebSocket()
    await wh.handle_chat(ws, s, content="q", study_uid="STD", series_uids=["SE1"])
    # The turn preprocessed with the recipe it was keyed against, not the new one.
    assert captured["num_slices"] == 5
