"""Tests for chat-middleware/websocket_handler.py — handle_chat + handle_websocket dispatcher."""
import asyncio
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
    monkeypatch.setattr(wh, "get_ollama_client", lambda *a, **kw: fake)

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
    monkeypatch.setattr(wh, "get_ollama_client", lambda *a, **kw: fake)

    async def _fake_preprocess(series_uid, study_uid, params, wado_url, image_folder):
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
    monkeypatch.setattr(wh, "get_ollama_client", lambda *a, **kw: fake)

    pp_calls = []
    async def _fake_preprocess(*a, **kw):
        pp_calls.append(a)
        return ["img"]
    monkeypatch.setattr(wh, "preprocess_series", _fake_preprocess)

    from image_cache import get_image_cache, CachedSeries
    cache = get_image_cache()
    cache.put("SE-cached", CachedSeries(series_uid="SE-cached", base64_images=["cached-img"]))

    from session_manager import get_session_manager
    s = get_session_manager().create_session("S3")
    ws = _FakeWebSocket()
    await wh.handle_chat(ws, s, content="q", study_uid="STD", series_uids=["SE-cached"])
    assert pp_calls == []                      # cache hit, no preprocess


async def test_handle_chat_emits_error_when_preprocess_fails(tmp_path, monkeypatch):
    _reset_singletons(tmp_path, monkeypatch)
    import websocket_handler as wh
    fake = _FakeOllamaForChat([])
    monkeypatch.setattr(wh, "get_ollama_client", lambda *a, **kw: fake)

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
    monkeypatch.setattr(wh, "get_ollama_client", lambda *a, **kw: fake)
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
    monkeypatch.setattr(wh, "get_ollama_client", lambda *a, **kw: fake)
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
    monkeypatch.setattr(wh, "get_ollama_client", lambda *a, **kw: _Boom())

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
    monkeypatch.setattr(wh, "get_ollama_client", lambda *a, **kw: _CancellingClient())

    ws = _FakeWebSocket()
    await wh.handle_chat(ws, s, content="x", study_uid="", series_uids=[])
    # History must NOT contain an assistant entry (the partial response).
    history = get_session_manager().get_history("S8")
    assert all(m.get("role") != "assistant" for m in history), \
        f"assistant entry leaked into history after cancel: {history}"
    # Specifically no entry at all (cancel happens before user message append too).
    assert history == []


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

import asyncio as _asyncio


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
    # Also reach the dispatcher'''s module-level alias if it imported get_ollama_client by name
    monkeypatch.setattr(wh, "get_ollama_client", lambda *a, **kw: fake_ollama)

    from fastapi import FastAPI, WebSocket
    from fastapi.testclient import TestClient

    app = FastAPI()

    @app.websocket("/ws/chat/{session_id}")
    async def _ws(websocket: WebSocket, session_id: str):
        await wh.handle_websocket(websocket, session_id)

    return app, TestClient(app)


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
