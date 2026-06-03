"""Tests for chat-middleware/ollama_client.py — async OpenAI-compatible SSE streaming client."""
import asyncio
import json

import pytest


# ---------------------------------------------------------------------------
# Lightweight aiohttp fakes — async context managers + async iteration.
# Avoids pulling aresponses; the SSE stream is just a list of bytes lines.
# ---------------------------------------------------------------------------

class _AsyncLineIter:
    def __init__(self, lines):
        self._lines = list(lines)
    def __aiter__(self):
        return self
    async def __anext__(self):
        if not self._lines:
            raise StopAsyncIteration
        return self._lines.pop(0)


class _FakeResponse:
    def __init__(self, status=200, body=b"", lines=None, json_payload=None):
        self.status = status
        self._body = body
        self._json = json_payload
        self.content = _AsyncLineIter(lines or [])
        self.closed = False

    async def text(self):
        return self._body.decode("utf-8")

    async def json(self):
        if self._json is not None:
            return self._json
        return json.loads(self._body)

    def close(self):
        self.closed = True


class _RespCM:
    def __init__(self, resp):
        self._resp = resp
    async def __aenter__(self):
        return self._resp
    async def __aexit__(self, *exc):
        return False


class _FakeAiohttpSession:
    """Fake aiohttp.ClientSession that returns a pre-built _FakeResponse for post/get."""
    def __init__(self, post_response=None, get_response=None):
        self._post_response = post_response
        self._get_response = get_response
        self.post_calls = []
        self.get_calls = []

    async def __aenter__(self):
        return self
    async def __aexit__(self, *exc):
        return False

    def post(self, url, **kw):
        self.post_calls.append({"url": url, **kw})
        return _RespCM(self._post_response)

    def get(self, url, **kw):
        self.get_calls.append({"url": url, **kw})
        return _RespCM(self._get_response)


def _sse_line(d):
    """Build an SSE data: line as bytes (matches what aiohttp StreamReader yields)."""
    return f"data: {json.dumps(d)}\n".encode("utf-8")


def _patch_session(monkeypatch, session):
    """Wire ollama_client.aiohttp.ClientSession to return our fake session."""
    import ollama_client
    monkeypatch.setattr(
        ollama_client.aiohttp, "ClientSession",
        lambda *a, **kw: session,
    )


# ---------- chat_stream ----------

async def test_chat_stream_yields_content_chunks_in_order(monkeypatch):
    """SSE chunks with content deltas are yielded as type=content events."""
    import ollama_client
    resp = _FakeResponse(status=200, lines=[
        _sse_line({"choices": [{"delta": {"content": "Hello "}}]}),
        _sse_line({"choices": [{"delta": {"content": "world"}}]}),
        b"data: [DONE]\n",
    ])
    session = _FakeAiohttpSession(post_response=resp)
    _patch_session(monkeypatch, session)

    client = ollama_client.OllamaClient(base_url="http://x", model="m")
    chunks = [c async for c in client.chat_stream(messages=[{"role": "user", "content": "hi"}])]
    assert chunks == [
        {"type": "content", "text": "Hello "},
        {"type": "content", "text": "world"},
    ]


async def test_chat_stream_yields_thinking_chunks_when_reasoning_content_present(monkeypatch):
    """reasoning_content delta -> type=thinking."""
    import ollama_client
    resp = _FakeResponse(status=200, lines=[
        _sse_line({"choices": [{"delta": {"reasoning_content": "let me think"}}]}),
        _sse_line({"choices": [{"delta": {"content": "answer"}}]}),
        b"data: [DONE]\n",
    ])
    session = _FakeAiohttpSession(post_response=resp)
    _patch_session(monkeypatch, session)

    client = ollama_client.OllamaClient(base_url="http://x", model="m")
    chunks = [c async for c in client.chat_stream(messages=[])]
    assert chunks == [
        {"type": "thinking", "text": "let me think"},
        {"type": "content", "text": "answer"},
    ]


async def test_chat_stream_skips_non_data_and_blank_lines(monkeypatch):
    """Non-'data: ' prefixed lines and blank lines are dropped silently.

    H3 fix: use b"\\n" (newline byte) for the blank-line case rather than b"" — the latter
    triggers the early-return `if not raw_line: continue` shortcut, bypassing the live
    decode + startswith branch the production code actually depends on."""
    import ollama_client
    resp = _FakeResponse(status=200, lines=[
        b"\n",                                                                       # blank SSE separator (real wire shape)
        b": comment\n",                                                              # not data:
        _sse_line({"choices": [{"delta": {"content": "ok"}}]}),
        b"data: [DONE]\n",
    ])
    session = _FakeAiohttpSession(post_response=resp)
    _patch_session(monkeypatch, session)

    client = ollama_client.OllamaClient(base_url="http://x", model="m")
    chunks = [c async for c in client.chat_stream(messages=[])]
    assert chunks == [{"type": "content", "text": "ok"}]


async def test_chat_stream_swallows_invalid_json_chunks(monkeypatch):
    """JSON decode errors on a chunk are logged and skipped — no crash."""
    import ollama_client
    resp = _FakeResponse(status=200, lines=[
        b"data: {not-valid-json\n",
        _sse_line({"choices": [{"delta": {"content": "after-bad"}}]}),
        b"data: [DONE]\n",
    ])
    session = _FakeAiohttpSession(post_response=resp)
    _patch_session(monkeypatch, session)

    client = ollama_client.OllamaClient(base_url="http://x", model="m")
    chunks = [c async for c in client.chat_stream(messages=[])]
    assert chunks == [{"type": "content", "text": "after-bad"}]


async def test_chat_stream_raises_on_non_200_status(monkeypatch):
    """Non-200 surfaces as a generic Exception with the body text."""
    import ollama_client
    resp = _FakeResponse(status=500, body=b"backend overloaded")
    session = _FakeAiohttpSession(post_response=resp)
    _patch_session(monkeypatch, session)

    client = ollama_client.OllamaClient(base_url="http://x", model="m")
    with pytest.raises(Exception, match="Ollama API error: 500"):
        async for _ in client.chat_stream(messages=[]):
            pass


async def test_chat_stream_honors_cancel_event_mid_stream(monkeypatch):
    """When cancel_event is set partway through, the stream stops + closes the response."""
    import ollama_client
    resp = _FakeResponse(status=200, lines=[
        _sse_line({"choices": [{"delta": {"content": "first"}}]}),
        _sse_line({"choices": [{"delta": {"content": "second"}}]}),
        _sse_line({"choices": [{"delta": {"content": "never"}}]}),
        b"data: [DONE]\n",
    ])
    session = _FakeAiohttpSession(post_response=resp)
    _patch_session(monkeypatch, session)

    cancel_event = asyncio.Event()
    client = ollama_client.OllamaClient(base_url="http://x", model="m")
    collected = []
    async for chunk in client.chat_stream(messages=[], cancel_event=cancel_event):
        collected.append(chunk)
        if len(collected) == 2:
            cancel_event.set()         # request cancellation after second chunk
    assert collected == [
        {"type": "content", "text": "first"},
        {"type": "content", "text": "second"},
    ]
    assert resp.closed is True


async def test_chat_stream_forwards_runtime_options_to_payload(monkeypatch):
    """runtime_options''' supported keys are propagated to the POST JSON body."""
    import ollama_client
    resp = _FakeResponse(status=200, lines=[b"data: [DONE]\n"])
    session = _FakeAiohttpSession(post_response=resp)
    _patch_session(monkeypatch, session)

    client = ollama_client.OllamaClient(base_url="http://x", model="m")
    runtime_options = {
        "max_tokens": 256, "temperature": 0.2, "top_p": 0.9,
        "stop": ["</s>"], "seed": 42, "think": True,
        "unsupported_key": "should-be-skipped",
    }
    [_ async for _ in client.chat_stream(messages=[{"role": "user", "content": "x"}],
                                          runtime_options=runtime_options)]
    payload = session.post_calls[0]["json"]
    for k in ("max_tokens", "temperature", "top_p", "stop", "seed", "think"):
        assert payload[k] == runtime_options[k]
    assert "unsupported_key" not in payload
    assert payload["stream"] is True
    assert payload["model"] == "m"


async def test_chat_stream_skips_options_with_none_value(monkeypatch):
    """runtime_options with None values are skipped (per `is not None` check)."""
    import ollama_client
    resp = _FakeResponse(status=200, lines=[b"data: [DONE]\n"])
    session = _FakeAiohttpSession(post_response=resp)
    _patch_session(monkeypatch, session)

    client = ollama_client.OllamaClient(base_url="http://x", model="m")
    [_ async for _ in client.chat_stream(messages=[], runtime_options={"max_tokens": None, "temperature": 0.5})]
    payload = session.post_calls[0]["json"]
    assert "max_tokens" not in payload
    assert payload["temperature"] == 0.5


# ---------- health_check ----------

async def test_health_check_ollama_returns_true_on_200(monkeypatch):
    import ollama_client
    resp = _FakeResponse(status=200, body=b"{}")
    session = _FakeAiohttpSession(get_response=resp)
    _patch_session(monkeypatch, session)
    client = ollama_client.OllamaClient(base_url="http://x", model="m", backend_type="ollama")
    assert await client.health_check() is True
    assert session.get_calls[0]["url"] == "http://x/api/tags"


async def test_health_check_llamacpp_uses_health_endpoint(monkeypatch):
    import ollama_client
    resp = _FakeResponse(status=200, body=b"{}")
    session = _FakeAiohttpSession(get_response=resp)
    _patch_session(monkeypatch, session)
    client = ollama_client.OllamaClient(base_url="http://x/", model="m", backend_type="llamacpp")
    assert await client.health_check() is True
    assert session.get_calls[0]["url"] == "http://x/health"     # trailing / stripped


async def test_health_check_returns_false_on_non_200(monkeypatch):
    import ollama_client
    resp = _FakeResponse(status=503, body=b"")
    session = _FakeAiohttpSession(get_response=resp)
    _patch_session(monkeypatch, session)
    client = ollama_client.OllamaClient(base_url="http://x", model="m")
    assert await client.health_check() is False


async def test_health_check_swallows_exception_and_returns_false(monkeypatch):
    """Network exception -> caught -> returns False."""
    import ollama_client
    def _raise(*a, **kw):
        raise RuntimeError("connection refused")
    monkeypatch.setattr(ollama_client.aiohttp, "ClientSession", _raise)
    client = ollama_client.OllamaClient(base_url="http://x", model="m")
    assert await client.health_check() is False


# ---------- list_models ----------

async def test_list_models_ollama_format(monkeypatch):
    import ollama_client
    resp = _FakeResponse(status=200, json_payload={"models": [{"name": "a"}, {"name": "b"}]})
    session = _FakeAiohttpSession(get_response=resp)
    _patch_session(monkeypatch, session)
    client = ollama_client.OllamaClient(base_url="http://x", model="m", backend_type="ollama")
    assert await client.list_models() == ["a", "b"]
    assert session.get_calls[0]["url"] == "http://x/api/tags"


async def test_list_models_llamacpp_format(monkeypatch):
    import ollama_client
    resp = _FakeResponse(status=200, json_payload={"data": [{"id": "x"}, {"id": "y"}]})
    session = _FakeAiohttpSession(get_response=resp)
    _patch_session(monkeypatch, session)
    client = ollama_client.OllamaClient(base_url="http://x", model="m", backend_type="llamacpp")
    assert await client.list_models() == ["x", "y"]
    assert session.get_calls[0]["url"] == "http://x/v1/models"


async def test_list_models_returns_empty_on_non_200(monkeypatch):
    import ollama_client
    resp = _FakeResponse(status=500, body=b"")
    session = _FakeAiohttpSession(get_response=resp)
    _patch_session(monkeypatch, session)
    client = ollama_client.OllamaClient(base_url="http://x", model="m")
    assert await client.list_models() == []


async def test_list_models_swallows_exception_and_returns_empty(monkeypatch):
    import ollama_client
    def _raise(*a, **kw):
        raise RuntimeError("network down")
    monkeypatch.setattr(ollama_client.aiohttp, "ClientSession", _raise)
    client = ollama_client.OllamaClient(base_url="http://x", model="m")
    assert await client.list_models() == []


# ---------- get_ollama_client singleton ----------

def test_get_ollama_client_singleton_initializes_from_config(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGE_FOLDER", str(tmp_path / "oc-img"))
    monkeypatch.setenv("OLLAMA_URL", "http://from-env:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "MyEnvModel:8b")
    import config
    config.config = None
    import ollama_client
    ollama_client.reset_ollama_client()
    c = ollama_client.get_ollama_client()
    assert c.base_url == "http://from-env:11434"
    assert c.model == "MyEnvModel:8b"


def test_get_ollama_client_returns_existing_singleton(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGE_FOLDER", str(tmp_path / "oc-img"))
    import config
    config.config = None
    import ollama_client
    ollama_client.reset_ollama_client()
    a = ollama_client.get_ollama_client()
    b = ollama_client.get_ollama_client()
    assert a is b


def test_get_ollama_client_warns_on_base_url_mismatch(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("IMAGE_FOLDER", str(tmp_path / "oc-img"))
    monkeypatch.setenv("OLLAMA_URL", "http://first")
    import config
    config.config = None
    import ollama_client
    ollama_client.reset_ollama_client()
    ollama_client.get_ollama_client()
    with caplog.at_level("WARNING"):
        c = ollama_client.get_ollama_client(base_url="http://second")
    assert c.base_url == "http://first"
    assert any("already initialized" in r.message for r in caplog.records)


def test_get_ollama_client_warns_on_model_mismatch(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("IMAGE_FOLDER", str(tmp_path / "oc-img"))
    monkeypatch.setenv("OLLAMA_MODEL", "ModelA")
    import config
    config.config = None
    import ollama_client
    ollama_client.reset_ollama_client()
    ollama_client.get_ollama_client()
    with caplog.at_level("WARNING"):
        c = ollama_client.get_ollama_client(model="ModelB")
    assert c.model == "ModelA"
    assert any("already initialized" in r.message for r in caplog.records)
