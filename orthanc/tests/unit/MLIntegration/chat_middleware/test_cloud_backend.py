"""Tests for the optional hosted cloud backend (Ollama Cloud or OpenRouter).

Covers the operator gate, the Bearer auth header, capability detection, the
per-provider wiring, and the guarantee that the API key never leaves the service.
"""
import asyncio
import json

import pytest


# ---------------------------------------------------------------------------
# aiohttp fakes. Mirrors the style of test_ollama_client.py, but records the
# requests so the auth header and the /api/show fan-out can be asserted on.
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, status=200, json_payload=None, body=b""):
        self.status = status
        self._json = json_payload
        self._body = body

    async def json(self):
        if self._json is not None:
            return self._json
        return json.loads(self._body)

    async def text(self):
        return self._body.decode("utf-8")

    def close(self):
        pass


class _RespCM:
    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *exc):
        return False


class _RecordingSession:
    """Session that records every call and replies from a routing table."""

    def __init__(self, routes):
        # routes: {(method, path_suffix): _FakeResponse or callable(body)->_FakeResponse}
        self._routes = routes
        self.calls = []

    def _resolve(self, method, url, json_body):
        for (m, suffix), resp in self._routes.items():
            if m == method and url.endswith(suffix):
                return resp(json_body) if callable(resp) else resp
        return _FakeResponse(status=404, body=b"no route")

    def get(self, url, headers=None, **kw):
        self.calls.append({"method": "GET", "url": url, "headers": headers or {}, "json": None})
        return _RespCM(self._resolve("GET", url, None))

    def post(self, url, json=None, headers=None, **kw):
        self.calls.append({"method": "POST", "url": url, "headers": headers or {}, "json": json})
        return _RespCM(self._resolve("POST", url, json))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


@pytest.fixture
def patch_session(monkeypatch):
    """Install a _RecordingSession as aiohttp.ClientSession; return the instance."""
    holder = {}

    def _install(routes):
        import aiohttp
        session = _RecordingSession(routes)
        holder["session"] = session
        monkeypatch.setattr(aiohttp, "ClientSession", lambda *a, **kw: session)
        return session

    return _install


def _reset_config(monkeypatch, **env):
    """Reset the config singleton and apply env vars."""
    for k in (
        "ALLOW_CLOUD_BACKEND",
        "CLOUD_PROVIDER",
        "CLOUD_MODEL_FILTER",
        "OLLAMA_API_KEY",
        "OLLAMA_CLOUD_URL",
        "OLLAMA_CLOUD_MODEL",
        "OPENROUTER_API_KEY",
        "OPENROUTER_URL",
        "OPENROUTER_MODEL",
        "OPENROUTER_ALLOW_DATA_COLLECTION",
    ):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import config
    config.config = None
    return config.init_config()


# ---------------------------------------------------------------------------
# config: the gate and its parsing
# ---------------------------------------------------------------------------

def test_cloud_disabled_by_default(monkeypatch):
    """Cloud must be off unless an operator opts in — it sends slices off-site."""
    cfg = _reset_config(monkeypatch)
    assert cfg.allow_cloud_backend is False
    assert cfg.cloud_api_key == ""
    assert cfg.cloud_url == "https://ollama.com"
    assert cfg.cloud_provider == "ollama"  # unchanged for deployments that predate the choice


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on", " True "])
def test_gate_accepts_truthy_spellings(monkeypatch, raw):
    assert _reset_config(monkeypatch, ALLOW_CLOUD_BACKEND=raw).allow_cloud_backend is True


@pytest.mark.parametrize("raw", ["0", "false", "no", "off", "", "banana"])
def test_gate_treats_anything_else_as_off(monkeypatch, raw):
    """An unparseable value must fail closed, not silently enable egress."""
    assert _reset_config(monkeypatch, ALLOW_CLOUD_BACKEND=raw).allow_cloud_backend is False


# ---------------------------------------------------------------------------
# runtime_config: provider selection
# ---------------------------------------------------------------------------

def test_provider_starts_local_even_when_cloud_enabled(monkeypatch):
    """A restart must never come back up already routed to the cloud."""
    _reset_config(monkeypatch, ALLOW_CLOUD_BACKEND="1", OLLAMA_API_KEY="k", OLLAMA_CLOUD_MODEL="qwen3.5")
    import runtime_config
    runtime_config._runtime_config = None
    rc = runtime_config.get_runtime_config()
    assert rc.provider.value == "local"


def test_active_model_follows_provider(monkeypatch):
    """Switching provider must not overwrite the other provider's model choice."""
    _reset_config(monkeypatch, ALLOW_CLOUD_BACKEND="1", OLLAMA_API_KEY="k")
    import runtime_config
    runtime_config._runtime_config = None
    rc = runtime_config.get_runtime_config()

    rc.update(model="local-model:7b", cloud_model="cloud-model:100b")
    assert rc.active_model == "local-model:7b"

    rc.update(provider="cloud")
    assert rc.active_model == "cloud-model:100b"

    rc.update(provider="local")
    assert rc.active_model == "local-model:7b"  # cloud choice did not clobber it


# ---------------------------------------------------------------------------
# ollama_client: auth header + factory refusals
# ---------------------------------------------------------------------------

def test_local_client_sends_no_auth_header():
    from ollama_client import OllamaClient
    assert OllamaClient("http://localhost:11434", "m")._auth_headers() == {}


def test_api_key_becomes_bearer_header():
    from ollama_client import OllamaClient
    c = OllamaClient("https://ollama.com", "m", api_key="sk-secret")
    assert c._auth_headers() == {"Authorization": "Bearer sk-secret"}


def test_api_key_absent_from_repr():
    """The key must not leak through an object dump in a log or traceback."""
    from ollama_client import OllamaClient
    c = OllamaClient("https://ollama.com", "m", api_key="sk-secret")
    assert "sk-secret" not in repr(c)


def test_build_cloud_client_refuses_when_gate_off(monkeypatch):
    _reset_config(monkeypatch, OLLAMA_API_KEY="k")
    from ollama_client import CloudBackendUnavailableError, build_cloud_client
    with pytest.raises(CloudBackendUnavailableError, match="disabled"):
        build_cloud_client(model="qwen3.5")


def test_build_cloud_client_refuses_without_api_key(monkeypatch):
    _reset_config(monkeypatch, ALLOW_CLOUD_BACKEND="1")
    from ollama_client import CloudBackendUnavailableError, build_cloud_client
    with pytest.raises(CloudBackendUnavailableError, match="API key"):
        build_cloud_client(model="qwen3.5")


def test_build_cloud_client_refuses_without_model(monkeypatch):
    """Better to refuse here than to fail mid-chat after preprocessing a series."""
    _reset_config(monkeypatch, ALLOW_CLOUD_BACKEND="1", OLLAMA_API_KEY="k")
    from ollama_client import CloudBackendUnavailableError, build_cloud_client
    with pytest.raises(CloudBackendUnavailableError, match="model"):
        build_cloud_client()


@pytest.mark.parametrize(
    ("env", "expected", "forbidden"),
    [
        ({"OLLAMA_API_KEY": "k"}, "OLLAMA_CLOUD_MODEL", "OLLAMA_MODEL"),
        (
            {"CLOUD_PROVIDER": "openrouter", "OPENROUTER_API_KEY": "k"},
            "OPENROUTER_MODEL",
            "OPENROUTER_CLOUD_MODEL",
        ),
    ],
)
def test_refusal_names_the_model_env_var_that_exists(monkeypatch, env, expected, forbidden):
    """The message has to name a variable the operator can actually set.

    It used to be built as f"{provider.upper()}_MODEL", which on Ollama produced
    OLLAMA_MODEL — the *local* model, so following the advice would have changed
    which model the self-hosted backend runs and left the cloud error in place.
    No formula fixes both: OpenRouter has no local counterpart and so no CLOUD in
    its name, and Ollama itself only uses CLOUD in some of its variables.
    """
    _reset_config(monkeypatch, ALLOW_CLOUD_BACKEND="1", **env)
    from ollama_client import CloudBackendUnavailableError, build_cloud_client

    with pytest.raises(CloudBackendUnavailableError) as excinfo:
        build_cloud_client()

    message = str(excinfo.value)
    assert expected in message
    assert forbidden not in message


def test_build_cloud_client_uses_cloud_url_and_key(monkeypatch):
    _reset_config(monkeypatch, ALLOW_CLOUD_BACKEND="1", OLLAMA_API_KEY="sk-1", OLLAMA_CLOUD_MODEL="qwen3.5")
    from ollama_client import build_cloud_client
    c = build_cloud_client()
    assert c.base_url == "https://ollama.com"
    assert c.model == "qwen3.5"
    assert c.api_key == "sk-1"


def test_cloud_client_is_not_the_shared_local_singleton(monkeypatch):
    """The keyed client must not become the process-wide singleton."""
    _reset_config(monkeypatch, ALLOW_CLOUD_BACKEND="1", OLLAMA_API_KEY="sk-1")
    import ollama_client
    ollama_client._ollama_client = None
    local = ollama_client.get_ollama_client()
    cloud = ollama_client.build_cloud_client(model="qwen3.5")
    assert cloud is not local
    assert local.api_key is None  # local client never acquires the key
    assert ollama_client.get_ollama_client() is local  # singleton untouched


def test_get_client_for_provider_local_applies_model_override(monkeypatch):
    _reset_config(monkeypatch)
    import ollama_client
    ollama_client._ollama_client = None
    c = ollama_client.get_client_for_provider("local", "other:7b")
    assert c.model == "other:7b"
    assert c.api_key is None


# ---------------------------------------------------------------------------
# chat_stream / listing send the auth header
# ---------------------------------------------------------------------------

def test_chat_stream_sends_bearer_header(patch_session):
    from ollama_client import OllamaClient

    session = patch_session({("POST", "/v1/chat/completions"): _FakeResponse(status=500, body=b"nope")})
    c = OllamaClient("https://ollama.com", "m", api_key="sk-secret")

    async def run():
        with pytest.raises(Exception):
            async for _ in c.chat_stream([{"role": "user", "content": "hi"}]):
                pass

    asyncio.run(run())
    post = next(x for x in session.calls if x["method"] == "POST")
    assert post["headers"]["Authorization"] == "Bearer sk-secret"


def test_list_models_detailed_prefers_api_show_over_api_tags(patch_session):
    """/api/tags under-reports vision; /api/show is authoritative.

    Verified against Ollama 0.32.11: /api/tags returned ["completion"] for a model
    whose /api/show returned ["completion", "vision"] and which does read images.
    A regression here would silently mislabel vision models as text-only.
    """
    from ollama_client import OllamaClient

    session = patch_session({
        ("GET", "/api/tags"): _FakeResponse(json_payload={
            "models": [
                {"name": "vision-model:1b", "capabilities": ["completion"]},   # under-reported
                {"name": "text-model:1b", "capabilities": ["completion"]},
            ]
        }),
        ("POST", "/api/show"): lambda body: _FakeResponse(json_payload={
            "capabilities": ["completion", "vision"]
            if body["model"] == "vision-model:1b"
            else ["completion"]
        }),
    })

    c = OllamaClient("https://ollama.com", "m", api_key="sk-secret")
    models = asyncio.run(c.list_models_detailed())

    by_name = {m["name"]: m for m in models}
    assert by_name["vision-model:1b"]["supports_vision"] is True
    assert by_name["text-model:1b"]["supports_vision"] is False

    # every /api/show carried the key too
    shows = [x for x in session.calls if x["method"] == "POST"]
    assert len(shows) == 2
    assert all(x["headers"]["Authorization"] == "Bearer sk-secret" for x in shows)


def test_list_models_detailed_keeps_model_when_show_fails(patch_session):
    """An /api/show failure yields unknown capabilities, not a dropped model."""
    from ollama_client import OllamaClient

    patch_session({
        ("GET", "/api/tags"): _FakeResponse(json_payload={"models": [{"name": "m:1b"}]}),
        ("POST", "/api/show"): _FakeResponse(status=500, body=b"boom"),
    })
    models = asyncio.run(OllamaClient("https://ollama.com", "m").list_models_detailed())
    assert models == [
        {
            "name": "m:1b",
            "capabilities": [],
            "supports_vision": False,
            "context_length": None,
            "tokens_per_image": None,
        }
    ]


def test_list_models_detailed_raises_on_tags_failure(patch_session):
    """A rejected key must surface as an error, not an empty (=='no models') list."""
    from ollama_client import ModelListError, OllamaClient

    patch_session({("GET", "/api/tags"): _FakeResponse(status=401, body=b"unauthorized")})
    with pytest.raises(ModelListError, match="401"):
        asyncio.run(OllamaClient("https://ollama.com", "m", api_key="bad").list_models_detailed())


def test_listing_error_names_the_failure_when_it_stringifies_empty(monkeypatch):
    """A timeout must not produce a blank reason.

    `str(TimeoutError())` is "", which yielded a bare "Model listing failed: " in
    the chat panel — observed live against ollama.com during a transient egress
    stall, with nothing to act on.
    """
    import aiohttp
    from ollama_client import ModelListError, OllamaClient

    class _Boom:
        def get(self, *a, **kw):
            raise TimeoutError()  # noqa: TRY301 — empty str() is the point

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(aiohttp, "ClientSession", lambda *a, **kw: _Boom())

    with pytest.raises(ModelListError) as excinfo:
        asyncio.run(OllamaClient("https://ollama.com", "m").list_models_detailed())
    assert "TimeoutError" in str(excinfo.value)


def test_describe_prefers_message_when_present():
    from ollama_client import _describe

    assert _describe(ValueError("bad thing")) == "ValueError: bad thing"
    assert _describe(TimeoutError()) == "TimeoutError"


def test_subscription_error_surfaces_the_sentence_not_the_json(patch_session):
    """A 403 from a gated cloud model must read as prose, not a JSON envelope.

    Observed live: selecting qwen3.5:397b on a free plan returned
    {"error": {"message": "this model requires a subscription, upgrade for
    access: https://ollama.com/upgrade (ref: ...)"}} and the chat panel showed the
    whole blob, burying the one sentence that says what to do.
    """
    from ollama_client import OllamaClient, UpstreamChatError

    body = json.dumps(
        {
            "error": {
                "message": (
                    "this model requires a subscription, upgrade for access: "
                    "https://ollama.com/upgrade (ref: abc123)"
                ),
                "type": "api_error",
                "param": None,
                "code": None,
            }
        }
    ).encode()

    patch_session({("POST", "/v1/chat/completions"): _FakeResponse(status=403, body=body)})
    client = OllamaClient("https://ollama.com", "qwen3.5:397b", api_key="sk-1")

    async def run():
        async for _ in client.chat_stream([{"role": "user", "content": "hi"}]):
            pass

    with pytest.raises(UpstreamChatError) as excinfo:
        asyncio.run(run())

    message = str(excinfo.value)
    assert "requires a subscription" in message
    assert "https://ollama.com/upgrade" in message
    assert "HTTP 403" in message
    # None of the JSON envelope leaks through.
    assert '"error"' not in message
    assert "api_error" not in message


def test_upstream_message_handles_the_shapes_backends_actually_send():
    from ollama_client import _upstream_message

    # Ollama Cloud / OpenAI-compatible envelope
    assert (
        _upstream_message('{"error": {"message": "nope", "type": "api_error"}}') == "nope"
    )
    # Local Ollama's bare string form
    assert _upstream_message('{"error": "model not found"}') == "model not found"
    # llama.cpp sometimes answers in plain text
    assert _upstream_message("upstream exploded") == "upstream exploded"
    # Empty body still yields something sayable
    assert "no detail" in _upstream_message("")


# ---------------------------------------------------------------------------
# chat_stream connect retry. Scoped tightly: a repeated generation would bill and
# run twice, so only provable connect failures may be retried.
# ---------------------------------------------------------------------------

@pytest.fixture
def no_retry_delay(monkeypatch):
    """Collapse the retry backoff. Binds the real asyncio.sleep first, or the
    replacement would call itself and blow the stack."""
    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda *_a, **_kw: real_sleep(0))


def _sse(*texts):
    lines = []
    for t in texts:
        lines.append(
            f'data: {{"choices":[{{"delta":{{"content":"{t}"}}}}]}}'.encode()
        )
    lines.append(b"data: [DONE]")
    return lines


class _StreamResponse:
    """200 response yielding SSE lines."""

    def __init__(self, lines):
        self.status = 200
        self._lines = list(lines)
        self.content = _AsyncLines(self._lines)

    async def text(self):
        return ""

    def close(self):
        pass


class _AsyncLines:
    def __init__(self, lines):
        self._lines = list(lines)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._lines:
            raise StopAsyncIteration
        return self._lines.pop(0)


def _session_factory(monkeypatch, behaviours):
    """aiohttp.ClientSession whose nth post() follows behaviours[n].

    Each behaviour is either an exception instance to raise or a response object.
    Returns a dict recording how many posts were attempted.
    """
    import aiohttp

    state = {"posts": 0}

    class _Session:
        def post(self, url, json=None, headers=None, **kw):
            i = state["posts"]
            state["posts"] += 1
            behaviour = behaviours[min(i, len(behaviours) - 1)]
            if isinstance(behaviour, Exception):
                raise behaviour
            return _RespCM(behaviour)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(aiohttp, "ClientSession", lambda *a, **kw: _Session())
    return state


def _connector_error():
    import aiohttp
    from aiohttp.client_reqrep import ConnectionKey

    key = ConnectionKey("ollama.com", 443, True, True, None, None, None)
    return aiohttp.ClientConnectorError(key, OSError("handshake stalled"))


async def _drain(client):
    return [c async for c in client.chat_stream([{"role": "user", "content": "hi"}])]


def test_chat_retries_once_when_the_connection_never_opened(monkeypatch, no_retry_delay):
    """A stalled TLS handshake provably never reached the model, so resend it."""
    from ollama_client import OllamaClient

    state = _session_factory(
        monkeypatch, [_connector_error(), _StreamResponse(_sse("hello"))]
    )

    chunks = asyncio.run(_drain(OllamaClient("https://ollama.com", "m")))
    assert state["posts"] == 2
    assert [c["text"] for c in chunks] == ["hello"]


def test_chat_does_not_retry_a_server_disconnect(monkeypatch, no_retry_delay):
    """The request may already have reached the model; resending would double-run it."""
    import aiohttp
    from ollama_client import OllamaClient

    state = _session_factory(monkeypatch, [aiohttp.ServerDisconnectedError()])

    with pytest.raises(aiohttp.ServerDisconnectedError):
        asyncio.run(_drain(OllamaClient("https://ollama.com", "m")))
    assert state["posts"] == 1


def test_chat_does_not_retry_a_bare_timeout(monkeypatch, no_retry_delay):
    """Generation may be under way; only *connect* timeouts are safe to repeat."""
    from ollama_client import OllamaClient

    state = _session_factory(monkeypatch, [TimeoutError()])

    with pytest.raises(TimeoutError):
        asyncio.run(_drain(OllamaClient("https://ollama.com", "m")))
    assert state["posts"] == 1


def test_chat_does_not_retry_an_http_error(monkeypatch, no_retry_delay):
    """A 403 is a real answer from the host — repeating it only delays the message."""
    from ollama_client import OllamaClient, UpstreamChatError

    body = json.dumps({"error": {"message": "requires a subscription"}}).encode()
    state = _session_factory(monkeypatch, [_FakeResponse(status=403, body=body)])

    with pytest.raises(UpstreamChatError, match="requires a subscription"):
        asyncio.run(_drain(OllamaClient("https://ollama.com", "m")))
    assert state["posts"] == 1


def test_chat_gives_up_after_the_attempt_limit(monkeypatch, no_retry_delay):
    from ollama_client import CHAT_CONNECT_ATTEMPTS, OllamaClient

    state = _session_factory(monkeypatch, [_connector_error()])

    with pytest.raises(Exception, match="Cannot connect"):
        asyncio.run(_drain(OllamaClient("https://ollama.com", "m")))
    assert state["posts"] == CHAT_CONNECT_ATTEMPTS


def test_chat_sets_an_explicit_connect_timeout(monkeypatch):
    """Without sock_connect, a stalled handshake fell through to asyncio's 60s
    watchdog instead of failing promptly (observed as a 61.3s failure)."""
    import aiohttp
    from ollama_client import CONNECT_TIMEOUT_SECONDS, OllamaClient

    seen = {}
    real_timeout = aiohttp.ClientTimeout

    class _Session:
        def post(self, *a, **kw):
            return _RespCM(_StreamResponse(_sse("x")))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    def _capture(*a, **kw):
        seen.update(kw)
        return _Session()

    monkeypatch.setattr(aiohttp, "ClientSession", _capture)
    asyncio.run(_drain(OllamaClient("https://ollama.com", "m")))

    timeout = seen.get("timeout")
    assert isinstance(timeout, real_timeout)
    assert timeout.sock_connect == CONNECT_TIMEOUT_SECONDS
    assert timeout.total == 300


def test_catalogue_request_is_retried_after_a_stalled_connection(monkeypatch):
    """A single stalled TLS handshake should become a slow success, not an error.

    Observed against ollama.com from inside Docker: a request that timed out
    succeeded 0.2s later on a fresh connection.
    """
    import aiohttp
    from ollama_client import OllamaClient

    attempts = {"n": 0}

    class _FlakySession:
        def get(self, url, headers=None, **kw):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise TimeoutError()
            return _RespCM(_FakeResponse(json_payload={"models": [{"name": "m:1b"}]}))

        def post(self, url, json=None, headers=None, **kw):
            return _RespCM(_FakeResponse(json_payload={"capabilities": ["completion", "vision"]}))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(aiohttp, "ClientSession", lambda *a, **kw: _FlakySession())

    models = asyncio.run(OllamaClient("https://ollama.com", "m").list_models_detailed())
    assert attempts["n"] == 2  # retried once
    assert models == [
        {
            "name": "m:1b",
            "capabilities": ["completion", "vision"],
            "supports_vision": True,
            "context_length": None,
            "tokens_per_image": None,
        }
    ]


def test_http_error_is_not_retried(monkeypatch):
    """A 401 is a definitive answer; retrying it would just stall the panel."""
    import aiohttp
    from ollama_client import ModelListError, OllamaClient

    attempts = {"n": 0}

    class _UnauthorizedSession:
        def get(self, url, headers=None, **kw):
            attempts["n"] += 1
            return _RespCM(_FakeResponse(status=401, body=b"unauthorized"))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(aiohttp, "ClientSession", lambda *a, **kw: _UnauthorizedSession())

    with pytest.raises(ModelListError, match="401"):
        asyncio.run(OllamaClient("https://ollama.com", "m", api_key="bad").list_models_detailed())
    assert attempts["n"] == 1


# ---------------------------------------------------------------------------
# HTTP surface. /chat-api/ is proxied without authentication, so these assert
# both the gate and that the key is never disclosed.
# ---------------------------------------------------------------------------

def _client(tmp_path, monkeypatch, **env):
    """Build a TestClient with fresh singletons and the given cloud env."""
    monkeypatch.setenv("IMAGE_FOLDER", str(tmp_path / "img"))
    monkeypatch.setenv("MAX_CACHE_ENTRIES", "5")
    _reset_config(monkeypatch, **env)
    import runtime_config; runtime_config._runtime_config = None
    import session_manager; session_manager._session_manager = None
    import image_cache; image_cache._image_cache = None
    import ollama_client; ollama_client._ollama_client = None

    import sys
    for m in ("app", "debug_routes", "websocket_handler"):
        sys.modules.pop(m, None)
    import app
    from fastapi.testclient import TestClient
    return TestClient(app.app)


def test_config_reports_cloud_disabled_and_never_returns_key(tmp_path, monkeypatch):
    """The key must not appear anywhere in the config payload."""
    client = _client(tmp_path, monkeypatch, ALLOW_CLOUD_BACKEND="1", OLLAMA_API_KEY="sk-super-secret")
    body = client.get("/debug/config")
    assert body.status_code == 200
    data = body.json()

    assert data["cloud_enabled"] is True
    assert data["cloud_configured"] is True  # presence only
    assert data["provider"] == "local"
    assert "sk-super-secret" not in json.dumps(data)


def test_config_reports_cloud_unconfigured_when_no_key(tmp_path, monkeypatch):
    data = _client(tmp_path, monkeypatch, ALLOW_CLOUD_BACKEND="1").get("/debug/config").json()
    assert data["cloud_enabled"] is True
    assert data["cloud_configured"] is False


def test_switch_to_cloud_forbidden_when_gate_off(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, OLLAMA_API_KEY="sk-1")
    r = client.put("/debug/config", json={"provider": "cloud", "cloud_model": "qwen3.5"})
    assert r.status_code == 403
    assert "disabled" in r.json()["detail"]
    # and the service stayed on local
    assert client.get("/debug/config").json()["provider"] == "local"


def test_switch_to_cloud_rejected_without_key(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, ALLOW_CLOUD_BACKEND="1")
    r = client.put("/debug/config", json={"provider": "cloud", "cloud_model": "qwen3.5"})
    assert r.status_code == 400
    assert "API key" in r.json()["detail"]
    assert client.get("/debug/config").json()["provider"] == "local"


def test_switch_to_cloud_rejected_without_model(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, ALLOW_CLOUD_BACKEND="1", OLLAMA_API_KEY="sk-1")
    r = client.put("/debug/config", json={"provider": "cloud"})
    assert r.status_code == 400
    assert "model" in r.json()["detail"].lower()
    assert client.get("/debug/config").json()["provider"] == "local"


def test_switch_to_cloud_succeeds_when_fully_configured(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, ALLOW_CLOUD_BACKEND="1", OLLAMA_API_KEY="sk-1")
    r = client.put("/debug/config", json={"provider": "cloud", "cloud_model": "qwen3.5"})
    assert r.status_code == 200
    data = r.json()
    assert data["provider"] == "cloud"
    assert data["cloud_model"] == "qwen3.5"
    assert data["active_model"] == "qwen3.5"
    assert "sk-1" not in json.dumps(data)


def test_rejected_switch_leaves_local_model_untouched(tmp_path, monkeypatch):
    """A 403 must not partially apply the rest of the payload."""
    client = _client(tmp_path, monkeypatch)
    before = client.get("/debug/config").json()["model"]
    r = client.put(
        "/debug/config",
        json={"provider": "cloud", "cloud_model": "qwen3.5", "model": "should-not-apply:1b"},
    )
    assert r.status_code == 403
    assert client.get("/debug/config").json()["model"] == before


def test_cloud_model_listing_forbidden_when_disabled(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    r = client.get("/debug/cloud/models")
    assert r.status_code == 403
    assert "disabled" in r.json()["detail"]


def test_cloud_model_listing_returns_vision_flags(tmp_path, monkeypatch, patch_session):
    client = _client(tmp_path, monkeypatch, ALLOW_CLOUD_BACKEND="1", OLLAMA_API_KEY="sk-1")
    patch_session({
        ("GET", "/api/tags"): _FakeResponse(json_payload={
            "models": [{"name": "seeing:1b"}, {"name": "blind:1b"}]
        }),
        ("POST", "/api/show"): lambda body: _FakeResponse(json_payload={
            "capabilities": ["completion", "vision"]
            if body["model"] == "seeing:1b"
            else ["completion"]
        }),
    })

    r = client.get("/debug/cloud/models")
    assert r.status_code == 200
    data = r.json()
    assert data["capabilities_reported"] is True
    by_name = {m["name"]: m for m in data["models"]}
    assert by_name["seeing:1b"]["supports_vision"] is True
    assert by_name["blind:1b"]["supports_vision"] is False
    assert "sk-1" not in json.dumps(data)


def test_cloud_model_listing_surfaces_upstream_rejection(tmp_path, monkeypatch, patch_session):
    """A bad key should read as an error, not as 'this account has no models'."""
    client = _client(tmp_path, monkeypatch, ALLOW_CLOUD_BACKEND="1", OLLAMA_API_KEY="bad")
    patch_session({("GET", "/api/tags"): _FakeResponse(status=401, body=b"unauthorized")})

    r = client.get("/debug/cloud/models")
    assert r.status_code == 502
    assert "401" in r.json()["detail"]


# ---------------------------------------------------------------------------
# Local model listing
#
# The panel used to take the local model as free text, so a typo or a model that
# was never pulled failed only when a message was sent, and failed as an opaque
# backend error rather than as "that model is not here".
# ---------------------------------------------------------------------------


def test_local_model_listing_returns_what_the_server_has(tmp_path, monkeypatch, patch_session):
    client = _client(tmp_path, monkeypatch)
    patch_session({
        ("GET", "/api/tags"): _FakeResponse(json_payload={
            "models": [{"name": "medgemma:4b"}, {"name": "llama4:8b"}]
        }),
        ("POST", "/api/show"): lambda body: _FakeResponse(json_payload={
            "capabilities": ["completion", "vision"]
            if body["model"] == "medgemma:4b"
            else ["completion"]
        }),
    })

    r = client.get("/debug/local/models")
    assert r.status_code == 200
    data = r.json()
    by_name = {m["name"]: m for m in data["models"]}
    assert by_name["medgemma:4b"]["supports_vision"] is True
    assert by_name["llama4:8b"]["supports_vision"] is False
    assert data["capabilities_reported"] is True


def test_local_model_listing_needs_no_cloud_gate(tmp_path, monkeypatch, patch_session):
    """The local catalogue is not behind ALLOW_CLOUD_BACKEND — it never left the site."""
    client = _client(tmp_path, monkeypatch)
    patch_session({
        ("GET", "/api/tags"): _FakeResponse(json_payload={"models": []}),
    })
    assert client.get("/debug/cloud/models").status_code == 403
    assert client.get("/debug/local/models").status_code == 200


def test_local_model_listing_reports_an_unreachable_server(tmp_path, monkeypatch, patch_session):
    """An unreachable Ollama and one with nothing pulled need different actions.

    Returning an empty list for the first would report the second.
    """
    client = _client(tmp_path, monkeypatch)
    patch_session({
        ("GET", "/api/tags"): _FakeResponse(status=500, body=b"boom"),
    })

    r = client.get("/debug/local/models")
    assert r.status_code == 502


def test_local_model_listing_distinguishes_empty_from_broken(tmp_path, monkeypatch, patch_session):
    client = _client(tmp_path, monkeypatch)
    patch_session({
        ("GET", "/api/tags"): _FakeResponse(json_payload={"models": []}),
    })

    r = client.get("/debug/local/models")
    assert r.status_code == 200
    assert r.json()["models"] == []



# ---------------------------------------------------------------------------
# OpenRouter as the alternative cloud provider.
#
# Chat is the same OpenAI-compatible POST, so these cover only what actually
# differs: which env vars are read, the catalogue dialect, the routing policy
# sent with every request, and the reasoning field name.
# ---------------------------------------------------------------------------

# One entry that reads images and one that does not, in OpenRouter's shape.
_OPENROUTER_CATALOGUE = {
    "data": [
        {
            "id": "google/gemini-2.5-pro",
            "architecture": {"input_modalities": ["text", "image", "file"]},
        },
        {
            "id": "deepseek/deepseek-r1",
            "architecture": {"input_modalities": ["text"]},
        },
    ]
}


def test_openrouter_provider_reads_its_own_url_key_and_model(monkeypatch):
    cfg = _reset_config(
        monkeypatch,
        CLOUD_PROVIDER="openrouter",
        ALLOW_CLOUD_BACKEND="1",
        OPENROUTER_API_KEY="sk-or-1",
        OPENROUTER_MODEL="google/gemini-2.5-pro",
    )
    assert cfg.cloud_provider == "openrouter"
    assert cfg.cloud_url == "https://openrouter.ai/api"
    assert cfg.cloud_api_key == "sk-or-1"
    assert cfg.cloud_model == "google/gemini-2.5-pro"
    assert cfg.cloud_label == "OpenRouter"
    assert cfg.cloud_key_env == "OPENROUTER_API_KEY"


def test_openrouter_does_not_pick_up_the_ollama_key(monkeypatch):
    """Keys must not cross providers: an Ollama key sent to OpenRouter is a leak
    of one service's credential to another, and would read as 'configured'."""
    cfg = _reset_config(
        monkeypatch, CLOUD_PROVIDER="openrouter", ALLOW_CLOUD_BACKEND="1", OLLAMA_API_KEY="sk-ollama"
    )
    assert cfg.cloud_api_key == ""


def test_unknown_cloud_provider_falls_back_to_ollama(monkeypatch):
    """A typo must not take the service down — the slot is gated anyway."""
    cfg = _reset_config(monkeypatch, CLOUD_PROVIDER="openrouterr")
    assert cfg.cloud_provider == "ollama"


def test_openrouter_client_denies_data_collection_by_default(monkeypatch):
    """OpenRouter brokers to third-party hosts, so the policy travels per request
    rather than resting on whatever the account dashboard happens to say."""
    _reset_config(
        monkeypatch,
        CLOUD_PROVIDER="openrouter",
        ALLOW_CLOUD_BACKEND="1",
        OPENROUTER_API_KEY="sk-or-1",
    )
    from ollama_client import build_cloud_client

    c = build_cloud_client(model="google/gemini-2.5-pro")
    assert c.base_url == "https://openrouter.ai/api"
    assert c.backend_type == "openrouter"
    assert c.extra_payload == {"provider": {"data_collection": "deny"}}


def test_openrouter_data_collection_can_be_relaxed(monkeypatch):
    """Denying excludes some providers, so an operator can opt back out of it."""
    _reset_config(
        monkeypatch,
        CLOUD_PROVIDER="openrouter",
        ALLOW_CLOUD_BACKEND="1",
        OPENROUTER_API_KEY="sk-or-1",
        OPENROUTER_ALLOW_DATA_COLLECTION="1",
    )
    from ollama_client import build_cloud_client

    c = build_cloud_client(model="m")
    assert c.extra_payload == {"provider": {"data_collection": "allow"}}


def test_ollama_cloud_client_sends_no_routing_block(monkeypatch):
    """`provider` is an OpenRouter key; Ollama must not receive it."""
    _reset_config(monkeypatch, ALLOW_CLOUD_BACKEND="1", OLLAMA_API_KEY="sk-1")
    from ollama_client import build_cloud_client

    assert build_cloud_client(model="qwen3.5").extra_payload == {}


def test_openrouter_chat_sends_the_policy_and_the_bearer(patch_session):
    from ollama_client import OllamaClient

    session = patch_session(
        {("POST", "/v1/chat/completions"): _StreamResponse(_sse("hi"))}
    )
    client = OllamaClient(
        "https://openrouter.ai/api",
        "google/gemini-2.5-pro",
        backend_type="openrouter",
        api_key="sk-or-1",
        extra_payload={"provider": {"data_collection": "deny"}},
    )

    chunks = asyncio.run(_drain(client))
    assert [c["text"] for c in chunks] == ["hi"]

    post = next(x for x in session.calls if x["method"] == "POST")
    assert post["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert post["headers"]["Authorization"] == "Bearer sk-or-1"
    assert post["json"]["provider"] == {"data_collection": "deny"}
    assert post["json"]["model"] == "google/gemini-2.5-pro"
    assert post["json"]["stream"] is True


def test_openrouter_listing_reads_vision_from_input_modalities(patch_session):
    """One GET carries the whole catalogue: OpenRouter has no /api/show to ask."""
    from ollama_client import OllamaClient

    session = patch_session(
        {("GET", "/v1/models"): _FakeResponse(json_payload=_OPENROUTER_CATALOGUE)}
    )
    client = OllamaClient(
        "https://openrouter.ai/api", "m", backend_type="openrouter", api_key="sk-or-1"
    )
    models = asyncio.run(client.list_models_detailed())

    by_name = {m["name"]: m for m in models}
    assert by_name["google/gemini-2.5-pro"]["supports_vision"] is True
    assert by_name["deepseek/deepseek-r1"]["supports_vision"] is False
    # Capabilities are reported for every entry, so the panel never has to fall
    # back to "vision unknown" here.
    assert by_name["google/gemini-2.5-pro"]["capabilities"] == ["text", "image", "file"]

    assert [x["url"] for x in session.calls] == ["https://openrouter.ai/api/v1/models"]
    assert session.calls[0]["headers"]["Authorization"] == "Bearer sk-or-1"


def test_openrouter_listing_raises_on_http_error(patch_session):
    """A rejected key must surface as an error, not as 'no models'."""
    from ollama_client import ModelListError, OllamaClient

    patch_session({("GET", "/v1/models"): _FakeResponse(status=401, body=b"unauthorized")})
    client = OllamaClient(
        "https://openrouter.ai/api", "m", backend_type="openrouter", api_key="bad"
    )
    with pytest.raises(ModelListError, match="401"):
        asyncio.run(client.list_models_detailed())


def test_model_filter_prunes_the_openrouter_catalogue(patch_session):
    """Several hundred models is not a menu; an operator can narrow it."""
    from ollama_client import OllamaClient

    patch_session({("GET", "/v1/models"): _FakeResponse(json_payload=_OPENROUTER_CATALOGUE)})
    client = OllamaClient(
        "https://openrouter.ai/api",
        "m",
        backend_type="openrouter",
        model_filter=("GEMINI",),  # matched case-insensitively
    )
    models = asyncio.run(client.list_models_detailed())
    assert [m["name"] for m in models] == ["google/gemini-2.5-pro"]


def test_model_filter_skips_the_api_show_fan_out(patch_session):
    """Filtering before the fan-out, so a pruned model costs no request."""
    from ollama_client import OllamaClient

    session = patch_session({
        ("GET", "/api/tags"): _FakeResponse(json_payload={
            "models": [{"name": "keep:1b"}, {"name": "drop:1b"}]
        }),
        ("POST", "/api/show"): _FakeResponse(json_payload={"capabilities": ["vision"]}),
    })
    client = OllamaClient("https://ollama.com", "m", model_filter=("keep",))
    models = asyncio.run(client.list_models_detailed())

    assert [m["name"] for m in models] == ["keep:1b"]
    shows = [x for x in session.calls if x["method"] == "POST"]
    assert [x["json"]["model"] for x in shows] == ["keep:1b"]


def test_reasoning_is_read_in_every_backend_dialect():
    """Ollama says reasoning_content, OpenRouter says reasoning (or a details
    array). Reading only the first left OpenRouter's thinking pane empty."""
    from ollama_client import _reasoning_text

    assert _reasoning_text({"reasoning_content": "ollama"}) == "ollama"
    assert _reasoning_text({"reasoning": "openrouter"}) == "openrouter"
    assert (
        _reasoning_text(
            {"reasoning_details": [
                {"type": "reasoning.text", "text": "step "},
                {"type": "reasoning.text", "text": "two"},
            ]}
        )
        == "step two"
    )
    # No reasoning at all, and a shape that carries no text, both read as absent.
    assert _reasoning_text({"content": "answer"}) == ""
    assert _reasoning_text({"reasoning_details": [{"type": "reasoning.encrypted"}]}) == ""


def test_openrouter_reasoning_streams_as_thinking_tokens(patch_session):
    from ollama_client import OllamaClient

    patch_session({
        ("POST", "/v1/chat/completions"): _StreamResponse([
            b'data: {"choices":[{"delta":{"reasoning":"weighing it"}}]}',
            b'data: {"choices":[{"delta":{"content":"answer"}}]}',
            b"data: [DONE]",
        ])
    })
    client = OllamaClient(
        "https://openrouter.ai/api", "m", backend_type="openrouter", api_key="sk-or-1"
    )
    chunks = asyncio.run(_drain(client))
    assert chunks == [
        {"type": "thinking", "text": "weighing it"},
        {"type": "content", "text": "answer"},
    ]


def test_openrouter_health_check_uses_the_catalogue(patch_session):
    """There is no /api/tags and no health route; /v1/models answering is proof."""
    from ollama_client import OllamaClient

    session = patch_session({("GET", "/v1/models"): _FakeResponse(json_payload={"data": []})})
    client = OllamaClient("https://openrouter.ai/api", "m", backend_type="openrouter")
    assert asyncio.run(client.health_check()) is True
    assert session.calls[0]["url"] == "https://openrouter.ai/api/v1/models"


# ---------------------------------------------------------------------------
# HTTP surface with OpenRouter configured
# ---------------------------------------------------------------------------

def test_config_names_the_configured_provider(tmp_path, monkeypatch):
    """The panel names the env var from this, and an operator reading the
    endpoint learns which service the cloud slot points at."""
    client = _client(
        tmp_path,
        monkeypatch,
        CLOUD_PROVIDER="openrouter",
        ALLOW_CLOUD_BACKEND="1",
        OPENROUTER_API_KEY="sk-or-secret",
    )
    data = client.get("/debug/config").json()

    assert data["cloud_provider"] == "openrouter"
    assert data["cloud_key_env"] == "OPENROUTER_API_KEY"
    assert data["cloud_url"] == "https://openrouter.ai/api"
    assert data["cloud_configured"] is True
    assert "sk-or-secret" not in json.dumps(data)


def test_refusal_names_the_env_var_the_operator_must_set(tmp_path, monkeypatch):
    """Telling an OpenRouter deployment to set OLLAMA_API_KEY sends the operator
    to the wrong dashboard."""
    client = _client(
        tmp_path, monkeypatch, CLOUD_PROVIDER="openrouter", ALLOW_CLOUD_BACKEND="1"
    )
    r = client.put("/debug/config", json={"provider": "cloud", "cloud_model": "m"})
    assert r.status_code == 400
    assert "OPENROUTER_API_KEY" in r.json()["detail"]
    assert "OpenRouter" in r.json()["detail"]


def test_disabled_gate_message_names_the_provider(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, CLOUD_PROVIDER="openrouter")
    r = client.get("/debug/cloud/models")
    assert r.status_code == 403
    assert "OpenRouter" in r.json()["detail"]
    assert "disabled" in r.json()["detail"]


def test_cloud_model_listing_works_against_openrouter(tmp_path, monkeypatch, patch_session):
    client = _client(
        tmp_path,
        monkeypatch,
        CLOUD_PROVIDER="openrouter",
        ALLOW_CLOUD_BACKEND="1",
        OPENROUTER_API_KEY="sk-or-1",
    )
    patch_session({("GET", "/v1/models"): _FakeResponse(json_payload=_OPENROUTER_CATALOGUE)})

    r = client.get("/debug/cloud/models")
    assert r.status_code == 200
    data = r.json()
    assert data["capabilities_reported"] is True
    by_name = {m["name"]: m for m in data["models"]}
    assert by_name["google/gemini-2.5-pro"]["supports_vision"] is True
    assert by_name["deepseek/deepseek-r1"]["supports_vision"] is False
    assert "sk-or-1" not in json.dumps(data)


# ---------------------------------------------------------------------------
# What bounds how many slices a model can be shown.
#
# The panel used to cap this at a hard-coded 50. The real bound is a property of
# the model, and Ollama reports the two numbers it is made of.
# ---------------------------------------------------------------------------

def test_image_budget_is_read_from_the_model_not_assumed():
    """Verified against a live /api/show for medgemma 1.5 4B (gemma3)."""
    from ollama_client import _image_budget

    context, per_image = _image_budget({
        "gemma3.context_length": 131072,
        "gemma3.mm.tokens_per_image": 256,
        "gemma3.vision.image_size": 896,
        "gemma3.vision.patch_size": 14,
    })
    assert (context, per_image) == (131072, 256)
    # ~500 images, which is the point: the honest bound is nothing like 50.
    assert context // per_image > 400


def test_image_budget_prefers_the_model_context_over_the_encoder_one():
    """`gemma3.vision.*` describes the image encoder, not the model's context."""
    from ollama_client import _image_budget

    context, _ = _image_budget({
        "gemma3.vision.context_length": 4096,
        "gemma3.context_length": 131072,
    })
    assert context == 131072


def test_image_budget_reports_nothing_when_the_model_says_nothing():
    """None is not zero: it means 'unreported', and the panel says so."""
    from ollama_client import _image_budget

    assert _image_budget({}) == (None, None)
    assert _image_budget({"llama.context_length": 8192}) == (8192, None)
    # A non-numeric value is not a budget.
    assert _image_budget({"x.context_length": "many"}) == (None, None)


def test_ollama_listing_carries_the_budget(patch_session):
    from ollama_client import OllamaClient

    patch_session({
        ("GET", "/api/tags"): _FakeResponse(json_payload={"models": [{"name": "medgemma:4b"}]}),
        ("POST", "/api/show"): _FakeResponse(json_payload={
            "capabilities": ["completion", "vision"],
            "model_info": {"gemma3.context_length": 131072, "gemma3.mm.tokens_per_image": 256},
        }),
    })
    models = asyncio.run(OllamaClient("https://ollama.com", "m").list_models_detailed())
    assert models[0]["context_length"] == 131072
    assert models[0]["tokens_per_image"] == 256


def test_openrouter_listing_carries_context_but_no_per_image_cost(patch_session):
    """OpenRouter publishes context_length and nothing about images: per_request_limits
    is null for all 430 models, and the real per-image cost varies by which provider
    the request is brokered to. Guessing here would present an estimate as fact."""
    from ollama_client import OllamaClient

    patch_session({
        ("GET", "/v1/models"): _FakeResponse(json_payload={
            "data": [
                {
                    "id": "google/gemini-3.8-flash",
                    "context_length": 1048576,
                    "architecture": {"input_modalities": ["text", "image"]},
                }
            ]
        })
    })
    client = OllamaClient("https://openrouter.ai/api", "m", backend_type="openrouter")
    models = asyncio.run(client.list_models_detailed())
    assert models[0]["context_length"] == 1048576
    assert models[0]["tokens_per_image"] is None
