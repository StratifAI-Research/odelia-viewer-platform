"""Tests for chat-middleware/app.py + debug_routes.py — FastAPI service surface via TestClient."""
import pytest


# ---------------------------------------------------------------------------
# Helper to build a fresh app with all singletons reset (so tests don\'t pollute one another).
# ---------------------------------------------------------------------------

def _fresh_app(tmp_path, monkeypatch, *, fake_ollama=None):
    """Reset all chat-middleware singletons, set env, import app fresh, return (app, client).

    fake_ollama: if provided, replaces get_ollama_client() result globally.
    """
    monkeypatch.setenv("IMAGE_FOLDER", str(tmp_path / "app-img"))
    monkeypatch.setenv("MAX_CACHE_ENTRIES", "5")
    import config; config.config = None
    import runtime_config; runtime_config._runtime_config = None
    import session_manager; session_manager._session_manager = None
    import image_cache; image_cache._image_cache = None
    import ollama_client
    ollama_client._ollama_client = None
    if fake_ollama is not None:
        monkeypatch.setattr(ollama_client, "get_ollama_client", lambda *a, **kw: fake_ollama)

    # Import (or re-import) app last.
    import sys
    sys.modules.pop("app", None)
    sys.modules.pop("debug_routes", None)
    sys.modules.pop("websocket_handler", None)
    import app
    from fastapi.testclient import TestClient
    return app, TestClient(app.app)


class _FakeOllama:
    """Minimal async fake for tests that touch /debug/health or the lifespan."""
    def __init__(self, healthy=True, models=None):
        self._healthy = healthy
        self._models = models if models is not None else ["m1", "m2"]
    async def health_check(self):
        return self._healthy
    async def list_models(self):
        return list(self._models)


# ---------- top-level /health ----------

def test_basic_health_endpoint_returns_status_healthy(tmp_path, monkeypatch):
    _app, client = _fresh_app(tmp_path, monkeypatch)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "healthy"}


# ---------- lifespan startup ----------

def test_app_lifespan_runs_startup_and_shutdown(tmp_path, monkeypatch):
    """Use `with TestClient(app)` to trigger the lifespan context.
    Verifies the startup code path (init_config + ollama health_check + list_models)."""
    fake = _FakeOllama(healthy=True, models=["mA"])
    _app, _client = _fresh_app(tmp_path, monkeypatch, fake_ollama=fake)
    from fastapi.testclient import TestClient
    with TestClient(_app.app) as live_client:
        r = live_client.get("/health")
        assert r.status_code == 200


def test_app_lifespan_logs_warning_when_ollama_unhealthy(tmp_path, monkeypatch, caplog):
    """Unhealthy backend logs a warning but startup still completes."""
    fake = _FakeOllama(healthy=False)
    _app, _client = _fresh_app(tmp_path, monkeypatch, fake_ollama=fake)
    from fastapi.testclient import TestClient
    with caplog.at_level("WARNING"):
        with TestClient(_app.app) as live_client:
            live_client.get("/health")
    assert any("Ollama connection: FAILED" in r.message for r in caplog.records)


# ---------- debug /config GET ----------

def test_debug_config_get_returns_current_config(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_URL", "http://my-ollama:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "MyModel:8b")
    monkeypatch.setenv("NUM_SLICES", "7")
    _app, client = _fresh_app(tmp_path, monkeypatch)
    r = client.get("/debug/config")
    assert r.status_code == 200
    payload = r.json()
    assert payload["model"] == "MyModel:8b"
    assert payload["preprocessing"]["num_slices"] == 7
    assert payload["ollama_static"]["url"] == "http://my-ollama:11434"
    assert "ollama_options" in payload


# ---------- debug /config PUT ----------

def test_debug_config_put_updates_system_prompt_and_model(tmp_path, monkeypatch):
    _app, client = _fresh_app(tmp_path, monkeypatch)
    r = client.put("/debug/config", json={
        "system_prompt": "be terse", "model": "Updated:13b",
    })
    assert r.status_code == 200
    payload = r.json()
    assert payload["system_prompt"] == "be terse"
    assert payload["model"] == "Updated:13b"


def test_debug_config_put_partial_preprocessing_persists(tmp_path, monkeypatch):
    """PUT with only some preprocessing fields preserves untouched ones."""
    _app, client = _fresh_app(tmp_path, monkeypatch)
    r = client.put("/debug/config", json={
        "preprocessing": {"num_slices": 11, "slice_strategy": "uniform"},
    })
    assert r.status_code == 200
    payload = r.json()
    assert payload["preprocessing"]["num_slices"] == 11
    assert payload["preprocessing"]["slice_strategy"] == "uniform"


def test_debug_config_put_clears_image_cache_when_preprocessing_changes(tmp_path, monkeypatch):
    """Auto-clear is a documented behavior — pin it."""
    _app, client = _fresh_app(tmp_path, monkeypatch)
    from image_cache import get_image_cache, CachedSeries
    cache = get_image_cache()
    cache.put("SE1", CachedSeries(series_uid="SE1", base64_images=["x"]))
    assert cache.size() == 1
    r = client.put("/debug/config", json={"preprocessing": {"num_slices": 3}})
    assert r.status_code == 200
    assert cache.size() == 0          # auto-cleared


def test_debug_config_put_does_not_clear_cache_for_non_preprocessing_updates(tmp_path, monkeypatch):
    _app, client = _fresh_app(tmp_path, monkeypatch)
    from image_cache import get_image_cache, CachedSeries
    cache = get_image_cache()
    cache.put("SE1", CachedSeries(series_uid="SE1", base64_images=["x"]))
    client.put("/debug/config", json={"system_prompt": "different"})
    assert cache.size() == 1          # preserved


def test_debug_config_put_updates_ollama_options(tmp_path, monkeypatch):
    _app, client = _fresh_app(tmp_path, monkeypatch)
    r = client.put("/debug/config", json={
        "ollama_options": {"max_tokens": 256, "temperature": 0.3},
    })
    payload = r.json()
    assert payload["ollama_options"]["max_tokens"] == 256
    assert payload["ollama_options"]["temperature"] == 0.3


def test_debug_config_put_invalid_slice_strategy_returns_422(tmp_path, monkeypatch):
    """M8: PUT with an unknown slice_strategy enum value -> 422 (pydantic validation).
    Pinning this prevents a future loosening of the enum validator from silently accepting
    invalid strategies (which would corrupt downstream slice extraction)."""
    _app, client = _fresh_app(tmp_path, monkeypatch)
    r = client.put("/debug/config", json={
        "preprocessing": {"slice_strategy": "not_a_real_strategy"},
    })
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert any("slice_strategy" in str(d).lower() for d in detail)


# ---------- /debug/cache ----------

def test_debug_cache_delete_clears_and_reports_count(tmp_path, monkeypatch):
    _app, client = _fresh_app(tmp_path, monkeypatch)
    from image_cache import get_image_cache, CachedSeries
    cache = get_image_cache()
    cache.put("a", CachedSeries(series_uid="a", base64_images=["x"]))
    cache.put("b", CachedSeries(series_uid="b", base64_images=["y"]))
    r = client.delete("/debug/cache")
    assert r.status_code == 200
    assert r.json() == {"cleared_entries": 2, "message": "Cleared 2 cached series"}
    assert cache.size() == 0


def test_debug_cache_stats_endpoint(tmp_path, monkeypatch):
    _app, client = _fresh_app(tmp_path, monkeypatch)
    from image_cache import get_image_cache, CachedSeries
    cache = get_image_cache()
    cache.put("uid.x", CachedSeries(series_uid="uid.x", base64_images=["..."]))
    r = client.get("/debug/cache/stats")
    assert r.status_code == 200
    payload = r.json()
    assert payload["size"] == 1
    assert "uid.x" in payload["series_uids"]


# ---------- /debug/sessions ----------

def test_debug_sessions_list_returns_active_sessions(tmp_path, monkeypatch):
    _app, client = _fresh_app(tmp_path, monkeypatch)
    from session_manager import get_session_manager
    sm = get_session_manager()
    sm.create_session("sess-1")
    sm.create_session("sess-2")
    r = client.get("/debug/sessions")
    assert r.status_code == 200
    ids = sorted(s["session_id"] for s in r.json()["sessions"])
    assert ids == ["sess-1", "sess-2"]


def test_debug_sessions_delete_existing(tmp_path, monkeypatch):
    _app, client = _fresh_app(tmp_path, monkeypatch)
    from session_manager import get_session_manager
    sm = get_session_manager()
    sm.create_session("doomed")
    r = client.delete("/debug/sessions/doomed")
    assert r.status_code == 200
    assert "doomed" in r.json()["message"]
    assert sm.get_session("doomed") is None


def test_debug_sessions_delete_missing_returns_404(tmp_path, monkeypatch):
    _app, client = _fresh_app(tmp_path, monkeypatch)
    r = client.delete("/debug/sessions/never-existed")
    assert r.status_code == 404
    assert "not found" in r.json()["detail"]


def test_debug_sessions_cleanup_removes_only_stale(tmp_path, monkeypatch):
    from datetime import datetime, timedelta
    _app, client = _fresh_app(tmp_path, monkeypatch)
    from session_manager import get_session_manager
    sm = get_session_manager()
    sm.create_session("fresh")
    stale = sm.create_session("stale")
    stale.last_activity = datetime.now() - timedelta(minutes=180)
    r = client.post("/debug/sessions/cleanup?max_age_minutes=60")
    assert r.status_code == 200
    assert r.json() == {"removed_sessions": 1, "max_age_minutes": 60}
    assert sm.get_session("fresh") is not None
    assert sm.get_session("stale") is None


# ---------- /debug/health ----------

def test_debug_health_returns_healthy_when_ollama_ok(tmp_path, monkeypatch):
    fake = _FakeOllama(healthy=True, models=["mA", "mB"])
    _app, client = _fresh_app(tmp_path, monkeypatch, fake_ollama=fake)
    r = client.get("/debug/health")
    assert r.status_code == 200
    payload = r.json()
    assert payload["status"] == "healthy"
    assert payload["ollama"]["connected"] is True
    assert payload["ollama"]["available_models"] == ["mA", "mB"]
    assert "cache" in payload
    assert "sessions" in payload


def test_debug_health_returns_degraded_when_ollama_down(tmp_path, monkeypatch):
    fake = _FakeOllama(healthy=False, models=[])
    _app, client = _fresh_app(tmp_path, monkeypatch, fake_ollama=fake)
    r = client.get("/debug/health")
    assert r.status_code == 200
    payload = r.json()
    assert payload["status"] == "degraded"
    assert payload["ollama"]["connected"] is False
    assert payload["ollama"]["available_models"] == []
