"""Tests for chat-middleware/config.py — ChatMiddlewareConfig dataclass + env-var loader."""
from pathlib import Path

import pytest


def test_chat_middleware_config_defaults_present():
    """from_env() with no env vars uses the documented defaults."""
    import config
    cfg = config.ChatMiddlewareConfig()
    assert cfg.ollama_url == "http://host.docker.internal:11434"
    assert cfg.backend_type == "ollama"
    assert cfg.num_slices == 5
    assert cfg.port == 5560
    assert isinstance(cfg.image_folder, Path)


def test_from_env_reads_all_overrides(monkeypatch):
    """Every env var maps to the right field."""
    monkeypatch.setenv("OLLAMA_URL", "http://custom:1234")
    monkeypatch.setenv("OLLAMA_MODEL", "MyModel:7b")
    monkeypatch.setenv("BACKEND_TYPE", "LLAMACPP")           # tested: lowered
    monkeypatch.setenv("WADO_BASE_URL", "http://other/dicom-web")
    monkeypatch.setenv("NUM_SLICES", "11")
    monkeypatch.setenv("IMAGE_FOLDER", "/tmp/custom-images")
    monkeypatch.setenv("MAX_CACHE_ENTRIES", "55")
    monkeypatch.setenv("HOST", "127.0.0.1")
    monkeypatch.setenv("PORT", "6000")

    import config
    cfg = config.ChatMiddlewareConfig.from_env()
    assert cfg.ollama_url == "http://custom:1234"
    assert cfg.ollama_model == "MyModel:7b"
    assert cfg.backend_type == "llamacpp"                    # .lower() applied
    assert cfg.wado_base_url == "http://other/dicom-web"
    assert cfg.num_slices == 11
    assert cfg.image_folder == Path("/tmp/custom-images")
    assert cfg.max_cache_entries == 55
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 6000


def test_init_config_creates_image_folder(tmp_path, monkeypatch):
    """init_config side effect: mkdir(parents=True, exist_ok=True) on image_folder."""
    target = tmp_path / "fresh-folder" / "deeper"
    monkeypatch.setenv("IMAGE_FOLDER", str(target))
    # Set all other required env vars to avoid side effects
    monkeypatch.setenv("NUM_SLICES", "5")
    monkeypatch.setenv("MAX_CACHE_ENTRIES", "100")
    monkeypatch.setenv("PORT", "5560")

    import config
    config.config = None     # clear global so init_config runs fresh
    cfg = config.init_config()
    assert target.is_dir()
    assert cfg.image_folder == target


def test_get_config_initializes_if_unset(tmp_path, monkeypatch):
    """get_config() with config=None calls init_config()."""
    monkeypatch.setenv("IMAGE_FOLDER", str(tmp_path / "auto-init"))
    import config
    config.config = None
    cfg = config.get_config()
    assert cfg is not None
    assert config.config is cfg     # singleton stored


def test_get_config_returns_existing_singleton(tmp_path, monkeypatch):
    """get_config() with config set returns the existing instance (no re-init)."""
    monkeypatch.setenv("IMAGE_FOLDER", str(tmp_path / "singleton"))
    import config
    first = config.init_config()
    second = config.get_config()
    assert first is second
