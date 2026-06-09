"""Tests for chat-middleware/runtime_config.py — mutable runtime configuration."""


def _fresh_config(tmp_path, monkeypatch, **env):
    """Wipe singletons + set env so RuntimeConfig() picks up our values."""
    monkeypatch.setenv("IMAGE_FOLDER", str(tmp_path / "rt-cfg-img"))
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    import config
    config.config = None
    import runtime_config
    runtime_config._runtime_config = None
    return runtime_config


def test_runtime_config_defaults_pull_from_static_config(tmp_path, monkeypatch):
    rc_mod = _fresh_config(tmp_path, monkeypatch, OLLAMA_MODEL="MedModel:7b", NUM_SLICES=7)
    rc = rc_mod.RuntimeConfig()
    assert rc.model == "MedModel:7b"             # from OLLAMA_MODEL
    assert rc.preprocessing.num_slices == 7      # from NUM_SLICES
    assert rc.system_prompt == rc_mod.DEFAULT_SYSTEM_PROMPT


def test_update_system_prompt_only_changes_that_field(tmp_path, monkeypatch):
    rc_mod = _fresh_config(tmp_path, monkeypatch)
    rc = rc_mod.RuntimeConfig()
    old_model = rc.model
    rc.update(system_prompt="be very terse")
    assert rc.system_prompt == "be very terse"
    assert rc.model == old_model                 # untouched


def test_update_model(tmp_path, monkeypatch):
    rc_mod = _fresh_config(tmp_path, monkeypatch)
    rc = rc_mod.RuntimeConfig()
    rc.update(model="NewModel:13b")
    assert rc.model == "NewModel:13b"


def test_update_preprocessing_num_slices_only(tmp_path, monkeypatch):
    rc_mod = _fresh_config(tmp_path, monkeypatch, NUM_SLICES=5)
    rc = rc_mod.RuntimeConfig()
    rc.update(preprocessing={"num_slices": 12})
    assert rc.preprocessing.num_slices == 12


def test_update_preprocessing_slice_strategy_from_string(tmp_path, monkeypatch):
    rc_mod = _fresh_config(tmp_path, monkeypatch)
    rc = rc_mod.RuntimeConfig()
    rc.update(preprocessing={"slice_strategy": "uniform"})
    from models import SliceStrategy
    assert rc.preprocessing.slice_strategy == SliceStrategy.UNIFORM


def test_update_preprocessing_slice_strategy_from_enum(tmp_path, monkeypatch):
    rc_mod = _fresh_config(tmp_path, monkeypatch)
    rc = rc_mod.RuntimeConfig()
    from models import SliceStrategy
    rc.update(preprocessing={"slice_strategy": SliceStrategy.FIRST_N})
    assert rc.preprocessing.slice_strategy == SliceStrategy.FIRST_N


def test_update_preprocessing_central_percentage(tmp_path, monkeypatch):
    rc_mod = _fresh_config(tmp_path, monkeypatch)
    rc = rc_mod.RuntimeConfig()
    rc.update(preprocessing={"central_percentage": 80})
    assert rc.preprocessing.central_percentage == 80


def test_update_preprocessing_ignores_none_values(tmp_path, monkeypatch):
    """The update() impl explicitly skips fields whose value is None."""
    rc_mod = _fresh_config(tmp_path, monkeypatch)
    rc = rc_mod.RuntimeConfig()
    before = rc.preprocessing.num_slices
    rc.update(preprocessing={"num_slices": None, "central_percentage": 75})
    assert rc.preprocessing.num_slices == before
    assert rc.preprocessing.central_percentage == 75


def test_update_ollama_options_known_keys(tmp_path, monkeypatch):
    rc_mod = _fresh_config(tmp_path, monkeypatch)
    rc = rc_mod.RuntimeConfig()
    rc.update(ollama_options={"max_tokens": 1024, "temperature": 0.3, "top_p": 0.9})
    assert rc.ollama_options.max_tokens == 1024
    assert rc.ollama_options.temperature == 0.3
    assert rc.ollama_options.top_p == 0.9


def test_update_ollama_options_ignores_unknown_keys(tmp_path, monkeypatch):
    """Unknown keys are skipped via hasattr(self.ollama_options, key) guard."""
    rc_mod = _fresh_config(tmp_path, monkeypatch)
    rc = rc_mod.RuntimeConfig()
    rc.update(ollama_options={"frequency_penalty": 1.0})   # not on OllamaOptions
    # No attribute added; update succeeded silently.
    assert not hasattr(rc.ollama_options, "frequency_penalty")


def test_ollama_options_to_dict_excludes_none_values():
    from runtime_config import OllamaOptions
    opts = OllamaOptions(max_tokens=512, temperature=0.5)
    d = opts.to_dict()
    assert d == {"max_tokens": 512, "temperature": 0.5}    # None values dropped


def test_ollama_options_to_full_dict_includes_none_values():
    from runtime_config import OllamaOptions
    opts = OllamaOptions(max_tokens=512)
    d = opts.to_full_dict()
    assert d["max_tokens"] == 512
    assert d["temperature"] is None
    assert d["seed"] is None


def test_to_dict_returns_full_state(tmp_path, monkeypatch):
    rc_mod = _fresh_config(tmp_path, monkeypatch)
    rc = rc_mod.RuntimeConfig()
    rc.update(system_prompt="x", model="m", preprocessing={"num_slices": 9})
    d = rc.to_dict()
    assert d["system_prompt"] == "x"
    assert d["model"] == "m"
    assert d["preprocessing"]["num_slices"] == 9
    assert "slice_strategy" in d["preprocessing"]
    assert "ollama_options" in d


def test_get_runtime_config_singleton(tmp_path, monkeypatch):
    rc_mod = _fresh_config(tmp_path, monkeypatch)
    a = rc_mod.get_runtime_config()
    b = rc_mod.get_runtime_config()
    assert a is b


def test_reset_runtime_config_replaces_singleton(tmp_path, monkeypatch):
    rc_mod = _fresh_config(tmp_path, monkeypatch)
    a = rc_mod.get_runtime_config()
    b = rc_mod.reset_runtime_config()
    assert a is not b
