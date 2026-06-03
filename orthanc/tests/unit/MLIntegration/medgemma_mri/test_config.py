"""Tests for medgemma-mri/config.py — MedGemmaConfig dataclass.

config.py imports torch at module level, so we inject torch_stub before loading.
"""
import sys
import dataclasses


def test_medgemma_config_fields_present(torch_stub):
    sys.modules.pop('config', None)
    from config import MedGemmaConfig
    field_names = {f.name for f in dataclasses.fields(MedGemmaConfig)}
    assert 'model_id' in field_names
    assert 'hf_token' in field_names
    assert 'device' in field_names
    assert 'torch_dtype' in field_names
    assert 'max_new_tokens' in field_names
    assert 'num_slices' in field_names


def test_medgemma_config_from_env_default_model_id(torch_stub, monkeypatch):
    sys.modules.pop('config', None)
    monkeypatch.delenv('MODEL_ID', raising=False)
    from config import MedGemmaConfig
    cfg = MedGemmaConfig.from_env()
    assert cfg.model_id == 'google/medgemma-1.5-4b-it'


def test_medgemma_config_from_env_custom_model_id(torch_stub, monkeypatch):
    sys.modules.pop('config', None)
    monkeypatch.setenv('MODEL_ID', 'google/medgemma-custom')
    from config import MedGemmaConfig
    cfg = MedGemmaConfig.from_env()
    assert cfg.model_id == 'google/medgemma-custom'


def test_medgemma_config_from_env_hf_token_from_env(torch_stub, monkeypatch):
    sys.modules.pop('config', None)
    monkeypatch.setenv('HF_TOKEN', 'hf_abc123')
    from config import MedGemmaConfig
    cfg = MedGemmaConfig.from_env()
    assert cfg.hf_token == 'hf_abc123'


def test_medgemma_config_from_env_hf_token_none_when_unset(torch_stub, monkeypatch):
    sys.modules.pop('config', None)
    monkeypatch.delenv('HF_TOKEN', raising=False)
    from config import MedGemmaConfig
    cfg = MedGemmaConfig.from_env()
    assert cfg.hf_token is None


def test_medgemma_config_from_env_default_torch_dtype(torch_stub, monkeypatch):
    sys.modules.pop('config', None)
    monkeypatch.delenv('TORCH_DTYPE', raising=False)
    from config import MedGemmaConfig
    cfg = MedGemmaConfig.from_env()
    assert cfg.torch_dtype == 'bfloat16'


def test_medgemma_config_from_env_max_new_tokens_default(torch_stub, monkeypatch):
    sys.modules.pop('config', None)
    monkeypatch.delenv('MAX_NEW_TOKENS', raising=False)
    from config import MedGemmaConfig
    cfg = MedGemmaConfig.from_env()
    assert cfg.max_new_tokens == 500


def test_medgemma_config_from_env_num_slices_default(torch_stub, monkeypatch):
    sys.modules.pop('config', None)
    monkeypatch.delenv('NUM_SLICES', raising=False)
    from config import MedGemmaConfig
    cfg = MedGemmaConfig.from_env()
    assert cfg.num_slices == 5


def test_medgemma_config_from_env_cpu_when_cuda_unavailable(torch_stub):
    torch_stub.cuda.is_available.return_value = False
    sys.modules.pop('config', None)
    from config import MedGemmaConfig
    cfg = MedGemmaConfig.from_env()
    assert cfg.device == 'cpu'


def test_medgemma_config_from_env_returns_instance(torch_stub):
    sys.modules.pop('config', None)
    from config import MedGemmaConfig
    cfg = MedGemmaConfig.from_env()
    assert isinstance(cfg, MedGemmaConfig)
