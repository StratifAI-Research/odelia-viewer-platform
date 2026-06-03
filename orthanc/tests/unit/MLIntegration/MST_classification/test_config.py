"""Tests for MST-classification/config.py — MSTConfig dataclass.

config.py imports torch at module level, so we inject a mock before loading.
torch_stub fixture is provided by tests/unit/MLIntegration/conftest.py.
"""
import sys
from pathlib import Path



def test_mst_config_fields_present(torch_stub, monkeypatch):
    # Evict cached config module to force re-import with stub
    sys.modules.pop('config', None)
    from config import MSTConfig
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(MSTConfig)}
    assert 'model_path' in field_names
    assert 'hf_token' in field_names
    assert 'device' in field_names


def test_mst_config_model_path_is_path(torch_stub, monkeypatch):
    sys.modules.pop('config', None)
    from config import MSTConfig
    cfg = MSTConfig(model_path=Path('./model'), hf_token=None, device='cpu')
    assert isinstance(cfg.model_path, Path)


def test_mst_config_device_field(torch_stub, monkeypatch):
    sys.modules.pop('config', None)
    from config import MSTConfig
    cfg = MSTConfig(model_path=Path('./model'), hf_token=None, device='cpu')
    assert cfg.device == 'cpu'


def test_mst_config_from_env_uses_model_path_env(torch_stub, monkeypatch):
    sys.modules.pop('config', None)
    monkeypatch.setenv('MODEL_PATH', '/custom/model')
    from config import MSTConfig
    cfg = MSTConfig.from_env()
    assert cfg.model_path == Path('/custom/model')


def test_mst_config_from_env_defaults(torch_stub, monkeypatch):
    sys.modules.pop('config', None)
    monkeypatch.delenv('MODEL_PATH', raising=False)
    monkeypatch.delenv('HF_TOKEN', raising=False)
    from config import MSTConfig
    cfg = MSTConfig.from_env()
    # default MODEL_PATH is ./mst_model
    assert cfg.model_path == Path('./mst_model')
    assert cfg.hf_token is None


def test_mst_config_from_env_cpu_when_cuda_unavailable(torch_stub, monkeypatch):
    torch_stub.cuda.is_available.return_value = False
    sys.modules.pop('config', None)
    from config import MSTConfig
    cfg = MSTConfig.from_env()
    assert cfg.device == 'cpu'


def test_mst_config_http_proxy_default_none(torch_stub, monkeypatch):
    sys.modules.pop('config', None)
    monkeypatch.delenv('HTTP_PROXY', raising=False)
    from config import MSTConfig
    cfg = MSTConfig.from_env()
    assert cfg.http_proxy is None
