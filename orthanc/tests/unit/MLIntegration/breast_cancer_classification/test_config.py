"""Tests for breast-cancer-classification/config.py — BreastCancerConfig dataclass.

config.py imports torch at module level, so we inject a mock via torch_stub
before loading.  torch_stub fixture is provided by
tests/unit/MLIntegration/conftest.py.
"""
import sys
from pathlib import Path
import dataclasses


def test_breast_cancer_config_fields_present(torch_stub):
    sys.modules.pop('config', None)
    from config import BreastCancerConfig
    field_names = {f.name for f in dataclasses.fields(BreastCancerConfig)}
    assert 'model_path' in field_names
    assert 'device' in field_names


def test_breast_cancer_config_model_path_is_path(torch_stub):
    sys.modules.pop('config', None)
    from config import BreastCancerConfig
    cfg = BreastCancerConfig(model_path=Path('./model.pth'), device='cpu')
    assert isinstance(cfg.model_path, Path)


def test_breast_cancer_config_device_field(torch_stub):
    sys.modules.pop('config', None)
    from config import BreastCancerConfig
    cfg = BreastCancerConfig(model_path=Path('./model.pth'), device='cpu')
    assert cfg.device == 'cpu'


def test_breast_cancer_config_from_env_uses_model_path_env(torch_stub, monkeypatch):
    sys.modules.pop('config', None)
    monkeypatch.setenv('MODEL_PATH', '/custom/model.pth')
    from config import BreastCancerConfig
    cfg = BreastCancerConfig.from_env()
    assert cfg.model_path == Path('/custom/model.pth')


def test_breast_cancer_config_from_env_default_model_path(torch_stub, monkeypatch):
    sys.modules.pop('config', None)
    monkeypatch.delenv('MODEL_PATH', raising=False)
    from config import BreastCancerConfig
    cfg = BreastCancerConfig.from_env()
    assert cfg.model_path == Path('./models/resnet18_abrv_b=32_split0-0.pth')


def test_breast_cancer_config_from_env_cpu_when_cuda_unavailable(torch_stub):
    torch_stub.cuda.is_available.return_value = False
    sys.modules.pop('config', None)
    from config import BreastCancerConfig
    cfg = BreastCancerConfig.from_env()
    assert cfg.device == 'cpu'


def test_breast_cancer_config_from_env_returns_instance(torch_stub):
    sys.modules.pop('config', None)
    from config import BreastCancerConfig
    cfg = BreastCancerConfig.from_env()
    assert isinstance(cfg, BreastCancerConfig)
