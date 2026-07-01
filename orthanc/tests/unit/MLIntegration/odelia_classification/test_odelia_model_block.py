"""ODV-214 unit tests for the generalized ODELIA model block.

Under tests/unit/MLIntegration/odelia_classification/ so the autouse
_force_odelia_path fixture (conftest.py) puts the service dir at sys.path[0] and
evicts colliding sibling module names before each test. Service modules are
therefore imported INSIDE each test, resolving to odelia-classification rather
than a sibling ML service left in sys.modules.

These use REAL torch/monai (not the torch_stub fixture): the build+forward test
needs a real model. MONAI's pretrained download is neutralized for hermeticity.
"""

import pytest
import torch


class TestResolveDevice:
    def test_cuda_when_requested_and_available(self, monkeypatch):
        import config

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setenv("MODEL_DEVICE", "cuda")
        assert config.resolve_device() == "cuda"

    def test_cpu_when_requested_even_with_gpu(self, monkeypatch):
        import config

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setenv("MODEL_DEVICE", "cpu")
        assert config.resolve_device() == "cpu"

    def test_cuda_requested_but_unavailable_fails(self, monkeypatch):
        import config

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setenv("MODEL_DEVICE", "cuda")
        with pytest.raises(RuntimeError, match="cuda"):
            config.resolve_device()

    def test_unset_fails_loudly(self, monkeypatch):
        import config

        monkeypatch.delenv("MODEL_DEVICE", raising=False)
        with pytest.raises(RuntimeError, match="MODEL_DEVICE"):
            config.resolve_device()

    def test_invalid_value_fails_loudly(self, monkeypatch):
        import config

        monkeypatch.setenv("MODEL_DEVICE", "gpu")
        with pytest.raises(RuntimeError, match="MODEL_DEVICE"):
            config.resolve_device()


class TestClassificationResponse:
    def test_argmax_and_passthrough(self):
        from response_builder import build_classification_response

        resp = build_classification_response([0.1, 0.7, 0.2], {"model_name": "Pimed"})
        assert resp["predicted_class"] == 1
        assert resp["probabilities"] == [0.1, 0.7, 0.2]
        assert resp["model_info"]["model_name"] == "Pimed"

    def test_empty_probs(self):
        from response_builder import build_classification_response

        resp = build_classification_response([], None)
        assert resp["predicted_class"] is None
        assert resp["model_info"] == {}


def test_build_model_requires_model_device(monkeypatch):
    """build_model fails loudly when MODEL_DEVICE is unset.

    Hermetic: resolve_device() raises before any model is instantiated.
    """
    monkeypatch.delenv("MODEL_DEVICE", raising=False)
    from model_loader import build_model

    with pytest.raises(RuntimeError, match="MODEL_DEVICE"):
        build_model("Pimed")


def test_build_and_forward_pimed(monkeypatch):
    """create_model builds a roster model that forwards to [B, 3] (random init).

    Pimed (MONAI ResNet wrapper) builds from scratch with no network access.
    """
    monkeypatch.setenv("MODEL_DEVICE", "cpu")

    from model_loader import build_model

    model, info = build_model("Pimed")
    assert info["model_name"] == "Pimed"
    assert info["weights"] == "init-only"

    x = torch.randn(1, 1, 32, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 3)
