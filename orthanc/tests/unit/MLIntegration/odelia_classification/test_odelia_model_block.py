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


# Challenge roster models: (name, dummy input shape, expected state_dict key count).
# All build init-only with no network access (pretrained backbones disabled).
# The key count is a regression guard on the preserved module structure.
_ROSTER = [
    ("Pimed", (1, 1, 32, 32, 32), 104),
    ("DivideAndConquer", (1, 1, 64, 64, 64), 450),
    ("BCN_AIM", (1, 1, 64, 64, 64), 161),
    ("LME_ABMIL", (1, 3, 16, 224, 224), 179),
    ("agaldran", (1, 1, 16, 112, 112), 397),
]


@pytest.mark.parametrize(("name", "shape", "n_keys"), _ROSTER, ids=[m[0] for m in _ROSTER])
def test_roster_model_builds_and_forwards(monkeypatch, name, shape, n_keys):
    """Each challenge roster model builds init-only and forwards to [B, 3].

    Random weights, no network (pretrained backbones off). Asserts the preserved
    state_dict key count so a structural regression is caught.
    """
    monkeypatch.setenv("MODEL_DEVICE", "cpu")

    from model_loader import build_model

    model, info = build_model(name)
    assert info["model_name"] == name
    assert info["weights"] == "init-only"
    assert len(model.state_dict()) == n_keys

    with torch.no_grad():
        out = model(torch.randn(*shape))
    assert out.shape == (1, 3)


def test_mst_builds_and_forwards(monkeypatch):
    """MST wiring (per-slice encode -> transformer fusion -> head) forwards to [B, 3].

    The DINOv2 backbone is mocked so this is network-free and fast; the real
    DINOv2 integration is exercised by smoke_test.py (and ODV-219).
    """
    monkeypatch.setenv("MODEL_DEVICE", "cpu")

    class _FakeDino(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.num_features = 384
            self.mask_token = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.randn(x.shape[0], self.num_features)

    monkeypatch.setattr(torch.hub, "load", lambda *a, **k: _FakeDino())

    from model_loader import build_model

    model, info = build_model("MST")
    assert info["model_name"] == "MST"

    with torch.no_grad():
        out = model(torch.randn(1, 1, 4, 224, 224))
    assert out.shape == (1, 3)
