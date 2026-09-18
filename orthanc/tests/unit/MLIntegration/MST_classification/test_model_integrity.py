import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest


def bundle(tmp_path):
    from model_integrity import REQUIRED_FILES
    digests = {}
    for name in REQUIRED_FILES:
        data = (name + "\n").encode()
        (tmp_path / name).write_bytes(data)
        digests[name] = hashlib.sha256(data).hexdigest()
    integrity = {"repo_id": "ODELIA-AI/MST", "revision": "a" * 40, "sha256": digests}
    path = tmp_path / "integrity.json"
    path.write_text(json.dumps(integrity))
    return path, integrity


def test_verifies_every_file_and_rejects_corruption(tmp_path):
    from model_integrity import read_integrity, verify_files
    path, integrity = bundle(tmp_path)
    assert read_integrity(path) == integrity
    assert len(verify_files(tmp_path, integrity)) == 4
    for name in integrity["sha256"]:
        original = (tmp_path / name).read_bytes()
        (tmp_path / name).write_bytes(b"tampered")
        with pytest.raises(ValueError, match=name.replace(".", r"\.")):
            verify_files(tmp_path, integrity)
        (tmp_path / name).write_bytes(original)
    (tmp_path / "state_dict.pt").unlink()
    with pytest.raises(FileNotFoundError):
        verify_files(tmp_path, integrity)


@pytest.mark.parametrize("change", ["branch", "missing", "extra", "digest", "repo"])
def test_rejects_invalid_integrity(tmp_path, change):
    from model_integrity import read_integrity
    path, integrity = bundle(tmp_path)
    if change == "branch": integrity["revision"] = "main"
    if change == "missing": del integrity["sha256"]["models.py"]
    if change == "extra": integrity["sha256"]["../evil.py"] = "a" * 64
    if change == "digest": integrity["sha256"]["models.py"] = "sha1"
    if change == "repo": integrity["repo_id"] = "another/repo"
    path.write_text(json.dumps(integrity))
    with pytest.raises(ValueError): read_integrity(path)


def test_loader_checks_before_import_and_uses_pinned_revision(tmp_path, monkeypatch):
    import model_loader
    from model_integrity import verify_files
    path, integrity = bundle(tmp_path)
    monkeypatch.setattr(model_loader, "MODEL_PATH", str(tmp_path))
    monkeypatch.setattr(model_loader, "HF_TOKEN", None)
    monkeypatch.setattr(model_loader, "read_integrity", lambda _: integrity)
    for name in integrity["sha256"]:
        (tmp_path / name).unlink()
    def fetch(**kw):
        name = kw["filename"]
        (tmp_path / name).write_bytes((name + "\n").encode())
        return str(tmp_path / name)
    download = MagicMock(side_effect=fetch)
    monkeypatch.setattr(model_loader, "hf_hub_download", download)
    assert model_loader.download_model_files() == verify_files(tmp_path, integrity)
    assert download.call_count == 4
    assert all(call.kwargs["revision"] == "a" * 40 for call in download.call_args_list)
    (tmp_path / "predict_attention.py").write_text("raise AssertionError('executed unverified code')")
    with pytest.raises(ValueError, match="integrity"):
        model_loader.load_model()

def test_cached_bundle_is_verified_without_network(tmp_path, monkeypatch):
    import model_loader
    _, integrity = bundle(tmp_path)
    monkeypatch.setattr(model_loader, "MODEL_PATH", str(tmp_path))
    monkeypatch.setattr(model_loader, "read_integrity", lambda _: integrity)
    download = MagicMock(side_effect=AssertionError("unexpected network access"))
    monkeypatch.setattr(model_loader, "hf_hub_download", download)
    assert len(model_loader.download_model_files()) == 4
    download.assert_not_called()
    (tmp_path / "models.py").write_text("corrupt")
    with pytest.raises(ValueError, match="integrity"):
        model_loader.download_model_files()

@pytest.mark.parametrize("corrupt", [False, True])
def test_cached_bundle_clears_download_proxies(tmp_path, monkeypatch, corrupt):
    import os
    import model_loader
    _, integrity = bundle(tmp_path)
    monkeypatch.setattr(model_loader, "MODEL_PATH", str(tmp_path))
    monkeypatch.setattr(model_loader, "read_integrity", lambda _: integrity)
    names = ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy")
    for name in names:
        monkeypatch.setenv(name, "http://download-proxy.example:8080")
    if corrupt:
        (tmp_path / "models.py").write_text("corrupt")
        with pytest.raises(ValueError, match="integrity"):
            model_loader.download_model_files()
    else:
        model_loader.download_model_files()
    assert all(name not in os.environ for name in names)


def test_loader_reads_local_config_and_weights_without_secondary_download(tmp_path, monkeypatch):
    import sys
    import torch
    import model_loader
    from types import ModuleType
    _, integrity = bundle(tmp_path)
    config_bytes = b'{"hparams": {"num_classes": 2}}'
    (tmp_path / "model_config.json").write_bytes(config_bytes)
    integrity["sha256"]["model_config.json"] = hashlib.sha256(config_bytes).hexdigest()
    monkeypatch.setattr(model_loader, "MODEL_PATH", str(tmp_path))
    monkeypatch.setattr(model_loader, "read_integrity", lambda _: integrity)
    download = MagicMock(side_effect=AssertionError("unexpected network access"))
    monkeypatch.setattr(model_loader, "hf_hub_download", download)
    models = ModuleType("models")
    models.MSTRegression = MagicMock()
    prediction = ModuleType("predict_attention")
    prediction.run_prediction = MagicMock()
    prediction.load_model = MagicMock(side_effect=AssertionError("secondary loader called"))
    monkeypatch.setitem(sys.modules, "models", models)
    monkeypatch.setitem(sys.modules, "predict_attention", prediction)
    weights = {"weight": torch.tensor([1.0])}
    read_weights = MagicMock(return_value=weights)
    monkeypatch.setattr(torch, "load", read_weights)
    monkeypatch.setattr(sys, "path", list(sys.path))
    model, predict, info = model_loader.load_model()
    models.MSTRegression.assert_called_once_with(weights=False, num_classes=2)
    read_weights.assert_called_once_with(str(tmp_path / "state_dict.pt"), map_location="cpu", weights_only=True)
    model.load_state_dict.assert_called_once_with(weights, strict=True)
    model.eval.assert_called_once()
    assert predict is prediction.run_prediction
    assert info["revision"] == integrity["revision"]
    download.assert_not_called()
    prediction.load_model.assert_not_called()
