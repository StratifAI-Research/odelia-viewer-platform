"""ODV-216: exercise trained artifacts and rejection paths using real tensors."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file


def artifact(tmp_path, state=None):
    from weights import sha256

    model = torch.nn.Linear(2, 3)
    state = model.state_dict() if state is None else state
    path = tmp_path / "weights.safetensors"
    save_file(state, str(path))
    release = {
        "model": "Example",
        "release": "2026-05",
        "weights": path.name,
        "weights_sha256": sha256(path),
    }
    (tmp_path / "release.json").write_text(json.dumps(release))
    return model, release


def test_strict_load_restores_weights_and_provenance(tmp_path):
    from weights import load_weights

    source, release = artifact(tmp_path)
    target = torch.nn.Linear(2, 3)
    info = load_weights(target, tmp_path, "Example", tmp_path / "models-lock.json")
    assert info["weights"] == "trained"
    assert info["model_release"] == "2026-05"
    assert info["weights_sha256"] == release["weights_sha256"]
    assert all(torch.equal(v, source.state_dict()[k]) for k, v in target.state_dict().items())


def test_modified_weights_are_rejected(tmp_path):
    from weights import load_weights

    model, _ = artifact(tmp_path)
    with (tmp_path / "weights.safetensors").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="SHA-256"):
        load_weights(model, tmp_path, "Example", tmp_path / "lock.json")


@pytest.mark.parametrize("problem", ["missing", "unexpected", "shape", "nonfinite"])
def test_valid_checksum_does_not_hide_bad_state(tmp_path, problem):
    from weights import load_weights

    state = torch.nn.Linear(2, 3).state_dict()
    if problem == "missing":
        del state["bias"]
    elif problem == "unexpected":
        state["extra"] = torch.ones(1)
    elif problem == "shape":
        state["weight"] = torch.ones(3, 5)
    else:
        state["weight"][0, 0] = float("nan")
    model, _ = artifact(tmp_path, state)
    with pytest.raises((RuntimeError, ValueError)):
        load_weights(model, tmp_path, "Example", tmp_path / "lock.json")


def test_conflicting_shared_parameters_are_rejected(tmp_path):
    from weights import load_weights

    model = torch.nn.Module()
    model.a = torch.nn.Linear(2, 3)
    model.b = model.a
    state = {k: v.clone() for k, v in model.state_dict().items()}
    state["b.weight"] += 1
    artifact(tmp_path, state)
    with pytest.raises(ValueError, match="Conflicting checkpoint aliases"):
        load_weights(model, tmp_path, "Example", tmp_path / "lock.json")


def test_wrong_release_model_rejected(tmp_path):
    from weights import load_weights

    model, _ = artifact(tmp_path)
    with pytest.raises(ValueError, match="does not describe"):
        load_weights(model, tmp_path, "Other", tmp_path / "lock.json")


def test_lock_checksum_is_verified_even_with_valid_release(tmp_path):
    from weights import load_weights

    model, _ = artifact(tmp_path)
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"model": "Example", "weights_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="SHA-256"):
        load_weights(model, tmp_path, "Example", lock)


def test_missing_weights_require_explicit_preview(monkeypatch):
    from model_loader import build_model

    monkeypatch.setenv("MODEL_DEVICE", "cpu")
    monkeypatch.delenv("ALLOW_INIT_WEIGHTS", raising=False)
    with pytest.raises(FileNotFoundError, match="Missing trained weights"):
        build_model("Pimed")


def test_missing_trained_artifact_never_falls_back(tmp_path, monkeypatch):
    import model_loader

    monkeypatch.setenv("MODEL_DEVICE", "cpu")
    monkeypatch.setenv("ALLOW_INIT_WEIGHTS", "1")
    monkeypatch.setattr(model_loader, "__file__", str(tmp_path / "model_loader.py"))
    (tmp_path / "models-lock.json").write_text(
        json.dumps({"model": "Pimed", "weights": "weights.safetensors"})
    )
    with pytest.raises(FileNotFoundError):
        model_loader.build_model("Pimed")


def test_preflight_rejects_nonfinite_logits(monkeypatch):
    from models import assert_forward_contract

    model = torch.nn.Linear(1, 1)
    monkeypatch.setattr(model, "forward", lambda x: torch.full((1, 3), float("nan")))
    with pytest.raises(AssertionError, match="nonfinite"):
        assert_forward_contract(model, "Pimed")


@pytest.mark.parametrize("revision", ["main", "v1", "abcdef0", "", "../anything"])
def test_hf_requires_immutable_revision(tmp_path, revision):
    from tools.fetch_model import fetch

    with pytest.raises(ValueError, match="40-character"):
        fetch("Pimed", "owner/private", revision, tmp_path / "no-token")


def test_hf_stages_pinned_snapshot_without_importing_code(tmp_path, monkeypatch):
    from tools import fetch_model

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    _, release = artifact(snapshot)
    (snapshot / "__init__.py").write_text('raise RuntimeError("must not execute during fetch")')
    (snapshot / "loader.py").write_text('raise RuntimeError("must not execute during fetch")')
    token = tmp_path / "secret"
    token.write_text("test-secret")
    revision = "a" * 40
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr(fetch_model, "snapshot_download", download)
    monkeypatch.setattr(fetch_model, "available_models", lambda: {"Example": "example"})
    monkeypatch.chdir(tmp_path)
    fetch_model.fetch("Example", "owner/private", revision, token)
    assert calls[0]["revision"] == revision
    assert calls[0]["token"] == "test-secret"
    installed = tmp_path / "models/example/weights.safetensors"
    assert installed.is_file() and not installed.is_symlink()
    lock = json.loads((tmp_path / "models-lock.json").read_text())
    assert lock["weights_sha256"] == release["weights_sha256"]
    assert lock["hf_revision"] == revision
    assert "test-secret" not in (tmp_path / "models-lock.json").read_text()
    assert not (tmp_path / "models/example/secret").exists()


def test_hf_corrupt_download_preserves_existing_subunit(tmp_path, monkeypatch):
    from tools import fetch_model

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    artifact(snapshot)
    (snapshot / "weights.safetensors").write_bytes(b"bad")
    existing = tmp_path / "models/example"
    existing.mkdir(parents=True)
    (existing / "marker").write_text("original")
    monkeypatch.setattr(fetch_model, "snapshot_download", lambda **kwargs: str(snapshot))
    monkeypatch.setattr(fetch_model, "available_models", lambda: {"Example": "example"})
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="SHA-256"):
        fetch_model.fetch("Example", "owner/private", "a" * 40, tmp_path / "absent")
    assert (existing / "marker").read_text() == "original"


def test_build_model_loads_weights_before_reporting_release(tmp_path, monkeypatch):
    import model_loader

    directory = tmp_path / "models/example"
    directory.mkdir(parents=True)
    source, release = artifact(directory)
    target = torch.nn.Linear(2, 3)
    monkeypatch.setattr(model_loader, "__file__", str(tmp_path / "model_loader.py"))
    monkeypatch.setattr(model_loader, "available_models", lambda: {"Example": "example"})
    monkeypatch.setattr(model_loader, "create_model", lambda *a, **kw: target)
    monkeypatch.setenv("MODEL_DEVICE", "cpu")
    monkeypatch.delenv("ALLOW_INIT_WEIGHTS", raising=False)
    model, info = model_loader.build_model("Example")
    assert not model.training
    assert info["version"] == release["release"]
    assert info["weights"] == "trained"
    assert all(torch.equal(v, source.state_dict()[k]) for k, v in model.state_dict().items())


def test_locally_baked_weights_load_from_generated_lock(tmp_path, monkeypatch):
    from tools.bake_weights import main
    from weights import load_weights

    directory = tmp_path / "models/pimed"
    directory.mkdir(parents=True)
    source = torch.nn.Linear(2, 3)
    staged = tmp_path / "staged.safetensors"
    save_file(source.state_dict(), str(staged))
    monkeypatch.chdir(tmp_path)
    assert main(["bake_weights", "Pimed", str(staged), "weights/local.safetensors"]) == 0
    target = torch.nn.Linear(2, 3)
    info = load_weights(target, directory, "Pimed", tmp_path / "models-lock.json")
    assert info["weights"] == "trained"
    assert all(torch.equal(v, source.state_dict()[k]) for k, v in target.state_dict().items())
