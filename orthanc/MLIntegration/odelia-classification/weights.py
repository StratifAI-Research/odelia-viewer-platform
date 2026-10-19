"""Strict local weight loading and release provenance for ODELIA subunits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_release(directory: Path, model_name: str) -> dict[str, Any]:
    """Read the optional release record; reject a mismatched architecture."""
    path = directory / "release.json"
    if not path.exists():
        return {}
    release = json.loads(path.read_text())
    if not isinstance(release, dict) or release.get("model") != model_name:
        raise ValueError(f"Release metadata does not describe {model_name}")
    if release.get("weights") != "weights.safetensors":
        raise ValueError("Release must declare weights.safetensors")
    return release


def load_weights(
    model: torch.nn.Module, directory: Path, model_name: str, lock_path: Path
) -> dict[str, Any]:
    """Verify the artifact and load every tensor, without network access."""
    path = directory / "weights.safetensors"
    release = read_release(directory, model_name)
    lock = json.loads(lock_path.read_text()) if lock_path.is_file() else {}
    if lock and lock.get("model") != model_name:
        raise ValueError("models-lock.json names a different model")
    digest = sha256(path)
    expected = [
        record["weights_sha256"] for record in (release, lock) if "weights_sha256" in record
    ]
    if not expected:
        raise ValueError("Weights need a SHA-256 in release.json or models-lock.json")
    if any(value != digest for value in expected):
        raise ValueError("Weights SHA-256 does not match release/build provenance")
    state = load_file(str(path), device="cpu")
    for name, tensor in state.items():
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Nonfinite checkpoint tensor: {name}")
    # Class weights are a training-only buffer, but preserve their explicit key
    # so strict loading still checks the entire exported state dict.
    if "_class_weight" in state and getattr(model, "_class_weight", None) is None:
        if hasattr(model, "_class_weight"):
            delattr(model, "_class_weight")
        model.register_buffer("_class_weight", torch.empty_like(state["_class_weight"]))
    # nnU-Net registers the same parameters through multiple module paths.
    # Reject conflicting aliases rather than allowing the last key to win.
    aliases: dict[tuple[Any, ...], str] = {}
    for name, tensor in model.state_dict().items():
        key = (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()))
        previous = aliases.setdefault(key, name)
        if name in state and previous in state and not torch.equal(state[name], state[previous]):
            raise ValueError(f"Conflicting checkpoint aliases: {previous}, {name}")
    model.load_state_dict(state, strict=True)
    return {
        "weights": "trained",
        "weights_sha256": digest,
        "model_release": release.get("release", lock.get("release", "unversioned")),
        "hf_repo": lock.get("hf_repo"),
        "hf_revision": lock.get("hf_revision"),
        "preprocessing_status": release.get("preprocessing_status", "unverified"),
    }
