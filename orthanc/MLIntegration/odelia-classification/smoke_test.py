"""
ODV-214 smoke check: build a roster model via create_model and run a forward
pass on a dummy volume.

Runs on CPU by defaulting MODEL_DEVICE=cpu. This is the acceptance smoke test
for the model-service block; per-model serving/E2E coverage is ODV-219 / ODV-220.

Usage:
    python smoke_test.py                       # Pimed on CPU
    MODEL_NAME=DivideAndConquer python smoke_test.py
    MODEL_DEVICE=cuda python smoke_test.py      # require a GPU
"""

import os

os.environ.setdefault("MODEL_DEVICE", "cpu")

import torch
from model_loader import build_model


def _dummy_shape() -> tuple[int, int, int]:
    # Divisible by 32 so any conv/transformer downsampling stages are happy.
    raw = os.getenv("SMOKE_INPUT_SHAPE", "32,32,32")
    d, h, w = (int(x) for x in raw.split(","))
    return d, h, w


def run(model_name: str = "Pimed") -> None:
    model, info = build_model(model_name)
    assert info["model_name"] == model_name

    d, h, w = _dummy_shape()
    x = torch.randn(1, 1, d, h, w)
    with torch.no_grad():
        out = model(x)

    assert out.shape[0] == 1, f"unexpected batch dim: {tuple(out.shape)}"
    print(f"OK: built '{model_name}', forward {tuple(x.shape)} -> {tuple(out.shape)}")


if __name__ == "__main__":
    run(os.getenv("MODEL_NAME", "Pimed"))
