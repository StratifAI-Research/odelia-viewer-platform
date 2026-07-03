"""
ODV-214 smoke check: build a subunit via create_model and run a forward pass on
a dummy volume at the subunit's declared INPUT_SIZE.

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
from models import input_size


def run(model_name: str = "Pimed") -> None:
    model, info = build_model(model_name)
    assert info["model_name"] == model_name

    d, h, w = input_size(model_name)
    x = torch.randn(1, 1, d, h, w)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (1, 3), f"unexpected output shape: {tuple(out.shape)}"
    print(f"OK: built '{model_name}', forward {tuple(x.shape)} -> {tuple(out.shape)}")


if __name__ == "__main__":
    run(os.getenv("MODEL_NAME", "Pimed"))
