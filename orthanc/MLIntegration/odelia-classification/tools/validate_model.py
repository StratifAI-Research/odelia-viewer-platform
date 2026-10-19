"""Validate baked weights and a full-size forward without HF credentials."""

from __future__ import annotations

import os
import sys

import torch
from model_loader import build_model
from models import assert_forward_contract


def main() -> None:
    os.environ["MODEL_DEVICE"] = "cpu"
    torch.set_num_threads(2)
    model, info = build_model(sys.argv[1])
    shape = assert_forward_contract(model, sys.argv[1])
    print(f"Validated {info['model_name']} {info['weights']}: logits {shape}")


if __name__ == "__main__":
    main()
