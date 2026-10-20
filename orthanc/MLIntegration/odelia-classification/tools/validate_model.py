"""Validate baked weights, optionally skipping the full-size forward for CI."""

from __future__ import annotations

import argparse
import os

import torch
from model_loader import build_model
from models import assert_forward_contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument(
        "--load-only", action="store_true", help="Strictly load weights without running inference"
    )
    args = parser.parse_args()
    os.environ["MODEL_DEVICE"] = "cpu"
    torch.set_num_threads(2)
    model, info = build_model(args.model)
    if args.load_only:
        print(f"Validated {info['model_name']} {info['weights']}: strict loading (no inference)")
        return
    shape = assert_forward_contract(model, args.model)
    print(f"Validated {info['model_name']} {info['weights']}: logits {shape}")


if __name__ == "__main__":
    main()
