"""
Configuration for the generalized ODELIA model service (ODV-214).

One image = one model: the served subunit is whichever one the build baked in
(resolved via the models package), not a runtime env choice. MODEL_DEVICE is the
only required runtime setting.
"""

import os
from dataclasses import dataclass

import torch
from models import resolve_baked_model


def resolve_device() -> str:
    """Select the inference device from the required ``MODEL_DEVICE`` env var.

    ``MODEL_DEVICE`` must be set explicitly so a device is never chosen
    implicitly:
      - ``cpu``  -> CPU (works with or without a GPU)
      - ``cuda`` -> GPU; fails loudly if no CUDA device is available
      - anything else, including unset -> fails loudly
    """
    device = os.getenv("MODEL_DEVICE", "").strip().lower()
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("MODEL_DEVICE=cuda but no CUDA GPU is available.")
        return "cuda"
    raise RuntimeError(
        f"MODEL_DEVICE must be set to 'cpu' or 'cuda' (got {os.getenv('MODEL_DEVICE')!r})."
    )


@dataclass
class ModelServiceConfig:
    """Runtime configuration for the model service."""

    model_name: str
    device: str

    @classmethod
    def from_env(cls) -> "ModelServiceConfig":
        """Resolve the baked subunit + device from the environment."""
        return cls(
            model_name=resolve_baked_model(),
            device=resolve_device(),
        )
