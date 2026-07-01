"""
Configuration for the generalized ODELIA model-service block (ODV-214).

This service builds any model by MODEL_NAME through the vendored MediSwarm
create_model factory. MST-classification/ is left untouched.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import torch


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
    """Configuration for the generalized model service."""

    model_name: str
    num_classes: int
    model_path: Path
    device: str
    # ODV-216 seam: trained weights are loaded from here when published.
    checkpoint_uri: str | None = None
    hf_token: str | None = None
    http_proxy: str | None = None
    https_proxy: str | None = None

    @classmethod
    def from_env(cls) -> "ModelServiceConfig":
        """Create configuration from environment variables."""
        return cls(
            model_name=os.getenv("MODEL_NAME", "MST"),
            num_classes=int(os.getenv("NUM_CLASSES", "3")),
            model_path=Path(os.getenv("MODEL_PATH", "./model")),
            device=resolve_device(),
            checkpoint_uri=os.getenv("CHECKPOINT_URI", None),
            hf_token=os.getenv("HF_TOKEN", None),
            http_proxy=os.getenv("HTTP_PROXY", None),
            https_proxy=os.getenv("HTTPS_PROXY", None),
        )
