"""
MST Classification service configuration
"""

import os
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class MSTConfig:
    """Configuration for MST classification service"""

    model_path: Path
    hf_token: str | None
    device: str
    http_proxy: str | None = None
    https_proxy: str | None = None

    @classmethod
    def from_env(cls) -> "MSTConfig":
        """Create configuration from environment variables"""
        return cls(
            model_path=Path(os.getenv("MODEL_PATH", "./mst_model")),
            hf_token=os.getenv("HF_TOKEN", None),
            device="cuda" if torch.cuda.is_available() else "cpu",
            http_proxy=os.getenv("HTTP_PROXY", None),
            https_proxy=os.getenv("HTTPS_PROXY", None),
        )
