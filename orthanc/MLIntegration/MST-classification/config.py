"""
MST Classification service configuration
"""
import os
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class MSTConfig:
    """Configuration for MST classification service"""
    model_path: Path
    hf_token: Optional[str]
    device: str
    http_proxy: Optional[str] = None
    https_proxy: Optional[str] = None

    @classmethod
    def from_env(cls) -> 'MSTConfig':
        """Create configuration from environment variables"""
        return cls(
            model_path=Path(os.getenv("MODEL_PATH", "./mst_model")),
            hf_token=os.getenv("HF_TOKEN", None),
            device="cuda" if torch.cuda.is_available() else "cpu",
            http_proxy=os.getenv("HTTP_PROXY", None),
            https_proxy=os.getenv("HTTPS_PROXY", None)
        )
