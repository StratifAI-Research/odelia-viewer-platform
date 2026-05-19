"""
Breast Cancer Classification service configuration
"""
import os
import torch
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BreastCancerConfig:
    """Configuration for breast cancer classification service"""
    model_path: Path
    device: str

    @classmethod
    def from_env(cls) -> 'BreastCancerConfig':
        """Create configuration from environment variables"""
        return cls(
            model_path=Path(os.getenv("MODEL_PATH", "./models/resnet18_abrv_b=32_split0-0.pth")),
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
