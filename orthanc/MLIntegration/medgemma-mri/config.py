"""
MedGemma MRI Classification service configuration
"""

import os
from dataclasses import dataclass

import torch


@dataclass
class MedGemmaConfig:
    """Configuration for MedGemma MRI classification service"""

    model_id: str
    hf_token: str | None
    device: str
    torch_dtype: str
    max_new_tokens: int
    num_slices: int

    @classmethod
    def from_env(cls) -> "MedGemmaConfig":
        """Create configuration from environment variables"""
        return cls(
            model_id=os.getenv("MODEL_ID", "google/medgemma-1.5-4b-it"),
            hf_token=os.getenv("HF_TOKEN", None),
            device="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=os.getenv("TORCH_DTYPE", "bfloat16"),
            max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "500")),
            num_slices=int(os.getenv("NUM_SLICES", "5")),
        )
