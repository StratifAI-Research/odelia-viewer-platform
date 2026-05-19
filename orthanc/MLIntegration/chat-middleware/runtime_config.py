"""
Mutable runtime configuration that can be updated via debug API.

This module provides runtime-adjustable settings that can be changed via the debug API
without restarting the service. Default values are initialized from static config (env vars).
"""
from dataclasses import dataclass, field
from typing import Optional
from models import SliceStrategy


DEFAULT_SYSTEM_PROMPT = """You are a radiology assistant specialized in medical image analysis.
You are viewing DICOM medical images from a study. Analyze the images carefully and
provide accurate, professional observations. If you cannot determine something with
confidence, say so clearly."""


@dataclass
class PreprocessingParams:
    """Preprocessing parameters that can be adjusted at runtime.
    
    Defaults are set from static config (env vars) at initialization time.
    """
    num_slices: int = 5  # Default, overridden by config.num_slices at init
    slice_strategy: SliceStrategy = SliceStrategy.CENTRAL
    central_percentage: int = 60  # For central strategy, % of volume to use


@dataclass
class OllamaOptions:
    """
    OpenAI-compatible generation options adjustable at runtime.

    Only fields supported by Ollama's /v1/chat/completions endpoint.
    num_ctx is NOT supported here — set it via a Modelfile instead.
    """
    max_tokens: Optional[int] = None        # Max tokens to generate
    temperature: Optional[float] = None     # Sampling temperature
    top_p: Optional[float] = None           # Top-p (nucleus) sampling
    stop: Optional[list] = None             # Stop sequences
    seed: Optional[int] = None              # Random seed for reproducibility
    think: Optional[bool] = None            # Enable thinking/reasoning mode (for models like DeepSeek R1, QwQ)

    def to_dict(self) -> dict:
        """Convert to dict, excluding None values (used as runtime_options for OllamaClient)"""
        result = {}
        if self.max_tokens is not None:
            result["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            result["temperature"] = self.temperature
        if self.top_p is not None:
            result["top_p"] = self.top_p
        if self.stop is not None:
            result["stop"] = self.stop
        if self.seed is not None:
            result["seed"] = self.seed
        if self.think is not None:
            result["think"] = self.think
        return result

    def to_full_dict(self) -> dict:
        """Convert to dict including None values for display"""
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stop": self.stop,
            "seed": self.seed,
            "think": self.think,
        }


class RuntimeConfig:
    """
    Singleton holding runtime-adjustable configuration.
    Can be modified via debug API without restarting the service.
    
    Default values are initialized from static config (env vars).
    """

    def __init__(self):
        # Import here to avoid circular imports
        from config import get_config
        config = get_config()
        
        self.system_prompt: str = DEFAULT_SYSTEM_PROMPT
        self.model: str = config.ollama_model
        
        # Initialize preprocessing from static config (env vars)
        self.preprocessing: PreprocessingParams = PreprocessingParams(
            num_slices=config.num_slices,  # From NUM_SLICES env var
        )
        
        self.ollama_options: OllamaOptions = OllamaOptions()

    def update(
        self,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        preprocessing: Optional[dict] = None,
        ollama_options: Optional[dict] = None
    ) -> None:
        """
        Update configuration values.

        Args:
            system_prompt: New system prompt for the LLM
            model: Model name override for the LLM
            preprocessing: Dict with preprocessing params to update
            ollama_options: Dict with Ollama generation options to update
        """
        if system_prompt is not None:
            self.system_prompt = system_prompt

        if model is not None:
            self.model = model

        if preprocessing is not None:
            if "num_slices" in preprocessing and preprocessing["num_slices"] is not None:
                self.preprocessing.num_slices = preprocessing["num_slices"]
            if "slice_strategy" in preprocessing and preprocessing["slice_strategy"] is not None:
                strategy = preprocessing["slice_strategy"]
                if isinstance(strategy, str):
                    self.preprocessing.slice_strategy = SliceStrategy(strategy)
                else:
                    self.preprocessing.slice_strategy = strategy
            if "central_percentage" in preprocessing and preprocessing["central_percentage"] is not None:
                self.preprocessing.central_percentage = preprocessing["central_percentage"]

        if ollama_options is not None:
            for key, value in ollama_options.items():
                if hasattr(self.ollama_options, key):
                    setattr(self.ollama_options, key, value)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary for API response"""
        return {
            "system_prompt": self.system_prompt,
            "model": self.model,
            "preprocessing": {
                "num_slices": self.preprocessing.num_slices,
                "slice_strategy": self.preprocessing.slice_strategy.value,
                "central_percentage": self.preprocessing.central_percentage,
            },
            "ollama_options": self.ollama_options.to_full_dict(),
        }


# Global runtime config instance
_runtime_config: Optional[RuntimeConfig] = None


def get_runtime_config() -> RuntimeConfig:
    """Get the global runtime configuration singleton"""
    global _runtime_config
    if _runtime_config is None:
        _runtime_config = RuntimeConfig()
    return _runtime_config


def reset_runtime_config() -> RuntimeConfig:
    """Reset runtime configuration to defaults (useful for testing)"""
    global _runtime_config
    _runtime_config = RuntimeConfig()
    return _runtime_config
