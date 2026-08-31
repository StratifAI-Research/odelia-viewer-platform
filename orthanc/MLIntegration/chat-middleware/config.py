"""
Static configuration loaded from environment variables
"""

import os
from dataclasses import dataclass
from pathlib import Path

# Ollama's hosted API. It behaves as a remote Ollama host, so the same
# /v1/chat/completions and /api/tags endpoints apply — the only differences from a
# local instance are the base URL and a Bearer token.
DEFAULT_OLLAMA_CLOUD_URL = "https://ollama.com"

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean env var. Absent/unrecognized values fall back to `default`."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUTHY


@dataclass
class ChatMiddlewareConfig:
    """Static configuration for the chat middleware service"""

    # LLM backend (external instance)
    ollama_url: str = "http://host.docker.internal:11434"
    ollama_model: str = "MedAIBase/MedGemma1.5:4b"
    backend_type: str = "ollama"  # "ollama" or "llamacpp"

    # Ollama Cloud (optional, opt-in).
    #
    # Selecting the cloud provider sends the preprocessed DICOM slices to a
    # third party, so it is gated: `allow_cloud_backend` defaults to False and
    # the middleware refuses to route to the cloud until an operator enables it.
    # The API key is operator-supplied and stays server-side — it is never
    # returned by any endpoint and never logged.
    allow_cloud_backend: bool = False
    ollama_cloud_url: str = DEFAULT_OLLAMA_CLOUD_URL
    ollama_cloud_api_key: str = ""
    ollama_cloud_model: str = ""  # Empty => the user picks one in the chat panel

    # WADO-RS
    wado_base_url: str = "http://orthanc-viewer:8042/dicom-web"

    # Preprocessing defaults (can be overridden via debug API)
    num_slices: int = 5
    image_folder: Path = Path("/tmp/chat-middleware-images")

    # Cache
    max_cache_entries: int = 100

    # Server
    host: str = "0.0.0.0"
    port: int = 5560

    @classmethod
    def from_env(cls) -> "ChatMiddlewareConfig":
        """Load configuration from environment variables"""
        return cls(
            ollama_url=os.getenv("OLLAMA_URL", "http://host.docker.internal:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "MedAIBase/MedGemma1.5:4b"),
            backend_type=os.getenv("BACKEND_TYPE", "ollama").lower(),
            allow_cloud_backend=_env_bool("ALLOW_CLOUD_BACKEND", False),
            ollama_cloud_url=os.getenv("OLLAMA_CLOUD_URL", DEFAULT_OLLAMA_CLOUD_URL),
            ollama_cloud_api_key=os.getenv("OLLAMA_API_KEY", ""),
            ollama_cloud_model=os.getenv("OLLAMA_CLOUD_MODEL", ""),
            wado_base_url=os.getenv("WADO_BASE_URL", "http://orthanc-viewer:8042/dicom-web"),
            num_slices=int(os.getenv("NUM_SLICES", "5")),
            image_folder=Path(os.getenv("IMAGE_FOLDER", "/tmp/chat-middleware-images")),
            max_cache_entries=int(os.getenv("MAX_CACHE_ENTRIES", "100")),
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "5560")),
        )


# Global config instance - initialized once at startup
config: ChatMiddlewareConfig = None


def init_config() -> ChatMiddlewareConfig:
    """Initialize global configuration"""
    global config
    config = ChatMiddlewareConfig.from_env()

    # Ensure image folder exists
    config.image_folder.mkdir(parents=True, exist_ok=True)

    return config


def get_config() -> ChatMiddlewareConfig:
    """Get global configuration, initializing if needed"""
    global config
    if config is None:
        return init_config()
    return config
