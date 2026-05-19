"""
Static configuration loaded from environment variables
"""
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ChatMiddlewareConfig:
    """Static configuration for the chat middleware service"""
    
    # LLM backend (external instance)
    ollama_url: str = "http://host.docker.internal:11434"
    ollama_model: str = "MedAIBase/MedGemma1.5:4b"
    backend_type: str = "ollama"  # "ollama" or "llamacpp"
    
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
