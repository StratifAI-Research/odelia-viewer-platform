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

# OpenRouter, the alternative hosted provider. Also OpenAI-compatible, so chat is
# the same POST; the base deliberately stops at /api because the client appends
# /v1/chat/completions itself. Its catalogue lives at /api/v1/models and has no
# Ollama-style /api/tags or /api/show — see OllamaClient.list_models_detailed.
DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api"

# Hosted providers the cloud slot can point at. One per deployment: they are
# alternatives, not a menu the reader chooses from at chat time.
CLOUD_PROVIDER_OLLAMA = "ollama"
CLOUD_PROVIDER_OPENROUTER = "openrouter"

# Per-provider defaults and the operator-facing names for each. The API key env
# var is named after the service the operator got the key from, so a key pasted
# from OpenRouter's dashboard goes into OPENROUTER_API_KEY, not OLLAMA_API_KEY.
_CLOUD_PROVIDERS = {
    CLOUD_PROVIDER_OLLAMA: {
        "label": "Ollama Cloud",
        "url_env": "OLLAMA_CLOUD_URL",
        "key_env": "OLLAMA_API_KEY",
        "model_env": "OLLAMA_CLOUD_MODEL",
        "default_url": DEFAULT_OLLAMA_CLOUD_URL,
    },
    CLOUD_PROVIDER_OPENROUTER: {
        "label": "OpenRouter",
        "url_env": "OPENROUTER_URL",
        "key_env": "OPENROUTER_API_KEY",
        "model_env": "OPENROUTER_MODEL",
        "default_url": DEFAULT_OPENROUTER_URL,
    },
}

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean env var. Absent/unrecognized values fall back to `default`."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUTHY


def _env_list(name: str) -> tuple[str, ...]:
    """Parse a comma-separated env var into a tuple, dropping blanks."""
    raw = os.getenv(name, "")
    return tuple(part.strip() for part in raw.split(",") if part.strip())


@dataclass
class ChatMiddlewareConfig:
    """Static configuration for the chat middleware service"""

    # LLM backend (external instance)
    ollama_url: str = "http://host.docker.internal:11434"
    ollama_model: str = "MedAIBase/MedGemma1.5:4b"
    backend_type: str = "ollama"  # "ollama" or "llamacpp"

    # Hosted cloud backend (optional, opt-in).
    #
    # Selecting the cloud provider sends the preprocessed DICOM slices to a
    # third party, so it is gated: `allow_cloud_backend` defaults to False and
    # the middleware refuses to route to the cloud until an operator enables it.
    # The API key is operator-supplied and stays server-side — it is never
    # returned by any endpoint and never logged.
    #
    # `cloud_provider` picks *which* hosted service the one cloud slot points at.
    # The two are alternatives: a deployment configures one, and the chat panel
    # keeps offering a single "cloud" choice either way.
    cloud_provider: str = CLOUD_PROVIDER_OLLAMA
    allow_cloud_backend: bool = False
    cloud_url: str = DEFAULT_OLLAMA_CLOUD_URL
    cloud_api_key: str = ""
    cloud_model: str = ""  # Empty => the user picks one in the chat panel

    # Optional catalogue filter, applied to whatever the cloud host lists.
    # Ollama Cloud offers a couple of dozen models; OpenRouter offers several
    # hundred, which is a model menu nobody can read. Each entry is matched as a
    # case-insensitive substring of the model id; empty means "offer everything".
    cloud_model_filter: tuple[str, ...] = ()

    # OpenRouter brokers each request out to one of several inference providers,
    # so "off-site" is a wider set of hosts than with a single-hop service. Deny
    # is the default: it restricts routing to providers that do not retain or
    # train on prompts, at the cost of some models becoming unavailable.
    openrouter_data_collection: str = "deny"

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

    @property
    def cloud_label(self) -> str:
        """Operator-facing name of the configured cloud service."""
        return _CLOUD_PROVIDERS[self.cloud_provider]["label"]

    @property
    def cloud_key_env(self) -> str:
        """Env var an operator sets to supply this provider's API key."""
        return _CLOUD_PROVIDERS[self.cloud_provider]["key_env"]

    @property
    def cloud_model_env(self) -> str:
        """Env var an operator sets to preselect this provider's model.

        Looked up rather than built from the provider name: the names do not
        follow one rule. Ollama needs CLOUD in OLLAMA_CLOUD_MODEL to tell it from
        the local OLLAMA_MODEL, but not in OLLAMA_API_KEY; OpenRouter has no local
        counterpart and so no CLOUD anywhere. Any formula gets one of them wrong.
        """
        return _CLOUD_PROVIDERS[self.cloud_provider]["model_env"]

    @classmethod
    def from_env(cls) -> "ChatMiddlewareConfig":
        """Load configuration from environment variables"""
        # An unrecognized CLOUD_PROVIDER falls back to Ollama rather than raising:
        # a typo here must not take the whole chat service down, and the cloud slot
        # is gated behind ALLOW_CLOUD_BACKEND anyway.
        provider = os.getenv("CLOUD_PROVIDER", CLOUD_PROVIDER_OLLAMA).strip().lower()
        if provider not in _CLOUD_PROVIDERS:
            provider = CLOUD_PROVIDER_OLLAMA
        spec = _CLOUD_PROVIDERS[provider]

        return cls(
            ollama_url=os.getenv("OLLAMA_URL", "http://host.docker.internal:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "MedAIBase/MedGemma1.5:4b"),
            backend_type=os.getenv("BACKEND_TYPE", "ollama").lower(),
            cloud_provider=provider,
            allow_cloud_backend=_env_bool("ALLOW_CLOUD_BACKEND", False),
            cloud_url=os.getenv(spec["url_env"], spec["default_url"]),
            cloud_api_key=os.getenv(spec["key_env"], ""),
            cloud_model=os.getenv(spec["model_env"], ""),
            cloud_model_filter=_env_list("CLOUD_MODEL_FILTER"),
            openrouter_data_collection=(
                "allow" if _env_bool("OPENROUTER_ALLOW_DATA_COLLECTION", False) else "deny"
            ),
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
