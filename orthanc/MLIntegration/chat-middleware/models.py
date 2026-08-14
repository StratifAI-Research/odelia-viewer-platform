"""
Pydantic models for WebSocket messages and debug API
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# Upper bound for the runtime system prompt. Operator-controlled config, not user
# input; ~16k chars (~4k tokens) far exceeds any realistic radiology prompt while
# bounding memory and preventing a degenerate value being sent on every LLM call.
MAX_SYSTEM_PROMPT_LEN = 16000

# =============================================================================
# Enums
# =============================================================================


class ClientMessageType(str, Enum):
    """Types of messages the client can send"""

    CHAT = "chat"  # User sends a message with context
    CANCEL = "cancel"  # Cancel current generation


class ServerMessageType(str, Enum):
    """Types of messages the server can send"""

    CONNECTED = "connected"  # Connection established, includes session_id
    TOKEN = "token"  # Streaming token
    THINKING_TOKEN = "thinking_token"  # Streaming reasoning/thinking token
    DONE = "done"  # Generation complete
    ERROR = "error"  # Error occurred
    PREPROCESSING = "preprocessing"  # Status during preprocessing


class Provider(str, Enum):
    """Which LLM backend chat requests are routed to."""

    LOCAL = "local"  # The self-hosted Ollama / llama.cpp instance (default)
    CLOUD = "cloud"  # Ollama Cloud — sends slices off-site, operator-gated


class SliceStrategy(str, Enum):
    """Slice extraction strategies for DICOM volumes"""

    CENTRAL = "central"  # Extract from central N% of volume
    UNIFORM = "uniform"  # Evenly spaced across entire volume
    FIRST_N = "first_n"  # First N slices
    LAST_N = "last_n"  # Last N slices


# =============================================================================
# WebSocket Messages
# =============================================================================

# A DICOM UID is at most 64 characters. Bounded generously rather than exactly so
# a non-conforming-but-harmless source is not rejected, while a degenerate value
# still cannot reach the cache key or the log.
MAX_UID_LEN = 128

# A message names the slices it wants explicitly, so the list needs an upper
# bound: it decides how many images are encoded into one LLM call. 64 sits well
# above the panel's own ceiling (50 slices per series).
MAX_SLICES_PER_SERIES = 64

# Attaching more than a handful of series already exceeds any vision model's
# image budget. The bound exists so one message cannot trigger an unbounded
# number of WADO retrievals.
MAX_SERIES_PER_MESSAGE = 16


class SliceSelection(BaseModel):
    """Which slices of one series a single message sends.

    Carried per message rather than read from the runtime config, so two
    questions in the same conversation can legitimately ask about different
    slices -- and so the panel's per-message provenance snapshot describes
    something this service actually honoured.

    Slices are named by SOPInstanceUID, never by index. The viewer orders a
    series by its own instance sort; this service builds its volume with GDCM's
    geometric sort. The two normally agree, but an index would mean *different
    pixels* on the day they do not, and neither side would report it. A UID
    resolves to exactly one slice of the reconstructed volume, or to none -- and
    "to none" is an error the user is told about, not something to paper over.
    """

    series_uid: str = Field(min_length=1, max_length=MAX_UID_LEN)

    # The slices to encode, in send order. Empty means "no explicit selection":
    # the service falls back to the runtime preprocessing recipe, which is what
    # a viewer predating this field gets.
    sop_instance_uids: list[str] = Field(default_factory=list, max_length=MAX_SLICES_PER_SERIES)

    # 1-based inclusive range the selection was sampled from, in the viewer's
    # slice order. Audit and logging only -- nothing is selected from these, so a
    # client that reports them wrongly cannot change which pixels are sent.
    range_start: int | None = Field(default=None, ge=1)
    range_end: int | None = Field(default=None, ge=1)
    total_slices: int | None = Field(default=None, ge=0)


class ClientMessage(BaseModel):
    """Message sent from client to server via WebSocket"""

    type: ClientMessageType
    content: str | None = None  # User message text (for CHAT)
    study_uid: str | None = None  # StudyInstanceUID (for CHAT)
    series_uids: list[str] | None = None  # Series context for this message (for CHAT)
    # Per-series slice selection for THIS message. A series listed in
    # `series_uids` with no entry here falls back to the runtime recipe.
    slice_selections: list[SliceSelection] | None = Field(
        default=None, max_length=MAX_SERIES_PER_MESSAGE
    )


class ServerMessage(BaseModel):
    """Message sent from server to client via WebSocket"""

    type: ServerMessageType
    content: str | None = None  # Token content, error message, or status
    session_id: str | None = None  # Returned on CONNECTED
    progress: float | None = None  # For preprocessing status (0.0-1.0)


# =============================================================================
# Debug API Models
# =============================================================================


class PreprocessingConfig(BaseModel):
    """Preprocessing configuration for debug API"""

    num_slices: int | None = None
    slice_strategy: SliceStrategy | None = None
    central_percentage: int | None = None


class OllamaOptionsConfig(BaseModel):
    """OpenAI-compatible generation options for debug API.
    Only fields supported by /v1/chat/completions."""

    max_tokens: int | None = None  # Max tokens to generate
    temperature: float | None = None  # Sampling temperature
    top_p: float | None = None  # Top-p (nucleus) sampling
    stop: list[str] | None = None  # Stop sequences
    seed: int | None = None  # Random seed for reproducibility


# Ollama model tags are `namespace/name:tag`-shaped. Bounded to keep a degenerate
# value out of every subsequent LLM call; generous next to real tags, the longest
# bundled one being ~40 chars.
MAX_MODEL_NAME_LEN = 200


class DebugConfigUpdate(BaseModel):
    """Request body for updating debug configuration"""

    system_prompt: str | None = Field(default=None, max_length=MAX_SYSTEM_PROMPT_LEN)
    model: str | None = Field(default=None, max_length=MAX_MODEL_NAME_LEN)
    preprocessing: PreprocessingConfig | None = None
    ollama_options: OllamaOptionsConfig | None = None
    provider: Provider | None = None
    cloud_model: str | None = Field(default=None, max_length=MAX_MODEL_NAME_LEN)


class DebugConfigResponse(BaseModel):
    """Response body for debug configuration endpoint.

    Deliberately carries no API key. /chat-api/ is proxied without
    authentication, so anything returned here is readable by anyone who can reach
    the viewer host; `cloud_configured` reports only whether a key is present.
    """

    system_prompt: str
    model: str  # Local model name (runtime-adjustable)
    preprocessing: dict[str, Any]
    ollama_static: dict[str, Any]  # Read-only: url, backend_type (from env vars)
    ollama_options: dict[str, Any]  # Adjustable: max_tokens, temperature, top_p, stop, seed

    # Provider routing
    provider: Provider  # Which backend chat is currently routed to
    cloud_model: str  # Selected cloud model tag ("" if none picked yet)
    active_model: str  # Model that the current provider will actually use
    cloud_enabled: bool  # Operator gate: ALLOW_CLOUD_BACKEND
    cloud_configured: bool  # Whether an API key is set (never the key itself)
    cloud_url: str  # Cloud host, for display


class CloudModelInfo(BaseModel):
    """A model offered by the cloud backend."""

    name: str
    capabilities: list[str] = Field(default_factory=list)
    supports_vision: bool = False


class CloudModelListResponse(BaseModel):
    """Response body for the cloud model listing endpoint."""

    models: list[CloudModelInfo]
    # True when the backend reported capability data at all. When False the
    # supports_vision flags are "unknown", not "no", and the UI should say so
    # instead of implying every model is text-only.
    capabilities_reported: bool


class SessionInfo(BaseModel):
    """Information about a session for debug API"""

    session_id: str
    created_at: str
    last_activity: str
    message_count: int


class SessionListResponse(BaseModel):
    """Response body for listing sessions"""

    sessions: list[SessionInfo]


class CacheClearResponse(BaseModel):
    """Response body for cache clear operation"""

    cleared_entries: int
    message: str
