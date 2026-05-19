"""
Pydantic models for WebSocket messages and debug API
"""
from pydantic import BaseModel
from typing import List, Optional
from enum import Enum


# =============================================================================
# Enums
# =============================================================================

class ClientMessageType(str, Enum):
    """Types of messages the client can send"""
    CHAT = "chat"      # User sends a message with context
    CANCEL = "cancel"  # Cancel current generation


class ServerMessageType(str, Enum):
    """Types of messages the server can send"""
    CONNECTED = "connected"           # Connection established, includes session_id
    TOKEN = "token"                   # Streaming token
    THINKING_TOKEN = "thinking_token" # Streaming reasoning/thinking token
    DONE = "done"                     # Generation complete
    ERROR = "error"                   # Error occurred
    PREPROCESSING = "preprocessing"   # Status during preprocessing


class SliceStrategy(str, Enum):
    """Slice extraction strategies for DICOM volumes"""
    CENTRAL = "central"    # Extract from central N% of volume
    UNIFORM = "uniform"    # Evenly spaced across entire volume
    FIRST_N = "first_n"    # First N slices
    LAST_N = "last_n"      # Last N slices


# =============================================================================
# WebSocket Messages
# =============================================================================cd 

class ClientMessage(BaseModel):
    """Message sent from client to server via WebSocket"""
    type: ClientMessageType
    content: Optional[str] = None            # User message text (for CHAT)
    study_uid: Optional[str] = None          # StudyInstanceUID (for CHAT)
    series_uids: Optional[List[str]] = None  # Series context for this message (for CHAT)


class ServerMessage(BaseModel):
    """Message sent from server to client via WebSocket"""
    type: ServerMessageType
    content: Optional[str] = None      # Token content, error message, or status
    session_id: Optional[str] = None   # Returned on CONNECTED
    progress: Optional[float] = None   # For preprocessing status (0.0-1.0)


# =============================================================================
# Debug API Models
# =============================================================================

class PreprocessingConfig(BaseModel):
    """Preprocessing configuration for debug API"""
    num_slices: Optional[int] = None
    slice_strategy: Optional[SliceStrategy] = None
    central_percentage: Optional[int] = None


class OllamaOptionsConfig(BaseModel):
    """OpenAI-compatible generation options for debug API.
    Only fields supported by /v1/chat/completions."""
    max_tokens: Optional[int] = None        # Max tokens to generate
    temperature: Optional[float] = None     # Sampling temperature
    top_p: Optional[float] = None           # Top-p (nucleus) sampling
    stop: Optional[List[str]] = None        # Stop sequences
    seed: Optional[int] = None              # Random seed for reproducibility


class DebugConfigUpdate(BaseModel):
    """Request body for updating debug configuration"""
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    preprocessing: Optional[PreprocessingConfig] = None
    ollama_options: Optional[OllamaOptionsConfig] = None


class DebugConfigResponse(BaseModel):
    """Response body for debug configuration endpoint"""
    system_prompt: str
    model: str               # Active model name (runtime-adjustable)
    preprocessing: dict
    ollama_static: dict      # Read-only: url, backend_type (from env vars)
    ollama_options: dict     # Adjustable: max_tokens, temperature, top_p, stop, seed


class SessionInfo(BaseModel):
    """Information about a session for debug API"""
    session_id: str
    created_at: str
    last_activity: str
    message_count: int


class SessionListResponse(BaseModel):
    """Response body for listing sessions"""
    sessions: List[SessionInfo]


class CacheClearResponse(BaseModel):
    """Response body for cache clear operation"""
    cleared_entries: int
    message: str
