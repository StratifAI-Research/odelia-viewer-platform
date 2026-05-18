"""
Debug REST API endpoints for development and testing.
Should be disabled or protected in production.
"""
import logging
from fastapi import APIRouter, HTTPException

from models import (
    DebugConfigUpdate,
    DebugConfigResponse,
    SessionListResponse,
    SessionInfo,
    CacheClearResponse
)
from config import get_config
from runtime_config import get_runtime_config
from session_manager import get_session_manager
from image_cache import get_image_cache
from ollama_client import get_ollama_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/debug", tags=["debug"])


@router.get("/config", response_model=DebugConfigResponse)
async def get_debug_config() -> DebugConfigResponse:
    """
    Get current configuration including system prompt, preprocessing, and Ollama settings.
    """
    config = get_config()
    runtime_config = get_runtime_config()

    return DebugConfigResponse(
        system_prompt=runtime_config.system_prompt,
        model=runtime_config.model,
        preprocessing=runtime_config.to_dict()["preprocessing"],
        ollama_static={
            "url": config.ollama_url,
        },
        ollama_options=runtime_config.ollama_options.to_full_dict()
    )


@router.put("/config", response_model=DebugConfigResponse)
async def update_debug_config(update: DebugConfigUpdate) -> DebugConfigResponse:
    """
    Update configuration at runtime.

    Accepts partial updates - only provided fields are changed.
    """
    runtime_config = get_runtime_config()

    # Convert preprocessing config to dict if provided
    preprocessing_dict = None
    if update.preprocessing:
        preprocessing_dict = update.preprocessing.model_dump(exclude_none=True)

    # Convert ollama_options config to dict if provided
    ollama_options_dict = None
    if update.ollama_options:
        ollama_options_dict = update.ollama_options.model_dump(exclude_none=True)

    # Apply updates
    runtime_config.update(
        system_prompt=update.system_prompt,
        model=update.model,
        preprocessing=preprocessing_dict,
        ollama_options=ollama_options_dict
    )

    # Auto-clear image cache when preprocessing params change
    if preprocessing_dict:
        cleared = get_image_cache().clear()
        logger.info(f"Auto-cleared {cleared} cache entries after preprocessing config change")

    logger.info(f"Updated runtime config: system_prompt={'changed' if update.system_prompt else 'unchanged'}, "
                f"model={'changed' if update.model else 'unchanged'}, "
                f"preprocessing={'changed' if preprocessing_dict else 'unchanged'}, "
                f"ollama_options={'changed' if ollama_options_dict else 'unchanged'}")

    # Return updated config
    config = get_config()
    return DebugConfigResponse(
        system_prompt=runtime_config.system_prompt,
        model=runtime_config.model,
        preprocessing=runtime_config.to_dict()["preprocessing"],
        ollama_static={
            "url": config.ollama_url,
        },
        ollama_options=runtime_config.ollama_options.to_full_dict()
    )


@router.delete("/cache", response_model=CacheClearResponse)
async def clear_cache() -> CacheClearResponse:
    """
    Clear the image cache.

    Useful when testing preprocessing changes.
    """
    image_cache = get_image_cache()
    cleared = image_cache.clear()

    logger.info(f"Cleared image cache: {cleared} entries removed")

    return CacheClearResponse(
        cleared_entries=cleared,
        message=f"Cleared {cleared} cached series"
    )


@router.get("/cache/stats")
async def get_cache_stats() -> dict:
    """
    Get image cache statistics.
    """
    image_cache = get_image_cache()
    return image_cache.stats()


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions() -> SessionListResponse:
    """
    List all active sessions.
    """
    session_manager = get_session_manager()
    sessions_data = session_manager.list_sessions()

    sessions = [
        SessionInfo(
            session_id=s["session_id"],
            created_at=s["created_at"],
            last_activity=s["last_activity"],
            message_count=s["message_count"]
        )
        for s in sessions_data
    ]

    return SessionListResponse(sessions=sessions)


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> dict:
    """
    Delete a specific session.
    """
    session_manager = get_session_manager()

    if session_manager.remove_session(session_id):
        logger.info(f"Deleted session via debug API: {session_id}")
        return {"message": f"Session {session_id} deleted"}
    else:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")


@router.post("/sessions/cleanup")
async def cleanup_sessions(max_age_minutes: int = 60) -> dict:
    """
    Clean up stale sessions.

    Args:
        max_age_minutes: Maximum session age in minutes (default: 60)
    """
    session_manager = get_session_manager()
    removed = session_manager.cleanup_stale(max_age_minutes)

    logger.info(f"Cleaned up {removed} stale sessions (max age: {max_age_minutes} min)")

    return {
        "removed_sessions": removed,
        "max_age_minutes": max_age_minutes
    }


@router.get("/health")
async def debug_health() -> dict:
    """
    Detailed health check including Ollama connectivity.
    """
    config = get_config()
    ollama_client = get_ollama_client()
    image_cache = get_image_cache()
    session_manager = get_session_manager()

    # Check Ollama
    ollama_healthy = await ollama_client.health_check()
    ollama_models = []
    if ollama_healthy:
        ollama_models = await ollama_client.list_models()

    return {
        "status": "healthy" if ollama_healthy else "degraded",
        "ollama": {
            "url": config.ollama_url,
            "model": config.ollama_model,
            "connected": ollama_healthy,
            "available_models": ollama_models
        },
        "cache": image_cache.stats(),
        "sessions": {
            "active": len(session_manager.sessions)
        }
    }
