"""
Debug REST API endpoints for development and testing.
Should be disabled or protected in production.
"""

import logging

from config import get_config
from fastapi import APIRouter, HTTPException
from image_cache import get_image_cache
from models import (
    CacheClearResponse,
    CloudModelListResponse,
    DebugConfigResponse,
    DebugConfigUpdate,
    Provider,
    SessionInfo,
    SessionListResponse,
)
from ollama_client import (
    CloudBackendUnavailableError,
    ModelListError,
    build_cloud_client,
    get_ollama_client,
)
from runtime_config import get_runtime_config
from session_manager import get_session_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/debug", tags=["debug"])


def _build_config_response() -> DebugConfigResponse:
    """Assemble the config response from static + runtime config.

    Shared by GET and PUT so the two can never drift into reporting different
    shapes for the same state.
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
        ollama_options=runtime_config.ollama_options.to_full_dict(),
        provider=runtime_config.provider,
        cloud_model=runtime_config.cloud_model,
        active_model=runtime_config.active_model,
        cloud_enabled=config.allow_cloud_backend,
        cloud_configured=bool(config.ollama_cloud_api_key),
        cloud_url=config.ollama_cloud_url,
    )


@router.get("/config", response_model=DebugConfigResponse)
async def get_debug_config() -> DebugConfigResponse:
    """
    Get current configuration including system prompt, preprocessing, and Ollama settings.
    """
    return _build_config_response()


@router.put("/config", response_model=DebugConfigResponse)
async def update_debug_config(update: DebugConfigUpdate) -> DebugConfigResponse:
    """
    Update configuration at runtime.

    Accepts partial updates - only provided fields are changed.
    """
    runtime_config = get_runtime_config()
    config = get_config()

    # Refuse to route to the cloud unless the operator enabled it and a key exists.
    # Checked before applying anything so a rejected request changes no state --
    # otherwise a caller could leave the service on a cloud provider it cannot use.
    if update.provider == Provider.CLOUD:
        if not config.allow_cloud_backend:
            raise HTTPException(
                status_code=403,
                detail=(
                    "The Ollama Cloud backend is disabled. An operator must set "
                    "ALLOW_CLOUD_BACKEND=1 on the chat-middleware service."
                ),
            )
        if not config.ollama_cloud_api_key:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No Ollama Cloud API key is configured. Set OLLAMA_API_KEY on "
                    "the chat-middleware service."
                ),
            )
        # A cloud provider with no model would fail on the next chat instead of here.
        if not (update.cloud_model or runtime_config.cloud_model):
            raise HTTPException(
                status_code=400,
                detail="Select a cloud model before switching to the cloud backend.",
            )

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
        ollama_options=ollama_options_dict,
        provider=update.provider.value if update.provider else None,
        cloud_model=update.cloud_model,
    )

    # Auto-clear image cache when preprocessing params change.
    #
    # No longer load-bearing for correctness: cache keys carry the recipe, so an
    # entry made under the old params simply misses rather than being served for
    # the new ones. Kept because those entries can never be hit again either, and
    # dropping them frees the memory immediately.
    if preprocessing_dict:
        cleared = get_image_cache().clear()
        logger.info(f"Auto-cleared {cleared} cache entries after preprocessing config change")

    logger.info(
        f"Updated runtime config: system_prompt={'changed' if update.system_prompt else 'unchanged'}, "
        f"model={'changed' if update.model else 'unchanged'}, "
        f"preprocessing={'changed' if preprocessing_dict else 'unchanged'}, "
        f"ollama_options={'changed' if ollama_options_dict else 'unchanged'}, "
        f"provider={runtime_config.provider.value}"
    )

    return _build_config_response()


@router.get("/cloud/models", response_model=CloudModelListResponse)
async def list_cloud_models() -> CloudModelListResponse:
    """
    List the models available from the cloud backend, with vision capability.

    The API key stays server-side: the browser asks this endpoint, which queries
    the cloud host itself. The key is never included in the response.

    Roughly half of Ollama's cloud models are text-only, and this chat sends
    DICOM slices as images, so `supports_vision` is the flag the panel needs to
    stop a user picking a model that cannot see the study at all.
    """
    try:
        # No model needed just to list; pass a placeholder so the "no model
        # selected" guard in build_cloud_client does not reject the listing.
        client = build_cloud_client(model="__listing__")
    except CloudBackendUnavailableError as e:
        # 403: the operator has not enabled/configured cloud. Not the caller's fault
        # and not fixable by retrying, so say so rather than returning an empty list.
        raise HTTPException(status_code=403, detail=str(e)) from e

    try:
        models = await client.list_models_detailed()
    except ModelListError as e:
        # 502: we reached out and the upstream refused (bad key, network, outage).
        raise HTTPException(status_code=502, detail=str(e)) from e

    capabilities_reported = any(m["capabilities"] for m in models)
    if models and not capabilities_reported:
        logger.info(
            "Cloud host reported no model capabilities; vision flags are unknown "
            "(older Ollama versions omit the field)"
        )

    return CloudModelListResponse(models=models, capabilities_reported=capabilities_reported)


@router.get("/local/models", response_model=CloudModelListResponse)
async def list_local_models() -> CloudModelListResponse:
    """
    List the models the local Ollama server actually has pulled.

    The panel used to take the local model as free text, which meant a typo, or a
    model that was never pulled, failed only when a message was sent — and failed
    as an opaque backend error rather than as "that model is not here". Asking the
    server what it has turns that into a list.

    Same shape as the cloud listing on purpose: to the reader they are one
    catalogue of models with two sources.
    """
    client = get_ollama_client()

    try:
        models = await client.list_models_detailed()
    except ModelListError as e:
        # 502 rather than an empty list: an unreachable Ollama and an Ollama with
        # no models pulled need different actions, and an empty list would say the
        # second when the truth is the first.
        raise HTTPException(status_code=502, detail=str(e)) from e

    capabilities_reported = any(m["capabilities"] for m in models)
    return CloudModelListResponse(models=models, capabilities_reported=capabilities_reported)


@router.delete("/cache", response_model=CacheClearResponse)
async def clear_cache() -> CacheClearResponse:
    """
    Clear the image cache.

    Useful when testing preprocessing changes.
    """
    image_cache = get_image_cache()
    cleared = image_cache.clear()

    logger.info(f"Cleared image cache: {cleared} entries removed")

    return CacheClearResponse(cleared_entries=cleared, message=f"Cleared {cleared} cached series")


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
            message_count=s["message_count"],
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

    return {"removed_sessions": removed, "max_age_minutes": max_age_minutes}


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
            "available_models": ollama_models,
        },
        "cache": image_cache.stats(),
        "sessions": {"active": len(session_manager.sessions)},
    }
