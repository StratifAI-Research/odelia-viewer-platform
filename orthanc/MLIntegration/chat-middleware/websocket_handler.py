"""
WebSocket handler for chat sessions
"""

import asyncio
import contextlib
import logging
from datetime import UTC, datetime

from config import get_config
from fastapi import WebSocket, WebSocketDisconnect
from image_cache import CachedSeries, get_image_cache
from models import ClientMessage, ClientMessageType, ServerMessageType
from ollama_client import CloudBackendUnavailableError, get_client_for_provider
from preprocessing import preprocess_series
from prompt_builder import get_prompt_builder
from pydantic import ValidationError
from runtime_config import get_runtime_config
from session_manager import Session, get_session_manager

logger = logging.getLogger(__name__)


async def send_message(websocket: WebSocket, msg_type: ServerMessageType, **kwargs: object) -> None:
    """Send a typed message to the client"""
    message = {"type": msg_type.value, **kwargs}
    await websocket.send_json(message)


async def handle_websocket(websocket: WebSocket, session_id: str) -> None:
    """
    Handle WebSocket connection for a chat session.

    Args:
        websocket: FastAPI WebSocket connection
        session_id: Session ID from URL path ('new' to create new session)
    """
    session_manager = get_session_manager()

    # Get or create session (if session_id is "new", generate new ID)
    session = session_manager.get_or_create_session(session_id)

    try:
        await websocket.accept()
        logger.info(f"WebSocket connected for session: {session.session_id}")

        # Send connection confirmation with actual session_id
        await send_message(websocket, ServerMessageType.CONNECTED, session_id=session.session_id)

        # Process incoming messages - DO NOT await tasks here to allow cancellation
        async for message in websocket.iter_json():
            # Validate the untrusted payload in isolation: a malformed/unexpected
            # message must not crash the handler or drop the connection.
            try:
                msg = ClientMessage(**message)
            except (ValidationError, TypeError) as e:
                logger.warning(f"Rejected malformed client message: {e}")
                await send_message(
                    websocket, ServerMessageType.ERROR, content=f"Invalid message: {e!s}"
                )
                continue

            try:
                if msg.type == ClientMessageType.CHAT:
                    # Cancel any existing generation before starting new one
                    if session.active_task and not session.active_task.done():
                        logger.info(
                            f"Cancelling previous generation for session {session.session_id}"
                        )
                        session.cancel_event.set()
                        # Wait briefly for task to notice cancellation
                        try:
                            await asyncio.wait_for(session.active_task, timeout=1.0)
                        except (TimeoutError, asyncio.CancelledError):
                            session.active_task.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await session.active_task
                        session.active_task = None

                    # Reset cancel event for new generation
                    session.cancel_event.clear()

                    # Create task but DON'T await it - let it run in background
                    task = asyncio.create_task(
                        handle_chat(
                            websocket,
                            session,
                            msg.content or "",
                            msg.study_uid or "",
                            msg.series_uids or [],
                        )
                    )
                    session.active_task = task

                    # Add callback to clean up when task completes
                    def task_done_callback(t: asyncio.Task) -> None:
                        if session.active_task == t:
                            session.active_task = None
                        if t.exception():
                            logger.error(f"Chat task failed: {t.exception()}")

                    task.add_done_callback(task_done_callback)
                    # Don't await - continue processing messages immediately

                elif msg.type == ClientMessageType.CANCEL:
                    logger.info(f"Cancellation requested for session {session.session_id}")
                    if session.active_task and not session.active_task.done():
                        session.cancel_event.set()
                        # Wait briefly for graceful cancellation
                        try:
                            await asyncio.wait_for(session.active_task, timeout=1.0)
                        except (TimeoutError, asyncio.CancelledError):
                            session.active_task.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await session.active_task
                        session.active_task = None
                        await send_message(websocket, ServerMessageType.DONE, content="Cancelled")

            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await send_message(
                    websocket, ServerMessageType.ERROR, content=f"Error processing message: {e!s}"
                )

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {session.session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session.session_id}: {e}")
    finally:
        # Cancel any running task on disconnect
        if session.active_task and not session.active_task.done():
            session.cancel_event.set()
            session.active_task.cancel()
        logger.debug(f"WebSocket handler finished for session: {session.session_id}")


async def handle_chat(
    websocket: WebSocket, session: Session, content: str, study_uid: str, series_uids: list[str]
) -> None:
    """
    Handle a chat message with study and series context.

    Args:
        websocket: WebSocket connection
        session: Current session
        content: User message content
        study_uid: StudyInstanceUID
        series_uids: List of SeriesInstanceUIDs for context
    """
    config = get_config()
    runtime_config = get_runtime_config()
    image_cache = get_image_cache()
    prompt_builder = get_prompt_builder()
    session_manager = get_session_manager()

    # Resolve the backend before any DICOM is fetched. A misconfigured cloud
    # provider should fail here rather than after retrieving and preprocessing a
    # series -- and, for the cloud case, before any image data has been assembled.
    try:
        ollama_client = get_client_for_provider(
            runtime_config.provider.value, runtime_config.active_model
        )
    except CloudBackendUnavailableError as e:
        logger.warning(f"Cloud backend unavailable for session {session.session_id}: {e}")
        await send_message(websocket, ServerMessageType.ERROR, content=str(e))
        return

    # Check if cancelled before we even start
    if session.cancel_event.is_set():
        logger.info(f"Chat already cancelled for session {session.session_id}")
        return

    session.last_activity = datetime.now(UTC)

    try:
        # 1. Ensure all requested series are cached (preprocess if needed)
        series_images = {}
        total = len(series_uids) if series_uids else 0

        for i, series_uid in enumerate(series_uids):
            # Check for cancellation
            if session.cancel_event.is_set():
                logger.info("Chat cancelled during preprocessing")
                return

            if not image_cache.has(series_uid):
                # Send preprocessing status
                progress = (i / total) if total > 0 else 0
                await send_message(
                    websocket,
                    ServerMessageType.PREPROCESSING,
                    content=f"Retrieving series {series_uid}...",
                    progress=progress,
                )

                # Preprocess and cache (uses RuntimeConfig for slice params)
                try:
                    images = await preprocess_series(
                        series_uid,
                        study_uid,
                        runtime_config.preprocessing,
                        config.wado_base_url,
                        config.image_folder,
                    )

                    image_cache.put(
                        series_uid,
                        CachedSeries(
                            series_uid=series_uid,
                            base64_images=images,
                            created_at=datetime.now(UTC),
                            last_accessed=datetime.now(UTC),
                        ),
                    )
                except Exception as e:
                    logger.error(f"Failed to preprocess series {series_uid}: {e}")
                    await send_message(
                        websocket,
                        ServerMessageType.ERROR,
                        content=f"Failed to retrieve series {series_uid}: {e!s}",
                    )
                    return

            # Get images from cache
            cached = image_cache.get(series_uid)
            if cached:
                series_images[series_uid] = cached.base64_images

        # Update preprocessing progress to complete
        if total > 0:
            await send_message(
                websocket,
                ServerMessageType.PREPROCESSING,
                content="Series ready, generating response...",
                progress=1.0,
            )

        # 2. Build messages and get user content for history storage
        messages, user_content_for_history = prompt_builder.build_messages(
            session.conversation_history, content, series_images
        )

        # 3. Stream response from Ollama
        full_response = ""
        try:
            # Model already resolved from the active provider above; do not
            # reassign it here or the cloud selection would be overwritten with
            # the local model tag.
            ollama_runtime_options = runtime_config.ollama_options.to_dict()

            async for chunk in ollama_client.chat_stream(
                messages, session.cancel_event, runtime_options=ollama_runtime_options
            ):
                if session.cancel_event.is_set():
                    logger.info("Chat cancelled during generation")
                    break

                if chunk["type"] == "thinking":
                    await send_message(
                        websocket, ServerMessageType.THINKING_TOKEN, content=chunk["text"]
                    )
                else:
                    full_response += chunk["text"]
                    await send_message(websocket, ServerMessageType.TOKEN, content=chunk["text"])
        except Exception as e:
            logger.error(f"Ollama streaming error: {e}")
            await send_message(
                websocket, ServerMessageType.ERROR, content=f"Error during generation: {e!s}"
            )
            return

        # 4. Store in conversation history (only if not cancelled and has content)
        if not session.cancel_event.is_set() and full_response:
            session_manager.append_message(session.session_id, "user", user_content_for_history)
            session_manager.append_message(session.session_id, "assistant", full_response)

        # 5. Signal completion
        await send_message(websocket, ServerMessageType.DONE)

    except Exception as e:
        logger.error(f"Error in handle_chat: {e}")
        import traceback

        traceback.print_exc()
        await send_message(websocket, ServerMessageType.ERROR, content=f"Unexpected error: {e!s}")
