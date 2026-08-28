"""
WebSocket handler for chat sessions
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime

from config import get_config
from fastapi import WebSocket, WebSocketDisconnect
from image_cache import CachedSeries, get_image_cache, make_cache_key
from models import ClientMessage, ClientMessageType, ServerMessageType, SliceSelection
from ollama_client import CloudBackendUnavailableError, get_client_for_provider
from preprocessing import SliceSelectionError, preprocess_series, recipe_signature
from prompt_builder import get_prompt_builder
from pydantic import ValidationError
from runtime_config import get_runtime_config
from session_manager import Session, get_session_manager

logger = logging.getLogger(__name__)


async def send_message(websocket: WebSocket, msg_type: ServerMessageType, **kwargs: object) -> None:
    """Send a typed message to the client"""
    message = {"type": msg_type.value, **kwargs}
    await websocket.send_json(message)


def make_task_done_callback(session: Session) -> Callable[[asyncio.Task], None]:
    """Release the session's hold on a finished generation, and report how it ended.

    A module-level factory rather than a closure in the dispatcher so the
    cancelled case can be tested without driving a socket to the point of
    forcing one.

    `Task.exception()` re-raises on a cancelled task, so cancellation has to be
    answered before asking. Reachable through the CANCEL message, and now
    through session deletion too, which cancels whatever the session was
    generating; without the guard the cleanup callback itself raises inside the
    event loop, where nothing is waiting to handle it.
    """

    def task_done_callback(t: asyncio.Task) -> None:
        if session.active_task is t:
            session.active_task = None
        if t.cancelled():
            logger.info(f"Chat task cancelled for session {session.session_id}")
            return
        if t.exception():
            logger.error(f"Chat task failed: {t.exception()}")

    return task_done_callback


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

    # The generation THIS socket started, if any. The session is shared -- the
    # same object is handed to anyone who asks for the ID -- so `active_task` is
    # not a record of what this handler is responsible for.
    owned_task: asyncio.Task | None = None

    try:
        await websocket.accept()
        logger.info(f"WebSocket connected for session: {session.session_id}")

        # Send connection confirmation with actual session_id
        await send_message(websocket, ServerMessageType.CONNECTED, session_id=session.session_id)

        # Process incoming messages - DO NOT await tasks here to allow cancellation
        async for message in websocket.iter_json():
            # This handler captured its Session before its first await, so
            # dropping the registry entry does not reach it. Without this check a
            # deleted session would go on being served -- and go on accumulating
            # history and image references -- through a connection that was
            # already open when the delete arrived.
            if session.closed:
                logger.info(f"Session {session.session_id} was deleted; closing its connection")
                break

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
                    # Replacing a generation is four steps with awaits between
                    # them, and a session is not private to one socket:
                    # `get_or_create_session` hands the same object to anyone who
                    # asks for the ID. Unserialized, two handlers can both get
                    # past the cancel and both create a generation, leaving two
                    # streaming into one conversation with only the later one
                    # referenced -- so the other cannot even be cancelled.
                    still_stopping = False
                    was_deleted = False
                    async with session.generation_lock:
                        # Cancel any existing generation before starting a new
                        # one. Through the session's own method rather than
                        # open-coded here: it holds the task it is stopping
                        # rather than re-reading a field that a concurrent delete
                        # can clear under it, bounds both of its waits so a
                        # generation that swallows cancellation cannot hold this
                        # loop, and does not mistake cancellation of this handler
                        # for the generation stopping.
                        if not await session.cancel_active_generation():
                            # It would not stop. Starting another one beside it
                            # would put two generations on one socket, both
                            # streaming to the same reader, and let the loser of
                            # that race write its turn into the history.
                            still_stopping = True
                        # Asked again, because the wait above suspends. A delete
                        # arriving during it would find this handler about to
                        # start a generation on a session already reported gone
                        # -- and, since reconnecting under the same ID makes a
                        # fresh session, that generation could go on to write its
                        # turn into someone else's conversation.
                        elif session.closed:
                            was_deleted = True
                        else:
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
                                    msg.slice_selections or [],
                                )
                            )
                            session.active_task = task
                            owned_task = task

                            task.add_done_callback(make_task_done_callback(session))
                            # Don't await - continue processing messages immediately

                    # Said outside the lock. A send waits on the socket, and this
                    # socket's reader stalling must not hold up another one's
                    # turn on the same session.
                    if still_stopping:
                        await send_message(
                            websocket,
                            ServerMessageType.ERROR,
                            content="The previous response is still stopping; please try again.",
                        )
                        continue
                    if was_deleted:
                        logger.info(
                            f"Session {session.session_id} was deleted; closing its connection"
                        )
                        break

                elif msg.type == ClientMessageType.CANCEL:
                    logger.info(f"Cancellation requested for session {session.session_id}")
                    had_generation = False
                    stopped = False
                    async with session.generation_lock:
                        if session.active_task and not session.active_task.done():
                            had_generation = True
                            stopped = await session.cancel_active_generation()

                    if had_generation and stopped:
                        await send_message(websocket, ServerMessageType.DONE, content="Cancelled")
                    elif had_generation:
                        # Not "Cancelled": that would be a claim about a
                        # generation still streaming. Not an error frame either --
                        # the panel has already shown the turn as cancelled, and a
                        # late error on top of that says three contradictory
                        # things about one message. The honest record is here; the
                        # flag stays set and the task keeps its pointer, so
                        # nothing will start beside it.
                        logger.warning(
                            f"Cancellation for session {session.session_id} "
                            "did not take; the generation is still running"
                        )

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
        # Cancel the generation THIS socket started, and only that one. The
        # session is shared, so `active_task` can belong to another connection --
        # ending someone else's answer because this socket went away would be a
        # conversation cut off by a disconnection somewhere else entirely.
        if owned_task is not None and not owned_task.done():
            if session.active_task is owned_task:
                session.cancel_event.set()
            owned_task.cancel()
        logger.debug(f"WebSocket handler finished for session: {session.session_id}")


def plan_series(
    series_uids: list[str], selections: list[SliceSelection]
) -> list[tuple[str, SliceSelection | None]]:
    """
    What to preprocess for one message, as (series, selection) pairs.

    One entry per selection rather than one per series. OHIF can split a series
    into several display sets and the panel sends a selection for each, which may
    legitimately differ -- a region of interest on one and not the other, say.
    Collapsing them to one selection per series either drops a crop or applies it
    to images it was never drawn on, and the transcript stays per-display-set
    either way, so its provenance would disagree with what the model saw.

    Two identical requests for one series are still collapsed: they would produce
    the same pixels, and the cache would serve the second from the first anyway.

    A series with no selection at all contributes one entry with None, which is
    the pre-existing behaviour.

    Args:
        series_uids: Series attached to the message, in display order
        selections: Per-display-set selections, in display order

    Returns:
        Pairs to preprocess, in order, without duplicates
    """
    plan: list[tuple[str, SliceSelection | None]] = []
    seen: set[tuple[str, str]] = set()

    for selection in selections:
        # `model_dump_json` is a faithful identity: two selections are the same
        # request exactly when every field matches, ROI included.
        fingerprint = (selection.series_uid, selection.model_dump_json())
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        plan.append((selection.series_uid, selection))

    selected_series = {selection.series_uid for selection in selections}
    for series_uid in series_uids:
        if series_uid in selected_series or any(uid == series_uid for uid, _ in plan):
            continue
        plan.append((series_uid, None))
        selected_series.add(series_uid)

    return plan


async def handle_chat(
    websocket: WebSocket,
    session: Session,
    content: str,
    study_uid: str,
    series_uids: list[str],
    slice_selections: list[SliceSelection] | None = None,
) -> None:
    """
    Handle a chat message with study and series context.

    Args:
        websocket: WebSocket connection
        session: Current session
        content: User message content
        study_uid: StudyInstanceUID
        series_uids: List of SeriesInstanceUIDs for context
        slice_selections: Per-series slice selections for THIS message. A series
            with no entry falls back to the runtime preprocessing recipe, which is
            what a viewer predating the field sends.
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

    # What to preprocess, as (series, selection) pairs -- one per selection, so a
    # series split across display sets keeps each one's own slices and crop.
    plan = plan_series(series_uids, slice_selections or [])

    # One read of the runtime preprocessing config, taken before any await.
    # `RuntimeConfig.update` mutates its params in place, so a PUT arriving while
    # this turn is suspended would otherwise change the recipe *after* the cache
    # key was computed from it -- storing one recipe's images under another's key.
    base_params = replace(runtime_config.preprocessing)

    try:
        # 1. Ensure everything the message asks for is cached (preprocess if needed).
        #
        # Keyed by cache key, not by series: one series can appear more than once
        # under different recipes, and a dict keyed on the series UID would keep
        # only the last -- sending fewer images than the message asked for.
        images_by_key: dict[str, list[str]] = {}
        total = len(plan)

        for i, (series_uid, selection) in enumerate(plan):
            # Check for cancellation
            if session.cancel_event.is_set():
                logger.info("Chat cancelled during preprocessing")
                return

            # Keyed on series AND recipe: two messages can name the same series and
            # mean different slices, and a series-only key would answer the second
            # with the first one's images.
            cache_key = make_cache_key(series_uid, recipe_signature(base_params, selection))

            if not image_cache.has(cache_key):
                # Send preprocessing status
                progress = (i / total) if total > 0 else 0
                await send_message(
                    websocket,
                    ServerMessageType.PREPROCESSING,
                    content=f"Retrieving series {series_uid}...",
                    progress=progress,
                )

                # Preprocess and cache (RuntimeConfig applies unless the message
                # named its own slices)
                try:
                    images = await preprocess_series(
                        series_uid,
                        study_uid,
                        base_params,
                        config.wado_base_url,
                        config.image_folder,
                        selection=selection,
                    )

                    image_cache.put(
                        cache_key,
                        CachedSeries(
                            series_uid=series_uid,
                            base64_images=images,
                            created_at=datetime.now(UTC),
                            last_accessed=datetime.now(UTC),
                        ),
                    )
                except SliceSelectionError as e:
                    # Retrieval worked; the requested slices are not in the series.
                    # Say that, rather than blaming the retrieval.
                    logger.warning(f"Unresolvable slice selection for {series_uid}: {e}")
                    await send_message(websocket, ServerMessageType.ERROR, content=str(e))
                    return
                except Exception as e:
                    logger.error(f"Failed to preprocess series {series_uid}: {e}")
                    await send_message(
                        websocket,
                        ServerMessageType.ERROR,
                        content=f"Failed to retrieve series {series_uid}: {e!s}",
                    )
                    return

            # Get images from cache
            cached = image_cache.get(cache_key)
            if cached:
                images_by_key[cache_key] = cached.base64_images

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
            session.conversation_history, content, images_by_key
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
        #
        # `closed` is checked as well as the cancel event, and it is checked on
        # the session object rather than by ID: reconnecting under a deleted ID
        # produces a *different* session under the same name, so a generation
        # that outlived its deletion would otherwise append this turn -- images
        # and all -- to a stranger's conversation.
        if not session.closed and not session.cancel_event.is_set() and full_response:
            session_manager.append_message(session.session_id, "user", user_content_for_history)
            session_manager.append_message(session.session_id, "assistant", full_response)

        # 5. Signal completion
        await send_message(websocket, ServerMessageType.DONE)

    except Exception as e:
        logger.error(f"Error in handle_chat: {e}")
        import traceback

        traceback.print_exc()
        await send_message(websocket, ServerMessageType.ERROR, content=f"Unexpected error: {e!s}")
