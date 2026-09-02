"""
In-memory session management for chat conversations
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime


def _utcnow() -> datetime:
    return datetime.now(UTC)


logger = logging.getLogger(__name__)


@dataclass
class Session:
    """Represents a chat session with conversation history"""

    session_id: str
    conversation_history: list[dict] = field(default_factory=list)
    created_at: datetime = field(default_factory=_utcnow)
    last_activity: datetime = field(default_factory=_utcnow)
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    generation_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    active_task: asyncio.Task | None = field(default=None)
    # Set when the session is removed. A WebSocket handler holds its Session
    # directly, from before its first await, so dropping the registry entry does
    # not reach it: without this it would go on serving -- and starting new
    # generations against -- a session the caller was told is gone.
    closed: bool = field(default=False)

    def __post_init__(self) -> None:
        # Ensure cancel_event is created if not provided
        if self.cancel_event is None:
            self.cancel_event = asyncio.Event()
        if self.generation_lock is None:
            self.generation_lock = asyncio.Lock()

    async def _settled(self, task: asyncio.Task, timeout: float) -> bool:
        """Wait up to `timeout` for `task` to finish. True if it did.

        Every way the task itself can end counts as settled, including raising:
        the caller wants the generation stopped, and a generation that died on
        its way out is stopped. Letting that exception through would report a
        deletion that has already happened as a failure of the deletion.

        Cancellation of *this* coroutine is the one thing not swallowed, and the
        two are told apart by `cancelling()` -- the count of cancel requests made
        against this coroutine. Asking whether the task is done instead is
        unsound: if it is cancelled in the same turn we are, it is already done
        when we look, and our own cancellation disappears.
        """
        outer = asyncio.current_task()
        requests_before = outer.cancelling() if outer is not None else 0
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        except TimeoutError:
            return False
        except asyncio.CancelledError:
            if outer is not None and outer.cancelling() > requests_before:
                raise
        except Exception as e:
            logger.error(f"Generation for session {self.session_id} failed while stopping: {e}")
        return True

    def _release(self, task: asyncio.Task | None) -> None:
        """Let go of a finished generation and reopen the session for the next one."""
        if self.active_task is not None and self.active_task is not task:
            # Something newer has taken the session over. Neither the pointer nor
            # the flag is ours to clear: the flag belongs to whatever is running
            # now, and clearing it would tell that generation it may carry on
            # when it has just been asked to stop.
            #
            # Defence rather than a fixed bug, and untested for that reason: the
            # dispatcher serializes replacement under `generation_lock`, so the
            # only way here is a deletion racing a chat, where `closed` already
            # stops the flag being cleared. Three lines so neither of those has
            # to stay true.
            return
        self.active_task = None
        # A closed session will not have a next generation, and clearing the flag
        # there could tell a task that swallowed its cancellation that it is
        # allowed to carry on.
        if not self.closed:
            self.cancel_event.clear()

    async def cancel_active_generation(self) -> bool:
        """Stop whatever this session is generating. True if nothing is running now.

        The task is read once and held. `self.active_task` is a mutable pointer
        and this method suspends: a handler that is still attached can replace
        the task while the wait is in progress, and the force-cancel below would
        then land on a generation that had done nothing wrong.

        Both waits are bounded, because a caller must not be held by a task that
        will not stop. A task that survives both is reported by returning False,
        and it keeps both its pointer and the cancel flag: the pointer is the
        only handle anything still has on it, and the flag is the only thing
        still asking it to stop. Callers must not treat False as "stopped" --
        starting another generation beside it, or telling the reader it was
        cancelled, would both be untrue.
        """
        task = self.active_task
        if task is None or task.done():
            self._release(task)
            return True

        logger.info(f"Cancelling active generation for session {self.session_id}")
        self.cancel_event.set()
        try:
            # Ask first, then insist.
            if not await self._settled(task, timeout=0.5):
                task.cancel()
                if not await self._settled(task, timeout=0.5):
                    logger.warning(
                        f"Generation for session {self.session_id} did not stop when cancelled"
                    )
                    return False
        except asyncio.CancelledError:
            # Whoever asked for this is going away. On a removal the session has
            # already been dropped, so nothing else owns this generation --
            # leaving it streaming would be a task nobody can reach and nobody
            # will stop.
            task.cancel()
            raise
        self._release(task)
        return True


class SessionManager:
    """
    Manages chat sessions in memory.
    Sessions are keyed by session_id.
    """

    def __init__(self) -> None:
        self.sessions: dict[str, Session] = {}

    def create_session(self, session_id: str | None = None) -> Session:
        """
        Create a new session.

        Args:
            session_id: Optional session ID. If None, generates a UUID.

        Returns:
            New Session instance
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        session = Session(session_id=session_id)
        self.sessions[session_id] = session
        logger.info(f"Created new session: {session_id}")
        return session

    def get_session(self, session_id: str) -> Session | None:
        """
        Get an existing session by ID.

        Args:
            session_id: The session ID to look up

        Returns:
            Session if found, None otherwise
        """
        return self.sessions.get(session_id)

    def get_or_create_session(self, session_id: str) -> Session:
        """
        Get existing session or create new one.
        If session_id is 'new', generates a new session ID.

        Args:
            session_id: The session ID, or 'new' to create a new session

        Returns:
            Session instance (existing or newly created)
        """
        if session_id == "new":
            return self.create_session()

        existing = self.get_session(session_id)
        if existing:
            logger.debug(f"Retrieved existing session: {session_id}")
            return existing

        # Session doesn't exist, create it with the provided ID
        return self.create_session(session_id)

    def append_message(self, session_id: str, role: str, content: str | list[dict]) -> None:
        """
        Append a message to a session's conversation history.

        Args:
            session_id: The session ID
            role: Message role ('user' or 'assistant')
            content: Message content — either a plain string (text-only / assistant)
                     or a content array (user messages with interleaved images)
        """
        session = self.get_session(session_id)
        if session is None:
            logger.warning(f"Attempted to append message to non-existent session: {session_id}")
            return

        session.conversation_history.append({"role": role, "content": content})
        session.last_activity = _utcnow()
        logger.debug(f"Appended {role} message to session {session_id}")

    def get_history(self, session_id: str) -> list[dict]:
        """
        Get conversation history for a session.

        Args:
            session_id: The session ID

        Returns:
            List of message dicts, empty list if session not found
        """
        session = self.get_session(session_id)
        if session is None:
            return []
        return session.conversation_history

    async def remove_session(self, session_id: str) -> bool:
        """
        Remove a session, stopping any generation still running for it.

        The session is popped from the registry *before* the generation is
        cancelled, and the order matters: cancellation awaits, and anything that
        looked the session up while it was suspended would have obtained a
        session already promised to the caller as gone.

        Popping is not on its own enough. A WebSocket handler holds its `Session`
        from before its first await and never looks the ID up again, so it is
        told through `closed` instead. Reconnecting under the same ID afterwards
        still yields a fresh, empty session -- that is `get_or_create_session`
        working as designed, not this removal failing.

        Cancelling is not optional. A generation left running against a removed
        session goes on retrieving and preprocessing images, and its closing
        `append_message` then finds no session and silently drops the turn --
        work paid for, an answer streamed to a client, and nothing recorded.

        Args:
            session_id: The session ID to remove

        Returns:
            True if removed, False if session didn't exist
        """
        session = self.sessions.pop(session_id, None)
        if session is None:
            return False
        # Marked before the await, so a handler still holding this Session stops
        # serving it at its next message rather than at some later point of its
        # own choosing. Popping alone does not reach such a handler: it captured
        # the object before its first await and never looks the ID up again.
        session.closed = True
        await session.cancel_active_generation()
        logger.info(f"Removed session: {session_id}")
        return True

    def list_sessions(self) -> list[dict]:
        """
        List all active sessions (for debug API).

        Returns:
            List of session info dicts
        """
        result = []
        for session in self.sessions.values():
            result.append(
                {
                    "session_id": session.session_id,
                    "created_at": session.created_at.isoformat(),
                    "last_activity": session.last_activity.isoformat(),
                    "message_count": len(session.conversation_history),
                }
            )
        return result

    async def cleanup_stale(self, max_age_minutes: int = 60) -> int:
        """
        Remove sessions that haven't been active for a while.

        `last_activity` advances when a turn starts and when its result is
        recorded, not while a response streams, so a long generation can look
        stale mid-flight. Sessions with a generation running are therefore left
        alone: this is a sweep for the abandoned, and cancelling a response
        someone is watching arrive is not what an operator asks for by calling it.

        Args:
            max_age_minutes: Maximum age in minutes before a session is considered stale

        Returns:
            Number of sessions removed
        """
        now = _utcnow()
        stale = []

        for session_id, session in self.sessions.items():
            if session.active_task is not None and not session.active_task.done():
                continue
            age = (now - session.last_activity).total_seconds() / 60
            if age > max_age_minutes:
                stale.append((session_id, session))

        removed = 0
        for session_id, candidate in stale:
            # Re-checked rather than trusted: the list is walked after removals
            # that could suspend, and a session further down it could by then
            # have picked up a generation, or have been deleted and recreated
            # under the same ID by a reconnect. Sweeping the abandoned must not
            # take a live conversation with it.
            #
            # Defence rather than a fixed bug: nothing between the filter above
            # and this loop yields today, because a session with a live
            # generation is filtered out and removing one without is synchronous.
            # Four lines so that stops being load-bearing.
            if self.sessions.get(session_id) is not candidate:
                continue
            if candidate.active_task is not None and not candidate.active_task.done():
                continue
            if await self.remove_session(session_id):
                removed += 1

        if removed:
            logger.info(f"Cleaned up {removed} stale sessions")

        return removed


# Global session manager instance
_session_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    """Get the global session manager singleton"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
