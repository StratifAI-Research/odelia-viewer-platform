"""
In-memory session management for chat conversations
"""
import asyncio
import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """Represents a chat session with conversation history"""
    session_id: str
    conversation_history: List[dict] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    generation_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    active_task: Optional[asyncio.Task] = field(default=None)

    def __post_init__(self):
        # Ensure cancel_event is created if not provided
        if self.cancel_event is None:
            self.cancel_event = asyncio.Event()
        if self.generation_lock is None:
            self.generation_lock = asyncio.Lock()

    async def cancel_active_generation(self) -> None:
        """Cancel any active generation task"""
        if self.active_task and not self.active_task.done():
            logger.info(f"Cancelling active generation for session {self.session_id}")
            self.cancel_event.set()
            # Give the task a moment to notice cancellation
            try:
                await asyncio.wait_for(asyncio.shield(self.active_task), timeout=0.5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                # If it doesn't stop gracefully, force cancel
                self.active_task.cancel()
                try:
                    await self.active_task
                except asyncio.CancelledError:
                    pass
            self.active_task = None
        # Reset cancel event for next generation
        self.cancel_event.clear()


class SessionManager:
    """
    Manages chat sessions in memory.
    Sessions are keyed by session_id.
    """

    def __init__(self):
        self.sessions: Dict[str, Session] = {}

    def create_session(self, session_id: Optional[str] = None) -> Session:
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

    def get_session(self, session_id: str) -> Optional[Session]:
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

    def append_message(self, session_id: str, role: str, content) -> None:
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
        session.last_activity = datetime.now()
        logger.debug(f"Appended {role} message to session {session_id}")

    def get_history(self, session_id: str) -> List[dict]:
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

    def remove_session(self, session_id: str) -> bool:
        """
        Remove a session.

        Args:
            session_id: The session ID to remove

        Returns:
            True if removed, False if session didn't exist
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Removed session: {session_id}")
            return True
        return False

    def list_sessions(self) -> List[dict]:
        """
        List all active sessions (for debug API).

        Returns:
            List of session info dicts
        """
        result = []
        for session in self.sessions.values():
            result.append({
                "session_id": session.session_id,
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "message_count": len(session.conversation_history)
            })
        return result

    def cleanup_stale(self, max_age_minutes: int = 60) -> int:
        """
        Remove sessions that haven't been active for a while.

        Args:
            max_age_minutes: Maximum age in minutes before a session is considered stale

        Returns:
            Number of sessions removed
        """
        now = datetime.now()
        stale_ids = []

        for session_id, session in self.sessions.items():
            age = (now - session.last_activity).total_seconds() / 60
            if age > max_age_minutes:
                stale_ids.append(session_id)

        for session_id in stale_ids:
            self.remove_session(session_id)

        if stale_ids:
            logger.info(f"Cleaned up {len(stale_ids)} stale sessions")

        return len(stale_ids)


# Global session manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get the global session manager singleton"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
