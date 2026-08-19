from __future__ import annotations
import logging
import threading

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)

# How many messages to keep per session. Counted in MESSAGES, so 12 is roughly
# 6 back-and-forth turns. Raise for longer context, lower for cheaper calls.
MAX_HISTORY_MESSAGES = 12


class WindowedChatMessageHistory(BaseChatMessageHistory):
    """In-memory chat history that keeps only the last N messages.

    Implements LangChain's `BaseChatMessageHistory` interface: `messages`,
    `add_messages`, and `clear`. Because it satisfies that interface,
    `RunnableWithMessageHistory` can use it without knowing anything about how
    it stores data.
    """

    def __init__(self, session_id: str, max_messages: int = MAX_HISTORY_MESSAGES):
        self.session_id = session_id
        self.max_messages = max_messages
        self._messages: list[BaseMessage] = []

    @property
    def messages(self) -> list[BaseMessage]:
        return list(self._messages)  # copy, so callers cannot mutate our state

    def add_messages(self, messages: list[BaseMessage]) -> None:
        self._messages.extend(messages)

        if len(self._messages) > self.max_messages:
            dropped = len(self._messages) - self.max_messages
            # Slice from the END: keep the most recent, discard the oldest.
            self._messages = self._messages[-self.max_messages:]
            logger.debug("Session %s: trimmed %d old messages", self.session_id, dropped)

    def clear(self) -> None:
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)


# session_id -> history. A plain dict guarded by a lock, because FastAPI serves
# requests from multiple threads and dict mutation from several threads at once
# is not safe.
_SESSION_STORE: dict[str, WindowedChatMessageHistory] = {}
_STORE_LOCK = threading.Lock()


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Return (creating if needed) the history for one conversation.

    THIS IS THE FUNCTION `RunnableWithMessageHistory` CALLS. Its signature --
    takes a session_id, returns a BaseChatMessageHistory -- is fixed by
    LangChain. Everything else about the storage is up to us.
    """
    with _STORE_LOCK:
        if session_id not in _SESSION_STORE:
            _SESSION_STORE[session_id] = WindowedChatMessageHistory(session_id)
            logger.info("Started new conversation session: %s", session_id)
        return _SESSION_STORE[session_id]


def clear_session(session_id: str) -> bool:
    """Wipe one conversation. Returns True if the session existed."""
    with _STORE_LOCK:
        history = _SESSION_STORE.pop(session_id, None)
    # `is not None`, NOT a truthiness check. This class defines __len__, so an
    # empty history is FALSY -- `if history:` would report "did not exist" for
    # any session that was created but never used. A subtle Python gotcha worth
    # remembering: defining __len__ silently changes how your object behaves in
    # a boolean context.
    if history is not None:
        history.clear()
        logger.info("Cleared session: %s", session_id)
        return True
    return False


def session_message_count(session_id: str) -> int:
    """How many messages are currently stored for a session."""
    with _STORE_LOCK:
        history = _SESSION_STORE.get(session_id)
    return len(history) if history else 0


def active_sessions() -> list[str]:
    """IDs of all sessions currently held in memory."""
    with _STORE_LOCK:
        return list(_SESSION_STORE.keys())
