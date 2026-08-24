"""In-memory conversation session state.

Deliberately simple: this assignment explicitly doesn't need auth or a
real datastore. A Session just holds this conversation's Claude-format
message history (capped) plus a couple of scalars used to build a
retrieval query that's aware of the previous turn (for "What about
Canada?" style follow-ups).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app import config


@dataclass
class Session:
    session_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    last_user_message: str = ""
    turn_index: int = 0

    def add_user_message(self, content: Any) -> None:
        self.messages.append({"role": "user", "content": content})
        self._trim()

    def add_assistant_message(self, content: Any) -> None:
        self.messages.append({"role": "assistant", "content": content})
        self._trim()

    def add_raw_message(self, message: dict[str, Any]) -> None:
        """Append an already-formed message dict (used for assistant
        tool-call messages and tool-result messages, which carry extra
        fields beyond role/content)."""
        self.messages.append(message)
        self._trim()

    def _trim(self) -> None:
        max_messages = config.MAX_HISTORY_TURNS * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def build_retrieval_query(self, current_message: str) -> str:
        """Fold in the previous user turn so short follow-ups like
        "What about Canada?" retrieve against the right topic instead of
        being treated as an unrelated, under-specified query."""
        if self.last_user_message and self.last_user_message != current_message:
            return f"{self.last_user_message}\n{current_message}"
        return current_message


class SessionStore:
    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def get_or_create(self, session_id: str) -> Session:
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(session_id=session_id)
        return self._sessions[session_id]
