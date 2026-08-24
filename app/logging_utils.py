"""Structured, per-turn trace logging.

Each turn produces one JSON line containing everything the assignment's
observability section asks for: the user message, relevant history,
retrieved passages + scores, tool calls + sanitized results, the final
response, and any fallback/handoff. Never logs secrets (no API keys, no
raw orders.json content beyond what was already sanitized for the model).
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app import config


@dataclass
class ToolCallTrace:
    tool_name: str
    arguments: dict[str, Any]
    result: dict[str, Any]


@dataclass
class RetrievedChunkTrace:
    filename: str
    heading: str
    score: float
    is_authoritative: bool


@dataclass
class TurnTrace:
    session_id: str
    turn_index: int
    timestamp: float
    user_message: str
    history_length: int
    retrieval_query: str
    retrieved_chunks: list[RetrievedChunkTrace] = field(default_factory=list)
    tool_calls: list[ToolCallTrace] = field(default_factory=list)
    final_response: str = ""
    handoff_recommended: bool = False
    error: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


class TraceLogger:
    def __init__(self, log_dir: Path = config.LOG_DIR, also_print: bool = False):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / "trace.jsonl"
        self.also_print = also_print

    def write(self, trace: TurnTrace) -> None:
        line = trace.to_json()
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        if self.also_print:
            print(f"[trace] {line}", file=sys.stderr)


def new_trace(session_id: str, turn_index: int, user_message: str, history_length: int, retrieval_query: str) -> TurnTrace:
    return TurnTrace(
        session_id=session_id,
        turn_index=turn_index,
        timestamp=time.time(),
        user_message=user_message,
        history_length=history_length,
        retrieval_query=retrieval_query,
    )
