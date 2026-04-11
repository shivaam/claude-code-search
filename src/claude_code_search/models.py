"""Shared dataclasses used across the package."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Message:
    session_id: str
    project: str
    uuid: str
    role: str           # "user" | "assistant"
    ts: str             # ISO-8601
    cwd: str | None
    text: str


@dataclass
class SessionMeta:
    session_id: str
    project: str
    source_path: str
    first_ts: str
    last_ts: str
    message_count: int
    cwd: str | None
    ai_title: str | None
    custom_title: str | None  # latest user-set display name (type=="custom-title")
    first_user_msg: str | None


@dataclass
class Chunk:
    session_id: str
    project: str
    uuid: str
    role: str
    ts: str
    cwd: str | None
    text: str
    chunk_idx: int


@dataclass
class Hit:
    session_id: str
    project: str
    cwd: str | None
    ts: str
    score: float
    snippet: str
    ai_title: str | None
    custom_title: str | None
    summary: str | None
    first_user_msg: str | None
    message_count: int
    summary_stale: bool
