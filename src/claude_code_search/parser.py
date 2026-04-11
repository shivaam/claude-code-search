"""Parse Claude Code .jsonl files into Messages + SessionMeta."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .config import ChunkingConfig
from .models import Message, SessionMeta


def _extract_text(content: Any) -> str:
    """Turn message.content into a plain-text string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                parts.append(str(block.get("text", "")))
            elif btype == "tool_result":
                inner = block.get("content")
                if isinstance(inner, str):
                    parts.append(inner)
                elif isinstance(inner, list):
                    for sub in inner:
                        if isinstance(sub, dict) and sub.get("type") == "text":
                            parts.append(str(sub.get("text", "")))
            # tool_use and image are intentionally skipped.
        return "\n".join(p for p in parts if p).strip()
    return ""


def parse_file(
    path: Path, cfg: ChunkingConfig
) -> tuple[list[Message], SessionMeta]:
    """Parse a .jsonl file. Returns the list of kept messages and the session meta.

    Malformed JSON lines are logged to stderr and skipped.
    """
    project = path.parent.name
    messages: list[Message] = []
    ai_title: str | None = None
    custom_title: str | None = None
    first_user_msg: str | None = None
    first_ts: str | None = None
    last_ts: str | None = None
    session_id: str | None = None
    last_cwd: str | None = None

    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(
                    f"[parser] {path}:{lineno} malformed JSON: {e}",
                    file=sys.stderr,
                )
                continue

            rtype = rec.get("type")
            sid = rec.get("sessionId")
            if sid and session_id is None:
                session_id = sid

            if rtype == "ai-title":
                ai_title = rec.get("aiTitle") or ai_title
                continue

            if rtype == "custom-title":
                # Latest wins: every rename appends a new record to the file.
                ct = rec.get("customTitle")
                if ct:
                    custom_title = ct
                continue

            if rtype not in ("user", "assistant"):
                continue
            if rec.get("isSidechain") and not cfg.include_sidechains:
                continue

            msg_obj = rec.get("message") or {}
            text = _extract_text(msg_obj.get("content"))
            if len(text) < cfg.min_chars:
                continue

            ts = rec.get("timestamp", "")
            cwd = rec.get("cwd")
            if cwd:
                last_cwd = cwd

            if first_ts is None:
                first_ts = ts
            last_ts = ts

            msg = Message(
                session_id=sid or "",
                project=project,
                uuid=rec.get("uuid", ""),
                role=rtype,
                ts=ts,
                cwd=cwd,
                text=text,
            )
            messages.append(msg)

            if rtype == "user" and first_user_msg is None:
                first_user_msg = text[:300]

    meta = SessionMeta(
        session_id=session_id or path.stem,
        project=project,
        source_path=str(path.resolve()),
        first_ts=first_ts or "",
        last_ts=last_ts or "",
        message_count=len(messages),
        cwd=last_cwd,
        ai_title=ai_title,
        custom_title=custom_title,
        first_user_msg=first_user_msg,
    )
    return messages, meta
