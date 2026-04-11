"""Split long Messages into overlapping Chunks."""
from __future__ import annotations

from .config import ChunkingConfig
from .models import Chunk, Message


def _find_cut(text: str, target: int, window: int = 100) -> int:
    """Find a nicer cut point near `target`, within ±window chars.

    Prefers paragraph (\\n\\n), then sentence (". "), else hard-cut.
    """
    lo = max(0, target - window)
    hi = min(len(text), target + window)
    idx = text.rfind("\n\n", lo, hi)
    if idx >= 0:
        return idx + 2
    idx = text.rfind(". ", lo, hi)
    if idx >= 0:
        return idx + 2
    return target


def chunk_message(msg: Message, cfg: ChunkingConfig) -> list[Chunk]:
    text = msg.text
    if len(text) <= cfg.max_chars:
        return [
            Chunk(
                session_id=msg.session_id,
                project=msg.project,
                uuid=msg.uuid,
                role=msg.role,
                ts=msg.ts,
                cwd=msg.cwd,
                text=text,
                chunk_idx=0,
            )
        ]

    chunks: list[Chunk] = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(len(text), start + cfg.max_chars)
        if end < len(text):
            end = _find_cut(text, end)
        piece = text[start:end]
        chunks.append(
            Chunk(
                session_id=msg.session_id,
                project=msg.project,
                uuid=msg.uuid,
                role=msg.role,
                ts=msg.ts,
                cwd=msg.cwd,
                text=piece,
                chunk_idx=idx,
            )
        )
        idx += 1
        if end >= len(text):
            break
        start = max(end - cfg.overlap, start + 1)
    return chunks
