"""Orchestrates walk → parse → chunk → embed → write."""
from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from . import walker
from .chunker import chunk_message
from .config import Config
from .models import Chunk, SessionMeta
from .parser import parse_file


class EmbedderProtocol(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


@dataclass
class IndexStats:
    files_indexed: int = 0
    files_skipped: int = 0
    chunks_written: int = 0
    sessions_written: int = 0


@dataclass
class ProgressEvent:
    """Emitted by Indexer as it works, for the CLI to render progress."""

    kind: str                     # "scanned" | "file_done" | "file_error"
    index: int = 0                # 1-based current file index
    total: int = 0                # total files to index this run
    files_seen: int = 0           # total files under root (only on "scanned")
    files_skipped: int = 0        # up-to-date files (only on "scanned")
    path: Path | None = None
    chunks_written: int = 0       # chunks written for this file (on "file_done")
    error: str | None = None      # error message (on "file_error")


ProgressFn = Callable[[ProgressEvent], None]


class Indexer:
    def __init__(
        self,
        conn: sqlite3.Connection,
        cfg: Config,
        embedder: EmbedderProtocol,
    ) -> None:
        self.conn = conn
        self.cfg = cfg
        self.embedder = embedder

    def run(
        self, root: Path, on_progress: ProgressFn | None = None
    ) -> IndexStats:
        stats = IndexStats()
        all_files = walker.find_files(Path(root))
        needed = walker.files_needing_reindex(all_files, self.conn)
        stats.files_skipped = len(all_files) - len(needed)

        if on_progress:
            on_progress(
                ProgressEvent(
                    kind="scanned",
                    total=len(needed),
                    files_seen=len(all_files),
                    files_skipped=stats.files_skipped,
                )
            )

        for i, path in enumerate(needed, 1):
            try:
                chunks_written = self._index_one(path)
                stats.files_indexed += 1
                if on_progress:
                    on_progress(
                        ProgressEvent(
                            kind="file_done",
                            index=i,
                            total=len(needed),
                            path=path,
                            chunks_written=chunks_written,
                        )
                    )
            except Exception as e:  # pragma: no cover
                msg = f"{type(e).__name__}: {e}"
                print(f"[indexer] {path}: {msg}", file=sys.stderr)
                if on_progress:
                    on_progress(
                        ProgressEvent(
                            kind="file_error",
                            index=i,
                            total=len(needed),
                            path=path,
                            error=msg,
                        )
                    )

        stats.chunks_written = self.conn.execute(
            "SELECT COUNT(*) FROM chunks"
        ).fetchone()[0]
        stats.sessions_written = self.conn.execute(
            "SELECT COUNT(*) FROM sessions"
        ).fetchone()[0]
        return stats

    def _index_one(self, path: Path) -> int:
        """Index one file. Returns the number of chunks written."""
        conn = self.conn
        cfg = self.cfg

        # Order matters: vec_chunks first (no FK cascade), then chunks.
        conn.execute(
            "DELETE FROM vec_chunks WHERE rowid IN "
            "(SELECT id FROM chunks WHERE source_path = ?)",
            (str(path),),
        )
        conn.execute("DELETE FROM chunks WHERE source_path = ?", (str(path),))

        messages, meta = parse_file(path, cfg.chunking)
        self._upsert_session(meta)

        chunks: list[Chunk] = [
            c for m in messages for c in chunk_message(m, cfg.chunking)
        ]
        if chunks:
            vecs = self.embedder.embed([c.text for c in chunks])
            for chunk, vec in zip(chunks, vecs):
                cur = conn.execute(
                    "INSERT INTO chunks "
                    "(session_id, project, uuid, role, ts, cwd, "
                    " chunk_idx, text, source_path) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        chunk.session_id,
                        chunk.project,
                        chunk.uuid,
                        chunk.role,
                        chunk.ts,
                        chunk.cwd,
                        chunk.chunk_idx,
                        chunk.text,
                        str(path),
                    ),
                )
                conn.execute(
                    "INSERT INTO vec_chunks (rowid, embedding) VALUES (?, ?)",
                    (cur.lastrowid, vec.tobytes()),
                )

        walker.mark_indexed(conn, path)
        conn.commit()
        return len(chunks)

    def _upsert_session(self, meta: SessionMeta) -> None:
        conn = self.conn
        existing = conn.execute(
            "SELECT message_count, summary FROM sessions WHERE session_id = ?",
            (meta.session_id,),
        ).fetchone()

        stale = 0
        if existing is not None:
            prev_count = existing["message_count"]
            had_summary = existing["summary"] is not None
            if (
                self.cfg.summarization.re_summarize_on_growth
                and had_summary
                and meta.message_count > prev_count
            ):
                stale = 1

        conn.execute(
            "INSERT INTO sessions "
            "(session_id, project, source_path, first_ts, last_ts, "
            " message_count, cwd, ai_title, custom_title, first_user_msg, "
            " summary_stale) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET "
            "project=excluded.project, source_path=excluded.source_path, "
            "first_ts=excluded.first_ts, last_ts=excluded.last_ts, "
            "message_count=excluded.message_count, cwd=excluded.cwd, "
            "ai_title=COALESCE(excluded.ai_title, sessions.ai_title), "
            "custom_title=COALESCE(excluded.custom_title, sessions.custom_title), "
            "first_user_msg=COALESCE(excluded.first_user_msg, sessions.first_user_msg), "
            "summary_stale=CASE WHEN ?=1 THEN 1 ELSE sessions.summary_stale END",
            (
                meta.session_id,
                meta.project,
                meta.source_path,
                meta.first_ts,
                meta.last_ts,
                meta.message_count,
                meta.cwd,
                meta.ai_title,
                meta.custom_title,
                meta.first_user_msg,
                stale,
                stale,
            ),
        )
