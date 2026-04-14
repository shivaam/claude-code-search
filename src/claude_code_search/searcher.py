"""Semantic search over indexed chunks, grouped by session."""
from __future__ import annotations

import sqlite3
from typing import Protocol

import numpy as np

from .config import Config
from .models import Hit


class EmbedderProtocol(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class Searcher:
    def __init__(
        self,
        conn: sqlite3.Connection,
        cfg: Config,
        embedder: EmbedderProtocol | None = None,
    ) -> None:
        self.conn = conn
        self.cfg = cfg
        self.embedder = embedder

    def search(
        self,
        query: str,
        project_filter: str | None = None,
        top_n: int | None = None,
    ) -> list[Hit]:
        qvec = self.embedder.embed([query])[0]
        # sqlite-vec hard-caps k at 4096.
        k = min(self.cfg.search.k_chunks, 4096)
        rows = self.conn.execute(
            "SELECT c.id, c.session_id, c.project, c.ts, c.cwd, c.text, "
            "       v.distance "
            "FROM vec_chunks v "
            "JOIN chunks c ON c.id = v.rowid "
            "WHERE v.embedding MATCH ? AND k = ? "
            "ORDER BY v.distance",
            (qvec.tobytes(), k),
        ).fetchall()

        best_per_session: dict[str, Hit] = {}
        for row in rows:
            if project_filter and project_filter not in row["project"]:
                continue
            sid = row["session_id"]
            score = 1.0 - float(row["distance"])
            if sid in best_per_session and best_per_session[sid].score >= score:
                continue
            meta = self._lookup_session_meta(sid)
            snippet = self._snippet(row["text"])
            hit = Hit(
                session_id=sid,
                project=row["project"],
                cwd=meta["cwd"] if meta else row["cwd"],
                ts=row["ts"],
                score=score,
                snippet=snippet,
                ai_title=meta["ai_title"] if meta else None,
                custom_title=meta["custom_title"] if meta else None,
                summary=meta["summary"] if meta else None,
                first_user_msg=meta["first_user_msg"] if meta else None,
                message_count=meta["message_count"] if meta else 0,
                summary_stale=bool(meta["summary_stale"]) if meta else False,
            )
            best_per_session[sid] = hit

        ordered = sorted(best_per_session.values(), key=lambda h: -h.score)
        limit = top_n if top_n is not None else self.cfg.search.top_n
        return ordered[:limit]

    def _lookup_session_meta(self, session_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT cwd, ai_title, custom_title, summary, first_user_msg, "
            "       message_count, summary_stale "
            "FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return dict(row) if row else None

    def grep_search(
        self,
        query: str,
        project_filter: str | None = None,
        top_n: int | None = None,
    ) -> list[Hit]:
        """Case-insensitive substring search. No embedder needed.

        Results are ranked by recency (most recent matching chunk first),
        grouped by session_id (one result per conversation).
        """
        like_pat = f"%{query}%"
        rows = self.conn.execute(
            "SELECT c.session_id, c.project, c.ts, c.cwd, c.text "
            "FROM chunks c "
            "WHERE c.text LIKE ? COLLATE NOCASE "
            "ORDER BY c.ts DESC",
            (like_pat,),
        ).fetchall()

        best_per_session: dict[str, Hit] = {}
        for row in rows:
            if project_filter and project_filter not in row["project"]:
                continue
            sid = row["session_id"]
            if sid in best_per_session:
                continue  # keep the most recent (first seen, since ORDER BY ts DESC)
            meta = self._lookup_session_meta(sid)
            snippet = self._highlight_snippet(row["text"], query)
            hit = Hit(
                session_id=sid,
                project=row["project"],
                cwd=meta["cwd"] if meta else row["cwd"],
                ts=row["ts"],
                score=1.0,  # grep matches are all equally "exact"
                snippet=snippet,
                ai_title=meta["ai_title"] if meta else None,
                custom_title=meta["custom_title"] if meta else None,
                summary=meta["summary"] if meta else None,
                first_user_msg=meta["first_user_msg"] if meta else None,
                message_count=meta["message_count"] if meta else 0,
                summary_stale=bool(meta["summary_stale"]) if meta else False,
            )
            best_per_session[sid] = hit

        limit = top_n if top_n is not None else self.cfg.search.top_n
        return list(best_per_session.values())[:limit]

    def _highlight_snippet(self, text: str, query: str) -> str:
        """Extract a snippet centered around the first match of query."""
        max_len = self.cfg.search.snippet_chars
        collapsed = " ".join(text.split())
        lower = collapsed.lower()
        idx = lower.find(query.lower())
        if idx < 0:
            return self._snippet(collapsed)
        # Center a window around the match.
        half = max_len // 2
        start = max(0, idx - half)
        end = min(len(collapsed), idx + len(query) + half)
        snippet = collapsed[start:end]
        if start > 0:
            snippet = "…" + snippet
        if end < len(collapsed):
            snippet = snippet + "…"
        return snippet

    def _snippet(self, text: str) -> str:
        max_len = self.cfg.search.snippet_chars
        collapsed = " ".join(text.split())
        if len(collapsed) <= max_len:
            return collapsed
        return collapsed[: max_len - 1].rstrip() + "…"
