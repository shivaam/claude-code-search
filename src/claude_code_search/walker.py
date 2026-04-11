"""Recursive .jsonl discovery + incremental-reindex decision."""
from __future__ import annotations

import datetime as _dt
import sqlite3
from pathlib import Path


def find_files(root: Path) -> list[Path]:
    return sorted(Path(root).rglob("*.jsonl"))


def files_needing_reindex(
    paths: list[Path], conn: sqlite3.Connection
) -> list[Path]:
    needed: list[Path] = []
    for p in paths:
        try:
            mtime = p.stat().st_mtime_ns
        except FileNotFoundError:
            continue
        row = conn.execute(
            "SELECT mtime_ns FROM files WHERE path = ?", (str(p),)
        ).fetchone()
        if row is None or row[0] < mtime:
            needed.append(p)
    return needed


def mark_indexed(conn: sqlite3.Connection, path: Path) -> None:
    mtime = path.stat().st_mtime_ns
    now = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )
    conn.execute(
        "INSERT OR REPLACE INTO files (path, mtime_ns, indexed_at) VALUES (?, ?, ?)",
        (str(path), mtime, now),
    )
