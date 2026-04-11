"""SQLite connection + schema initialisation.

Loads the sqlite-vec extension on every connection so the vec0 virtual
table is available.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import sqlite_vec


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS files (
    path       TEXT PRIMARY KEY,
    mtime_ns   INTEGER NOT NULL,
    indexed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id     TEXT PRIMARY KEY,
    project        TEXT NOT NULL,
    source_path    TEXT NOT NULL,
    first_ts       TEXT NOT NULL,
    last_ts        TEXT NOT NULL,
    message_count  INTEGER NOT NULL,
    cwd            TEXT,
    ai_title       TEXT,
    custom_title   TEXT,
    first_user_msg TEXT,
    summary        TEXT,
    summary_model  TEXT,
    summarized_at  TEXT,
    summary_stale  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    project     TEXT NOT NULL,
    uuid        TEXT NOT NULL,
    role        TEXT NOT NULL,
    ts          TEXT NOT NULL,
    cwd         TEXT,
    chunk_idx   INTEGER NOT NULL,
    text        TEXT NOT NULL,
    source_path TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_session ON chunks(session_id);
CREATE INDEX IF NOT EXISTS idx_chunks_source  ON chunks(source_path);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    try:
        sqlite_vec.load(conn)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "sqlite-vec extension failed to load. "
            "Install with `pip install sqlite-vec`."
        ) from e
    finally:
        conn.enable_load_extension(False)
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0("
        "embedding float[384]"
        ")"
    )
    # Idempotent column additions for existing databases created before a
    # given column was added to the schema. SQLite's ADD COLUMN raises
    # OperationalError if the column already exists.
    _ensure_column(conn, "sessions", "custom_title", "TEXT")
    conn.commit()


def _ensure_column(
    conn: sqlite3.Connection, table: str, column: str, col_type: str
) -> None:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    names = {r[1] for r in rows}
    if column not in names:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
