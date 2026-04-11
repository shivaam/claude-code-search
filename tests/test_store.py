from pathlib import Path

import numpy as np

from claude_code_search.store import connect, init_schema


def test_connect_and_init_schema(tmp_path: Path) -> None:
    conn = connect(tmp_path / "x.db")
    init_schema(conn)
    tables = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table')"
        )
    }
    assert {"files", "sessions", "chunks"}.issubset(tables)
    conn.execute(
        "INSERT INTO vec_chunks(rowid, embedding) VALUES (1, ?)",
        (np.zeros(384, dtype=np.float32).tobytes(),),
    )
    conn.commit()
    cnt = conn.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0]
    assert cnt == 1
