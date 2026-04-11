import os
import time
from pathlib import Path

from claude_code_search.store import connect, init_schema
from claude_code_search.walker import (
    files_needing_reindex,
    find_files,
    mark_indexed,
)


def test_find_files_recurses(tmp_path: Path) -> None:
    (tmp_path / "a.jsonl").write_text("{}")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.jsonl").write_text("{}")
    (tmp_path / "sub" / "ignore.txt").write_text("x")
    found = sorted(find_files(tmp_path))
    assert [p.name for p in found] == ["a.jsonl", "b.jsonl"]


def test_needs_reindex_on_first_seen_and_on_change(tmp_path: Path) -> None:
    f = tmp_path / "a.jsonl"
    f.write_text("x")
    conn = connect(tmp_path / "db.sqlite")
    init_schema(conn)

    assert files_needing_reindex([f], conn) == [f]
    mark_indexed(conn, f)
    conn.commit()
    assert files_needing_reindex([f], conn) == []

    time.sleep(0.01)
    f.write_text("xy")
    os.utime(f, None)
    assert files_needing_reindex([f], conn) == [f]
