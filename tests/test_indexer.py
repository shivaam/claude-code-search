import os
import shutil
from pathlib import Path

import numpy as np

from claude_code_search.config import Config
from claude_code_search.indexer import Indexer
from claude_code_search.store import connect, init_schema


class FakeEmbedder:
    """Deterministic pseudo-random embedder for tests. Offline, fast."""

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), 384), dtype=np.float32)
        for i, t in enumerate(texts):
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.standard_normal(384).astype(np.float32)
            v /= np.linalg.norm(v) or 1.0
            out[i] = v
        return out


def _fresh_root(tmp_path: Path) -> Path:
    src_fix = Path(__file__).parent / "fixtures" / "sample_project"
    dst = tmp_path / "sample_project"
    dst.mkdir(parents=True)
    for jsonl in src_fix.glob("*.jsonl"):
        shutil.copy(jsonl, dst / jsonl.name)
    return tmp_path


def test_indexer_populates_all_tables(tmp_path: Path) -> None:
    root = _fresh_root(tmp_path)
    conn = connect(tmp_path / "db.sqlite")
    init_schema(conn)

    idx = Indexer(conn, Config(), embedder=FakeEmbedder())
    stats = idx.run(root)

    assert stats.files_indexed == 2
    assert stats.chunks_written >= 3

    file_rows = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    session_rows = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    chunk_rows = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    vec_rows = conn.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0]
    assert file_rows == 2
    assert session_rows == 2
    assert chunk_rows == vec_rows
    assert chunk_rows >= 3


def test_indexer_is_incremental(tmp_path: Path) -> None:
    root = _fresh_root(tmp_path)
    conn = connect(tmp_path / "db.sqlite")
    init_schema(conn)
    idx = Indexer(conn, Config(), embedder=FakeEmbedder())

    idx.run(root)
    stats2 = idx.run(root)
    assert stats2.files_indexed == 0

    target = root / "sample_project" / "bbb.jsonl"
    with target.open("a") as fh:
        fh.write(
            '\n{"type":"user","uuid":"z","sessionId":"bbb",'
            '"timestamp":"2026-04-10T11:05:00Z","cwd":"/tmp/q",'
            '"isSidechain":false,"message":{"role":"user",'
            '"content":"Another question about DAGs please."}}\n'
        )
    os.utime(target, None)

    stats3 = idx.run(root)
    assert stats3.files_indexed == 1


def test_indexer_marks_summary_stale_on_growth(tmp_path: Path) -> None:
    root = _fresh_root(tmp_path)
    conn = connect(tmp_path / "db.sqlite")
    init_schema(conn)
    cfg = Config()
    cfg.summarization.re_summarize_threshold = 1  # low threshold for test
    idx = Indexer(conn, cfg, embedder=FakeEmbedder())
    idx.run(root)

    conn.execute(
        "UPDATE sessions SET summary = ?, summary_model = ?, "
        "summarized_at = ?, summary_stale = 0 WHERE session_id = 'bbb'",
        ("- [2026-04-10] discussed dags", "fake", "2026-04-10T12:00:00Z"),
    )
    conn.commit()

    target = root / "sample_project" / "bbb.jsonl"
    with target.open("a") as fh:
        fh.write(
            '\n{"type":"user","uuid":"z2","sessionId":"bbb",'
            '"timestamp":"2026-04-10T11:06:00Z","cwd":"/tmp/q",'
            '"isSidechain":false,"message":{"role":"user",'
            '"content":"One more DAG question to ask about scheduler."}}\n'
        )
    os.utime(target, None)

    idx.run(root)
    row = conn.execute(
        "SELECT summary_stale FROM sessions WHERE session_id = 'bbb'"
    ).fetchone()
    assert row[0] == 1
