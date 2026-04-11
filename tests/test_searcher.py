from pathlib import Path

import numpy as np

from claude_code_search.config import Config
from claude_code_search.indexer import Indexer
from claude_code_search.searcher import Searcher
from claude_code_search.store import connect, init_schema
from tests.test_indexer import _fresh_root


class QueryAwareFake:
    """Bag-of-chars fake so identical text yields identical vector."""

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), 384), dtype=np.float32)
        for i, t in enumerate(texts):
            v = np.zeros(384, dtype=np.float32)
            for ch in t.lower():
                v[ord(ch) % 384] += 1.0
            n = np.linalg.norm(v) or 1.0
            out[i] = v / n
        return out


def test_search_returns_expected_session(tmp_path: Path) -> None:
    root = _fresh_root(tmp_path)
    conn = connect(tmp_path / "db.sqlite")
    init_schema(conn)
    fake = QueryAwareFake()
    Indexer(conn, Config(), embedder=fake).run(root)

    searcher = Searcher(conn, Config(), embedder=fake)
    hits = searcher.search("DAG parse failures")
    assert len(hits) >= 1
    assert hits[0].session_id == "bbb"
    assert hits[0].ai_title == "Fix DAG parse failures"
    assert hits[0].message_count >= 2


def test_search_groups_by_session(tmp_path: Path) -> None:
    root = _fresh_root(tmp_path)
    conn = connect(tmp_path / "db.sqlite")
    init_schema(conn)
    fake = QueryAwareFake()
    Indexer(conn, Config(), embedder=fake).run(root)

    hits = Searcher(conn, Config(), embedder=fake).search("Airflow")
    session_ids = [h.session_id for h in hits]
    assert len(session_ids) == len(set(session_ids))
