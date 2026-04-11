from pathlib import Path

from claude_code_search.config import Config
from claude_code_search.indexer import Indexer
from claude_code_search.store import connect, init_schema
from claude_code_search.summarizer import Summarizer
from tests.test_indexer import FakeEmbedder, _fresh_root


class FakeOllama:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate(self, model: str, prompt: str, options: dict | None = None) -> str:
        self.calls.append({"model": model, "prompt": prompt})
        return (
            "- [2026-04-09 10:00] Asked how to fix the Airflow scheduler\n"
            "- [2026-04-09 10:00] Inspected logs and tool output\n"
        )


def test_summarize_session_writes_summary_to_db(tmp_path: Path) -> None:
    root = _fresh_root(tmp_path)
    conn = connect(tmp_path / "db.sqlite")
    init_schema(conn)
    Indexer(conn, Config(), embedder=FakeEmbedder()).run(root)

    fake = FakeOllama()
    summ = Summarizer(conn, Config(), ollama=fake)
    summ.summarize_session("aaa")

    row = conn.execute(
        "SELECT summary, summary_model, summary_stale "
        "FROM sessions WHERE session_id='aaa'"
    ).fetchone()
    assert row["summary"] is not None
    assert "- [2026-04-09 10:00]" in row["summary"]
    assert row["summary_model"] == "gemma2:9b"
    assert row["summary_stale"] == 0
    assert len(fake.calls) == 1


def test_run_once_summarises_all_pending_sessions(tmp_path: Path) -> None:
    root = _fresh_root(tmp_path)
    conn = connect(tmp_path / "db.sqlite")
    init_schema(conn)
    Indexer(conn, Config(), embedder=FakeEmbedder()).run(root)

    fake = FakeOllama()
    summ = Summarizer(conn, Config(), ollama=fake)
    processed = summ.run_once()
    assert processed == 2
    remaining = conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE summary IS NULL"
    ).fetchone()[0]
    assert remaining == 0
