from pathlib import Path

from claude_code_search.config import ChunkingConfig
from claude_code_search.parser import parse_file

FIX = Path(__file__).parent / "fixtures" / "sample_project" / "aaa.jsonl"


def test_parser_extracts_user_and_assistant_text() -> None:
    messages, _meta = parse_file(FIX, ChunkingConfig())
    texts = [m.text for m in messages]
    assert any("fix a broken Airflow scheduler" in t for t in texts)
    assert any("scheduler logs first" in t for t in texts)
    assert any("file1" in t for t in texts)
    assert not any("sidechain noise" in t for t in texts)
    assert not any(t.strip() == "ok" for t in texts)


def test_parser_produces_session_meta() -> None:
    _messages, meta = parse_file(FIX, ChunkingConfig())
    assert meta.session_id == "aaa"
    assert meta.project == "sample_project"
    assert meta.ai_title == "Debug Airflow scheduler"
    assert meta.first_user_msg is not None
    assert "fix a broken Airflow scheduler" in meta.first_user_msg
    assert meta.message_count >= 3
    assert meta.first_ts.startswith("2026-04-09T10:00:01")
    assert meta.cwd == "/tmp/p"
    assert meta.source_path.endswith("aaa.jsonl")


def test_parser_can_include_sidechains() -> None:
    cfg = ChunkingConfig()
    cfg.include_sidechains = True
    messages, _ = parse_file(FIX, cfg)
    assert any("sidechain noise" in m.text for m in messages)


def test_parser_captures_custom_title_latest_wins() -> None:
    # Fixture contains two custom-title records for session aaa. The second
    # one is the user's current manual rename and must win.
    _, meta = parse_file(FIX, ChunkingConfig())
    assert meta.custom_title == "second manual title (latest)"
    # ai_title is still captured independently.
    assert meta.ai_title == "Debug Airflow scheduler"
