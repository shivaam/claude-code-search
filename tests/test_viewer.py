from pathlib import Path

from claude_code_search.config import ChunkingConfig
from claude_code_search.parser import parse_file
from claude_code_search.viewer import (
    find_best_match_index,
    render_messages,
    slice_around,
)


FIX = Path(__file__).parent / "fixtures" / "sample_project" / "aaa.jsonl"


def test_render_messages_produces_readable_lines() -> None:
    messages, _ = parse_file(FIX, ChunkingConfig())
    out = render_messages(messages)
    assert "[2026-04-09" in out
    assert "user:" in out or "assistant:" in out


def test_slice_around_respects_context() -> None:
    messages, _ = parse_file(FIX, ChunkingConfig())
    sliced = slice_around(messages, pivot=1, context=1)
    assert len(sliced) == 3


def test_find_best_match_index_prefers_literal_substring() -> None:
    messages, _ = parse_file(FIX, ChunkingConfig())
    idx = find_best_match_index(messages, "scheduler logs")
    assert 0 <= idx < len(messages)
    assert "scheduler logs" in messages[idx].text
