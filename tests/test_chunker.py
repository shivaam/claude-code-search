from claude_code_search.chunker import chunk_message
from claude_code_search.config import ChunkingConfig
from claude_code_search.models import Message


def _msg(text: str) -> Message:
    return Message(
        session_id="s", project="p", uuid="u", role="user",
        ts="2026-01-01T00:00:00Z", cwd=None, text=text,
    )


def test_short_message_yields_one_chunk() -> None:
    chunks = chunk_message(_msg("hello world"), ChunkingConfig())
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"
    assert chunks[0].chunk_idx == 0


def test_long_message_is_split_with_overlap() -> None:
    cfg = ChunkingConfig()
    cfg.max_chars = 100
    cfg.overlap = 20
    text = "a" * 250
    chunks = chunk_message(_msg(text), cfg)
    assert len(chunks) >= 3
    assert all(len(c.text) <= cfg.max_chars for c in chunks)
    assert chunks[0].chunk_idx == 0
    assert chunks[-1].chunk_idx == len(chunks) - 1
    assert chunks[0].text.startswith("a")
    assert chunks[-1].text.endswith("a")


def test_exact_boundary_one_chunk() -> None:
    cfg = ChunkingConfig()
    cfg.max_chars = 100
    cfg.overlap = 20
    chunks = chunk_message(_msg("x" * 100), cfg)
    assert len(chunks) == 1
