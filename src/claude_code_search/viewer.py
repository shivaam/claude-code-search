"""Pretty-printer for conversation messages used by the `show` command."""
from __future__ import annotations

import textwrap

from .models import Message


def _format_ts(ts: str) -> str:
    if not ts:
        return "?"
    return ts.replace("T", " ")[:16]


def render_messages(messages: list[Message], width: int = 100) -> str:
    lines: list[str] = []
    for m in messages:
        header = f"[{_format_ts(m.ts)}] {m.role}:"
        body = textwrap.fill(
            m.text,
            width=width,
            initial_indent="  ",
            subsequent_indent="  ",
            replace_whitespace=False,
        )
        lines.append(header)
        lines.append(body)
        lines.append("")
    return "\n".join(lines)


def slice_around(
    messages: list[Message], pivot: int, context: int
) -> list[Message]:
    lo = max(0, pivot - context)
    hi = min(len(messages), pivot + context + 1)
    return messages[lo:hi]


def find_best_match_index(messages: list[Message], query: str) -> int:
    """Literal substring match; fall back to token overlap."""
    q = query.lower().strip()
    for i, m in enumerate(messages):
        if q in m.text.lower():
            return i
    q_tokens = set(q.split())
    best_i, best_score = 0, -1
    for i, m in enumerate(messages):
        tokens = set(m.text.lower().split())
        score = len(q_tokens & tokens)
        if score > best_score:
            best_i, best_score = i, score
    return best_i


def render_truncated(messages: list[Message], edge: int = 5) -> str:
    """First `edge` + last `edge` messages with an omission marker."""
    if len(messages) <= edge * 2:
        return render_messages(messages)
    head = render_messages(messages[:edge])
    tail = render_messages(messages[-edge:])
    marker = f"... ({len(messages) - edge * 2} messages omitted) ...\n\n"
    return head + marker + tail
