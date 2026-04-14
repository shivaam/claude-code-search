"""Argparse entrypoint that wires subcommands to modules."""
from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

from .config import Config, load_config, write_default_config
from .indexer import Indexer
from .searcher import Searcher
from .store import connect, init_schema
from .scheduler import install as sched_install, uninstall as sched_uninstall, status as sched_status
from .summarizer import (
    Summarizer,
    daemon_status,
    log_file_path,
    pid_file_path,
    start_daemon,
    stop_daemon,
)
from .viewer import (
    find_best_match_index,
    render_messages,
    render_truncated,
    slice_around,
)


# --- Config loading --------------------------------------------------------


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parent


def _load_cfg(path_arg: str | None) -> Config:
    project = _project_root()
    cfg_path = Path(path_arg) if path_arg else project / "config.toml"
    if not cfg_path.exists() and not path_arg:
        try:
            write_default_config(cfg_path)
        except OSError:
            pass
    return load_config(cfg_path, project_root=project)


def _make_embedder(cfg: Config):
    """Isolated for easy monkeypatching in tests."""
    from .embedder import Embedder

    return Embedder(
        model_name=cfg.embedding.model,
        device=cfg.embedding.device,
        batch_size=cfg.embedding.batch_size,
    )


# --- Subcommand handlers ---------------------------------------------------


def _cmd_index(cfg: Config, args: argparse.Namespace) -> int:
    import time

    if args.rebuild:
        db_path = Path(cfg.paths.db)
        if db_path.exists():
            print(f"rebuilding from scratch: deleting {db_path}", file=sys.stderr)
            db_path.unlink()

    print(
        f"scanning {cfg.paths.root}", file=sys.stderr, flush=True
    )
    conn = connect(Path(cfg.paths.db))
    init_schema(conn)

    print(
        f"loading embedding model {cfg.embedding.model} (device={cfg.embedding.device})...",
        file=sys.stderr,
        flush=True,
    )
    embedder = _make_embedder(cfg)

    started = time.monotonic()
    reporter = _ProgressReporter(sys.stderr)
    stats = Indexer(conn, cfg, embedder=embedder).run(
        Path(cfg.paths.root), on_progress=reporter
    )
    reporter.finish()

    elapsed = time.monotonic() - started
    print(
        f"done in {_format_elapsed(elapsed)}: "
        f"indexed {stats.files_indexed} files "
        f"(skipped {stats.files_skipped}); "
        f"total {stats.chunks_written} chunks, {stats.sessions_written} sessions.",
        file=sys.stderr,
    )
    return 0


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


class _ProgressReporter:
    """Renders indexer progress to a stream. Uses \\r overwrite on a TTY,
    one-line-per-event on a pipe/file."""

    def __init__(self, stream) -> None:
        self.stream = stream
        self.is_tty = bool(getattr(stream, "isatty", lambda: False)())
        self.total_chunks = 0
        self.last_line_len = 0

    def __call__(self, ev) -> None:
        if ev.kind == "scanned":
            self.stream.write(
                f"found {ev.files_seen} files; "
                f"{ev.total} need (re)indexing, "
                f"{ev.files_skipped} up-to-date\n"
            )
            self.stream.flush()
            if ev.total == 0:
                return
            self.total_chunks = 0
            return

        if ev.kind == "file_done":
            self.total_chunks += ev.chunks_written
            name = ev.path.name if ev.path else "?"
            if len(name) > 40:
                name = name[:37] + "..."
            msg = (
                f"[{ev.index:>4}/{ev.total}] +{ev.chunks_written:>3} chunks "
                f"(total {self.total_chunks}): {name}"
            )
            if self.is_tty:
                # \r overwrite to keep one rolling line; pad to clear prior text.
                pad = max(0, self.last_line_len - len(msg))
                self.stream.write("\r" + msg + " " * pad)
                self.last_line_len = len(msg)
                # Newline every 50 files so scrollback has landmarks.
                if ev.index % 50 == 0:
                    self.stream.write("\n")
                    self.last_line_len = 0
            else:
                self.stream.write(msg + "\n")
            self.stream.flush()
            return

        if ev.kind == "file_error":
            # Errors always get their own line.
            if self.is_tty and self.last_line_len:
                self.stream.write("\n")
                self.last_line_len = 0
            self.stream.write(
                f"  [ERROR] {ev.path}: {ev.error}\n"
            )
            self.stream.flush()

    def finish(self) -> None:
        if self.is_tty and self.last_line_len:
            self.stream.write("\n")
            self.stream.flush()
            self.last_line_len = 0


_NOISE_TAG_RE = None  # lazy
_BULLET_RE = None  # lazy


def _use_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _term_width(default: int = 100) -> int:
    import shutil

    try:
        return shutil.get_terminal_size((default, 24)).columns
    except Exception:
        return default


def _c(code: str, text: str) -> str:
    if not _use_color():
        return text
    return f"\033[{code}m{text}\033[0m"


def _clean_fallback_title(text: str, max_len: int = 100) -> str:
    """Strip IDE/system noise tags and collapse whitespace for display."""
    global _NOISE_TAG_RE
    if _NOISE_TAG_RE is None:
        import re

        _NOISE_TAG_RE = re.compile(
            r"<(ide_opened_file|system-reminder|command-name|command-message|"
            r"command-args|local-command-stdout|ide_selection)[^>]*>.*?"
            r"</\1>",
            re.DOTALL | re.IGNORECASE,
        )
    cleaned = _NOISE_TAG_RE.sub("", text)
    cleaned = " ".join(cleaned.split())
    if len(cleaned) > max_len:
        cleaned = cleaned[: max_len - 1].rstrip() + "…"
    return cleaned or "(untitled)"


def _extract_bullets(summary: str, max_bullets: int = 6) -> list[str]:
    """Pull out bullet-looking lines from an LLM-generated outline.

    Accepts lines starting with -, *, •, a digit, or a bracketed/bare
    timestamp like [2026-04-11 17:37] or 2026-04-11 17:37.
    Ignores paragraph prose (which gemma2:9b sometimes drifts into).
    Returns [] if nothing looks like a bullet — caller should fall back.
    """
    global _BULLET_RE
    if _BULLET_RE is None:
        import re

        _BULLET_RE = re.compile(
            r"^\s*(?:"
            r"[-*•]\s+"                          # markdown bullet
            r"|\d+[\.)]\s+"                      # 1. or 1)
            r"|\[?\d{4}-\d{2}-\d{2}"             # timestamp prefix
            r")"
        )
    bullets: list[str] = []
    for line in summary.splitlines():
        line = line.strip()
        if not line or line.startswith("**"):
            continue
        if _BULLET_RE.match(line):
            clean = line.lstrip("-*• ").strip()
            # Strip leading **bold** markdown — gemma2 often writes
            # "- **Topic name:** actual content" and we don't want the **.
            import re as _re

            clean = _re.sub(r"\*\*([^*]+?)\*\*", r"\1", clean)
            clean = clean.replace("**", "")
            bullets.append(clean)
            if len(bullets) >= max_bullets:
                break
    return bullets


def _wrap_snippet(text: str, width: int, indent: str) -> str:
    """Wrap a snippet for display, preserving the indent on continuation lines."""
    import textwrap

    text = " ".join(text.split())
    if len(text) <= width:
        return text
    lines = textwrap.wrap(
        text,
        width=width,
        initial_indent="",
        subsequent_indent=indent,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return "\n".join(lines)


def _strip_project_prefix(cfg: Config, project: str, width: int = 22) -> str:
    prefix = cfg.output.strip_project_prefix
    if project.startswith(prefix):
        project = project[len(prefix):]
    if len(project) > width:
        project = project[: width - 1] + "…"
    return project.ljust(width)


def _format_hit(cfg: Config, rank: int, hit) -> str:
    project = _strip_project_prefix(cfg, hit.project)
    date = hit.ts[:10] if hit.ts else "?"

    term_w = _term_width()
    # Reserve left indent (6 cols for bullet area, 4 for other body lines).
    body_w = max(60, term_w - 4)
    bullet_w = max(60, term_w - 8)

    header = (
        f"{_c('1', f'{rank:>2}.')} "
        f"{_c('36', f'{hit.score:.2f}')}  "
        f"{_c('37', date)}  "
        f"{_c('35', project)}  "
        f"{_c('2', hit.session_id)}  "
        f"{_c('2', f'({hit.message_count} msgs)')}"
    )
    lines = [header]

    # Title line — custom_title (your manual rename) > ai_title (Claude Code's
    # auto-generated) > first user message as a last resort. Wrap to body width.
    raw_title = (
        hit.custom_title
        or hit.ai_title
        or _clean_fallback_title(hit.first_user_msg or "", max_len=400)
    )
    title_wrapped = _wrap_snippet(raw_title, width=body_w, indent="    ")
    lines.append(f"    {_c('1', title_wrapped)}")

    # Outline bullets — wrap each to terminal width with hanging indent.
    if hit.summary:
        bullets = _extract_bullets(hit.summary)
        for b in bullets:
            wrapped = _wrap_snippet(b, width=bullet_w, indent="        ")
            lines.append(f"      {_c('2', '•')} {wrapped}")

    # Match snippet, wrapped.
    wrapped = _wrap_snippet(hit.snippet, width=body_w, indent="      ")
    lines.append(f'    {_c("33", "↳")} {_c("2", wrapped)}')

    # Resume command, dimmed.
    resume_cmd = (
        f"(cd {shlex.quote(hit.cwd or '.')} && claude -r {hit.session_id})"
    )
    lines.append(f"    {_c('2', '$')} {_c('2', resume_cmd)}")

    return "\n".join(lines)


def _format_hit_compact(cfg: Config, rank: int, hit) -> str:
    """One line per result — scannable table."""
    project = _strip_project_prefix(cfg, hit.project, width=18)
    date = hit.ts[:10] if hit.ts else "?"
    title = (
        hit.custom_title
        or hit.ai_title
        or _clean_fallback_title(hit.first_user_msg or "", max_len=60)
    )
    if len(title) > 58:
        title = title[:57] + "…"
    return (
        f"{_c('1', f'{rank:>2}.')} "
        f"{_c('36', f'{hit.score:.2f}')}  "
        f"{_c('37', date)}  "
        f"{_c('35', project)}  "
        f"{_c('2', hit.session_id[:8])}…  "
        f"{title}"
    )


def _cmd_search(cfg: Config, args: argparse.Namespace) -> int:
    conn = connect(Path(cfg.paths.db))
    init_schema(conn)

    if args.grep:
        # Literal substring search — no embedding model needed, instant.
        searcher = Searcher(conn, cfg, embedder=None)
        hits = searcher.grep_search(
            args.query, project_filter=args.project, top_n=args.n
        )
    else:
        embedder = _make_embedder(cfg)
        hits = Searcher(conn, cfg, embedder=embedder).search(
            args.query, project_filter=args.project, top_n=args.n
        )

    if args.json:
        import json

        out = [
            {
                "rank": i + 1,
                "session_id": h.session_id,
                "project": h.project,
                "cwd": h.cwd,
                "ts": h.ts,
                "score": h.score,
                "custom_title": h.custom_title,
                "ai_title": h.ai_title,
                "summary": h.summary,
                "snippet": h.snippet,
                "message_count": h.message_count,
            }
            for i, h in enumerate(hits)
        ]
        print(json.dumps(out, indent=2))
        return 0

    if not hits:
        print("(no results)")
        return 0

    if args.compact:
        print()
        for i, hit in enumerate(hits, 1):
            print(_format_hit_compact(cfg, i, hit))
        print()
    else:
        sep = _c("2", "─" * 78)
        print()
        for i, hit in enumerate(hits, 1):
            print(_format_hit(cfg, i, hit))
            if i < len(hits):
                print(f"    {sep}")
        print()

    remaining = conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE summary IS NULL"
    ).fetchone()[0]
    status = daemon_status(cfg, conn)
    if remaining > 0 and not status["running"]:
        print(
            _c(
                "2",
                f"hint: {remaining} conversations not yet summarized — "
                f"run 'ccsearch summarize --daemon' to populate",
            )
        )
    return 0


def _cmd_show(cfg: Config, args: argparse.Namespace) -> int:
    conn = connect(Path(cfg.paths.db))
    init_schema(conn)
    row = conn.execute(
        "SELECT source_path FROM sessions WHERE session_id = ?",
        (args.session_id,),
    ).fetchone()
    if row is None:
        print(f"unknown session_id: {args.session_id}", file=sys.stderr)
        return 2

    from .parser import parse_file

    messages, _ = parse_file(Path(row["source_path"]), cfg.chunking)
    if args.full:
        print(render_messages(messages))
        return 0
    if args.query:
        ctx = args.context if args.context is not None else cfg.show.default_context
        pivot = find_best_match_index(messages, args.query)
        print(render_messages(slice_around(messages, pivot, ctx)))
        return 0
    print(render_truncated(messages))
    return 0


def _cmd_resume(cfg: Config, args: argparse.Namespace) -> int:
    conn = connect(Path(cfg.paths.db))
    init_schema(conn)
    row = conn.execute(
        "SELECT cwd FROM sessions WHERE session_id = ?", (args.session_id,)
    ).fetchone()
    if row is None:
        print(f"unknown session_id: {args.session_id}", file=sys.stderr)
        return 2
    cwd = row["cwd"] or os.getcwd()
    print(
        f"cd {shlex.quote(cwd)} && claude -r {args.session_id}", file=sys.stderr
    )
    os.chdir(cwd)
    os.execvp("claude", ["claude", "-r", args.session_id])


def _cmd_summarize(cfg: Config, args: argparse.Namespace) -> int:
    conn = connect(Path(cfg.paths.db))
    init_schema(conn)
    summ = Summarizer(conn, cfg)

    if args.status:
        s = daemon_status(cfg, conn)
        pid_info = f" (pid {s['pid']})" if s["pid"] else ""
        print(f"daemon: {'running' if s['running'] else 'stopped'}{pid_info}")
        print(f"summaries: {s['done']}/{s['total']}  remaining: {s['remaining']}")
        print(f"log: {log_file_path(cfg)}")
        return 0

    if args.stop:
        ok = stop_daemon(cfg)
        print("daemon stopped" if ok else "no daemon was running")
        return 0 if ok else 1

    if args.session:
        summ.summarize_session(args.session)
        print(f"summarized {args.session}")
        return 0

    if args.all:
        n = summ.run_once()
        print(f"summarized {n} sessions")
        return 0

    if args.daemon:
        def _factory():
            c = connect(Path(cfg.paths.db))
            init_schema(c)
            return c

        try:
            start_daemon(cfg, _factory)
            print(f"daemon started (pid file: {pid_file_path(cfg)})")
            return 0
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            return 1

    print(
        "specify one of --daemon, --status, --stop, --session, --all",
        file=sys.stderr,
    )
    return 2


def _cmd_stats(cfg: Config, args: argparse.Namespace) -> int:
    conn = connect(Path(cfg.paths.db))
    init_schema(conn)
    files = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    with_sum = conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE summary IS NOT NULL"
    ).fetchone()[0]
    db_size = (
        Path(cfg.paths.db).stat().st_size if Path(cfg.paths.db).exists() else 0
    )
    status = daemon_status(cfg, conn)
    pid_info = f" (pid {status['pid']})" if status["pid"] else ""
    print(f"db:              {cfg.paths.db} ({db_size} bytes)")
    print(f"files indexed:   {files}")
    print(f"sessions:        {sessions}")
    print(f"chunks:          {chunks}")
    print(f"summaries done:  {with_sum}/{sessions}")
    print(f"embedding model: {cfg.embedding.model} ({cfg.embedding.device})")
    print(f"summary model:   {cfg.summarization.model}")
    print(f"daemon:          {'running' if status['running'] else 'stopped'}{pid_info}")
    return 0


# --- Argparse setup --------------------------------------------------------


def _cmd_schedule(cfg: Config, args: argparse.Namespace) -> int:
    if args.install:
        print(sched_install(cfg))
        return 0
    if args.uninstall:
        print(sched_uninstall(cfg))
        return 0
    # Default: show status.
    print(sched_status(cfg))
    return 0


_SUBCOMMANDS = {"index", "search", "show", "resume", "summarize", "stats", "schedule"}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ccsearch")
    p.add_argument("--config", default=None, help="path to config.toml")
    sub = p.add_subparsers(dest="cmd")

    p_index = sub.add_parser("index")
    p_index.add_argument("--root", default=None)
    p_index.add_argument("--rebuild", action="store_true")

    p_search = sub.add_parser("search")
    p_search.add_argument("query")
    p_search.add_argument("-n", type=int, default=None)
    p_search.add_argument("--project", default=None)
    p_search.add_argument("--json", action="store_true")
    p_search.add_argument(
        "-c", "--compact", action="store_true", help="one line per result"
    )
    p_search.add_argument(
        "-g",
        "--grep",
        action="store_true",
        help="literal case-insensitive search (fast, no embedding model needed)",
    )

    p_show = sub.add_parser("show")
    p_show.add_argument("session_id")
    p_show.add_argument("--query", default=None)
    p_show.add_argument("--context", type=int, default=None)
    p_show.add_argument("--full", action="store_true")

    p_resume = sub.add_parser("resume")
    p_resume.add_argument("session_id")

    p_summ = sub.add_parser("summarize")
    p_summ.add_argument("--daemon", action="store_true")
    p_summ.add_argument("--status", action="store_true")
    p_summ.add_argument("--stop", action="store_true")
    p_summ.add_argument("--session", default=None)
    p_summ.add_argument("--all", action="store_true")

    sub.add_parser("stats")

    p_sched = sub.add_parser("schedule")
    p_sched.add_argument(
        "--install", action="store_true", help="install daily scheduled index"
    )
    p_sched.add_argument(
        "--uninstall", action="store_true", help="remove scheduled index"
    )

    return p


def _insert_search_shortcut(argv: list[str]) -> list[str]:
    """Detect `ccsearch "query"` (no subcommand) and rewrite to `search query`."""
    out = list(argv)
    i = 0
    # Skip global flags that take a value (--config PATH).
    while i < len(out):
        tok = out[i]
        if tok == "--config":
            i += 2
            continue
        if tok.startswith("--"):
            i += 1
            continue
        break
    if i < len(out) and out[i] not in _SUBCOMMANDS:
        out.insert(i, "search")
    return out


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    argv = _insert_search_shortcut(argv)

    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.cmd is None:
        parser.print_help()
        return 0

    cfg = _load_cfg(args.config)
    if args.cmd == "index":
        if args.root:
            cfg.paths.root = args.root
        return _cmd_index(cfg, args)
    if args.cmd == "search":
        return _cmd_search(cfg, args)
    if args.cmd == "show":
        return _cmd_show(cfg, args)
    if args.cmd == "resume":
        return _cmd_resume(cfg, args)
    if args.cmd == "summarize":
        return _cmd_summarize(cfg, args)
    if args.cmd == "stats":
        return _cmd_stats(cfg, args)
    if args.cmd == "schedule":
        return _cmd_schedule(cfg, args)
    parser.print_help()
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
