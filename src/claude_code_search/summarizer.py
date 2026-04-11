"""Generate time-segmented conversation outlines via ollama."""
from __future__ import annotations

import datetime as _dt
import os
import signal
import sqlite3
import sys
import time
from pathlib import Path

from .config import ChunkingConfig, Config
from .ollama_client import OllamaClient, OllamaError
from .parser import parse_file

PROMPT_TEMPLATE = """\
You are summarizing a Claude Code conversation.

Produce a bulleted markdown outline of what was discussed, in chronological order.
- Up to {max_bullets} bullets.
- Each bullet begins with a timestamp in format [YYYY-MM-DD HH:MM].
- Each bullet is one sentence naming the topic or goal of that section.
- Split into a new bullet when the conversation topic changes OR when there
  is a time gap larger than {time_gap_min} minutes.
- Do not include tool call details or code snippets.
- Output ONLY the bullets, nothing else.

Conversation:
{transcript}
"""


def _now_iso() -> str:
    return (
        _dt.datetime.now(_dt.UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _build_transcript(
    source_path: Path, cfg: ChunkingConfig, max_chars: int
) -> str:
    """Re-parse the .jsonl into clean, non-overlapping messages.

    Trims each message to ~2000 chars; samples every Nth if the full
    transcript exceeds max_chars.
    """
    messages, _ = parse_file(source_path, cfg)
    lines: list[str] = []
    for m in messages:
        text = m.text if len(m.text) <= 2000 else m.text[:2000] + " …"
        ts = m.ts.replace("T", " ")[:16] if m.ts else "?"
        lines.append(f"[{ts}] {m.role}: {text}")

    transcript = "\n".join(lines)
    if len(transcript) <= max_chars or not lines:
        return transcript

    # Sample down: keep first and last, stride through the middle.
    target_lines = max(4, max_chars // 500)
    if len(lines) <= target_lines:
        return transcript[:max_chars]
    stride = max(1, len(lines) // target_lines)
    sampled = [lines[0]] + lines[1:-1:stride] + [lines[-1]]
    return "\n".join(sampled)[:max_chars]


class Summarizer:
    def __init__(
        self,
        conn: sqlite3.Connection,
        cfg: Config,
        ollama: OllamaClient | None = None,
    ) -> None:
        self.conn = conn
        self.cfg = cfg
        self.ollama = ollama or OllamaClient(url=cfg.summarization.ollama_url)

    def pending_session_ids(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT session_id FROM sessions "
            "WHERE summary IS NULL OR summary_stale = 1 "
            "ORDER BY last_ts DESC"
        ).fetchall()
        return [r["session_id"] for r in rows]

    def summarize_session(self, session_id: str) -> None:
        row = self.conn.execute(
            "SELECT source_path FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown session_id {session_id}")

        transcript = _build_transcript(
            Path(row["source_path"]), self.cfg.chunking, max_chars=20000
        )
        prompt = PROMPT_TEMPLATE.format(
            max_bullets=self.cfg.summarization.max_bullets,
            time_gap_min=self.cfg.summarization.time_gap_min_for_bullet,
            transcript=transcript,
        )
        try:
            response = self.ollama.generate(
                model=self.cfg.summarization.model, prompt=prompt
            )
        except OllamaError as e:
            msg = str(e).lower()
            if "not found" in msg or "try pulling" in msg or "model" in msg and "404" in msg:
                raise OllamaError(
                    f"ollama model '{self.cfg.summarization.model}' is not "
                    f"pulled. Fix one of:\n"
                    f"  1) Pull it:  ollama pull {self.cfg.summarization.model}\n"
                    f"  2) Or edit config.toml and set "
                    f"[summarization].model to a model you already have "
                    f"(run `ollama list` to see what's installed).\n"
                    f"Original error: {e}"
                ) from e
            raise

        self.conn.execute(
            "UPDATE sessions SET summary = ?, summary_model = ?, "
            "summarized_at = ?, summary_stale = 0 WHERE session_id = ?",
            (response, self.cfg.summarization.model, _now_iso(), session_id),
        )
        self.conn.commit()

    def run_once(self) -> int:
        """Summarize all pending sessions in the foreground."""
        processed = 0
        for sid in self.pending_session_ids():
            try:
                self.summarize_session(sid)
                processed += 1
            except OllamaError as e:
                print(f"[summarizer] ollama error on {sid}: {e}", file=sys.stderr)
        return processed


# --- Daemon -----------------------------------------------------------------


def pid_file_path(cfg: Config) -> Path:
    return Path(cfg.paths.db).parent / "summarize.pid"


def log_file_path(cfg: Config) -> Path:
    return Path(cfg.paths.db).parent / "summarize.log"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def daemon_status(cfg: Config, conn: sqlite3.Connection) -> dict:
    pid_path = pid_file_path(cfg)
    pid = None
    running = False
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text().strip())
            running = _pid_alive(pid)
        except ValueError:
            pass
    done = conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE summary IS NOT NULL"
    ).fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    return {
        "pid": pid if running else None,
        "running": running,
        "done": done,
        "total": total,
        "remaining": total - done,
    }


def stop_daemon(cfg: Config) -> bool:
    pid_path = pid_file_path(cfg)
    if not pid_path.exists():
        return False
    try:
        pid = int(pid_path.read_text().strip())
    except ValueError:
        pid_path.unlink(missing_ok=True)
        return False
    if not _pid_alive(pid):
        pid_path.unlink(missing_ok=True)
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pid_path.unlink(missing_ok=True)
        return False
    for _ in range(30):
        if not _pid_alive(pid):
            pid_path.unlink(missing_ok=True)
            return True
        time.sleep(0.5)
    return False


def run_daemon_loop(
    cfg: Config,
    conn: sqlite3.Connection,
    summarizer: Summarizer,
) -> None:
    stop = {"flag": False, "reason": None}

    log_path = log_file_path(cfg)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a")
    log.write(f"[{_now_iso()}] daemon starting (pid {os.getpid()})\n")
    log.flush()

    def _handle(signum, frame):  # pragma: no cover
        name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        stop["flag"] = True
        stop["reason"] = name
        try:
            log.write(f"[{_now_iso()}] received {name}, will finish current session and exit\n")
            log.flush()
        except Exception:
            pass

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGHUP, _handle)

    processed = 0
    while not stop["flag"]:
        pending = summarizer.pending_session_ids()
        if not pending:
            log.write(f"[{_now_iso()}] queue drained after {processed} sessions\n")
            break
        for sid in pending:
            if stop["flag"]:
                break
            tries = 0
            succeeded = False
            while tries < 3:
                try:
                    summarizer.summarize_session(sid)
                    log.write(f"[ok] {sid}\n")
                    log.flush()
                    succeeded = True
                    processed += 1
                    break
                except OllamaError as e:
                    tries += 1
                    log.write(f"[retry {tries}] {sid}: {e}\n")
                    log.flush()
                    time.sleep(30)
                except Exception as e:
                    # Unexpected failure: log the full exception so we can
                    # diagnose instead of crashing the whole daemon.
                    import traceback
                    log.write(f"[error] {sid}: {type(e).__name__}: {e}\n")
                    log.write(traceback.format_exc())
                    log.flush()
                    tries = 3
                    break
            if not succeeded:
                log.write(f"[skip] {sid}\n")
                log.flush()

    reason = stop["reason"] or "clean"
    log.write(f"[{_now_iso()}] daemon exiting ({reason}) after {processed} sessions\n")
    log.close()


def start_daemon(cfg: Config, conn_factory) -> None:
    """Fork + detach a daemon that summarizes until the queue drains."""
    pid_path = pid_file_path(cfg)
    if pid_path.exists():
        try:
            existing = int(pid_path.read_text().strip())
        except ValueError:
            existing = 0
        if existing and _pid_alive(existing):
            raise RuntimeError(f"daemon already running (pid {existing})")
        pid_path.unlink(missing_ok=True)

    pid = os.fork()
    if pid > 0:
        return
    # Child.
    os.setsid()
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()))
    try:
        conn = conn_factory()
        summ = Summarizer(conn, cfg)
        run_daemon_loop(cfg, conn, summ)
    finally:
        pid_path.unlink(missing_ok=True)
        os._exit(0)
