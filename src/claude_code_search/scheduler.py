"""Install / uninstall / status for scheduled daily indexing.

On macOS: writes a LaunchAgent plist.
On Linux: writes a crontab entry.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

from .config import Config

LABEL = "com.ccsearch.index"
_PLIST_DIR = Path.home() / "Library" / "LaunchAgents"


def _ccsearch_bin() -> str:
    """Resolve the absolute path to the ccsearch entrypoint."""
    # Prefer the symlink on PATH (e.g. ~/.local/bin/ccsearch).
    found = shutil.which("ccsearch")
    if found:
        return str(Path(found).resolve())
    # Fall back to the venv binary next to this file.
    here = Path(__file__).resolve()
    venv_bin = here.parents[2] / ".venv" / "bin" / "ccsearch"
    if venv_bin.exists():
        return str(venv_bin)
    return "ccsearch"  # hope it's on PATH at runtime


def _config_path(cfg: Config) -> str | None:
    """Return the config.toml path if one exists in the project root."""
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / "config.toml"
        if candidate.exists():
            return str(candidate)
    return None


# --- macOS LaunchAgent ----------------------------------------------------


def _plist_path() -> Path:
    return _PLIST_DIR / f"{LABEL}.plist"


def _build_plist(cfg: Config) -> str:
    bin_path = _ccsearch_bin()
    config_flag = ""
    config_path = _config_path(cfg)
    if config_path:
        config_flag = f"""
        <string>--config</string>
        <string>{config_path}</string>"""

    # Build the command: index, then optionally summarize --all.
    program_args = f"""        <string>{bin_path}</string>{config_flag}
        <string>index</string>"""

    # If summarize is enabled, chain a second invocation via bash -c.
    if cfg.schedule.summarize:
        cmd_index = f"{bin_path} index"
        cmd_summarize = f"{bin_path} summarize --all"
        if config_path:
            cmd_index += f" --config {config_path}"
            cmd_summarize = f"{bin_path} --config {config_path} summarize --all"
        combined = f"{cmd_index} && {cmd_summarize}"
        program_args = f"""        <string>/bin/bash</string>
        <string>-c</string>
        <string>{combined}</string>"""

    log_dir = Path(cfg.paths.db).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    return dedent(f"""\
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
      "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
    <plist version="1.0">
    <dict>
        <key>Label</key>
        <string>{LABEL}</string>
        <key>ProgramArguments</key>
        <array>
    {program_args}
        </array>
        <key>StartCalendarInterval</key>
        <dict>
            <key>Hour</key>
            <integer>{cfg.schedule.hour}</integer>
            <key>Minute</key>
            <integer>{cfg.schedule.minute}</integer>
        </dict>
        <key>StandardOutPath</key>
        <string>{log_dir / "schedule-stdout.log"}</string>
        <key>StandardErrorPath</key>
        <string>{log_dir / "schedule-stderr.log"}</string>
        <key>RunAtLoad</key>
        <false/>
    </dict>
    </plist>
    """)


def install_macos(cfg: Config) -> str:
    _PLIST_DIR.mkdir(parents=True, exist_ok=True)
    plist = _plist_path()
    # Unload first if already loaded.
    if plist.exists():
        subprocess.run(
            ["launchctl", "unload", str(plist)],
            capture_output=True,
        )
    plist.write_text(_build_plist(cfg))
    subprocess.run(
        ["launchctl", "load", str(plist)],
        capture_output=True,
        check=True,
    )
    return (
        f"installed: {plist}\n"
        f"runs daily at {cfg.schedule.hour:02d}:{cfg.schedule.minute:02d}"
        + (f" (index + summarize)" if cfg.schedule.summarize else " (index only)")
    )


def uninstall_macos() -> str:
    plist = _plist_path()
    if not plist.exists():
        return "not installed"
    subprocess.run(["launchctl", "unload", str(plist)], capture_output=True)
    plist.unlink(missing_ok=True)
    return f"uninstalled: removed {plist}"


def status_macos(cfg: Config) -> str:
    plist = _plist_path()
    if not plist.exists():
        return "not installed"
    result = subprocess.run(
        ["launchctl", "list", LABEL],
        capture_output=True,
        text=True,
    )
    loaded = result.returncode == 0
    log_dir = Path(cfg.paths.db).parent
    stdout_log = log_dir / "schedule-stdout.log"
    last_run = ""
    if stdout_log.exists():
        import os

        mtime = os.path.getmtime(stdout_log)
        import datetime

        last_run = datetime.datetime.fromtimestamp(mtime).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    lines = [
        f"plist:     {plist}",
        f"loaded:    {'yes' if loaded else 'no'}",
        f"schedule:  daily at {cfg.schedule.hour:02d}:{cfg.schedule.minute:02d}",
        f"summarize: {'yes' if cfg.schedule.summarize else 'no'}",
    ]
    if last_run:
        lines.append(f"last run:  {last_run}")
    lines.append(f"stdout:    {log_dir / 'schedule-stdout.log'}")
    lines.append(f"stderr:    {log_dir / 'schedule-stderr.log'}")
    return "\n".join(lines)


# --- Linux crontab --------------------------------------------------------


_CRON_TAG = "# ccsearch-scheduled-index"


def _cron_line(cfg: Config) -> str:
    bin_path = _ccsearch_bin()
    config_path = _config_path(cfg)
    cmd = f"{bin_path} index"
    if config_path:
        cmd = f"{bin_path} --config {config_path} index"
    if cfg.schedule.summarize:
        summarize = f"{bin_path} summarize --all"
        if config_path:
            summarize = f"{bin_path} --config {config_path} summarize --all"
        cmd = f"{cmd} && {summarize}"
    log_dir = Path(cfg.paths.db).parent
    return (
        f"{cfg.schedule.minute} {cfg.schedule.hour} * * * "
        f"{cmd} >> {log_dir / 'schedule-stdout.log'} "
        f"2>> {log_dir / 'schedule-stderr.log'} {_CRON_TAG}"
    )


def install_linux(cfg: Config) -> str:
    line = _cron_line(cfg)
    # Read existing crontab, remove old entries, add new one.
    result = subprocess.run(
        ["crontab", "-l"], capture_output=True, text=True
    )
    existing = result.stdout if result.returncode == 0 else ""
    cleaned = "\n".join(
        l for l in existing.splitlines() if _CRON_TAG not in l
    )
    new_crontab = (cleaned.rstrip() + "\n" + line + "\n").lstrip()
    subprocess.run(
        ["crontab", "-"], input=new_crontab, text=True, check=True
    )
    return (
        f"installed crontab entry\n"
        f"runs daily at {cfg.schedule.hour:02d}:{cfg.schedule.minute:02d}"
        + (f" (index + summarize)" if cfg.schedule.summarize else " (index only)")
    )


def uninstall_linux() -> str:
    result = subprocess.run(
        ["crontab", "-l"], capture_output=True, text=True
    )
    if result.returncode != 0 or _CRON_TAG not in result.stdout:
        return "not installed"
    cleaned = "\n".join(
        l for l in result.stdout.splitlines() if _CRON_TAG not in l
    )
    subprocess.run(
        ["crontab", "-"], input=cleaned + "\n", text=True, check=True
    )
    return "uninstalled: removed crontab entry"


def status_linux(cfg: Config) -> str:
    result = subprocess.run(
        ["crontab", "-l"], capture_output=True, text=True
    )
    if result.returncode != 0 or _CRON_TAG not in result.stdout:
        return "not installed"
    line = next(
        l for l in result.stdout.splitlines() if _CRON_TAG in l
    )
    return f"installed: {line}"


# --- Dispatch by platform -------------------------------------------------


def install(cfg: Config) -> str:
    if platform.system() == "Darwin":
        return install_macos(cfg)
    return install_linux(cfg)


def uninstall(cfg: Config) -> str:
    if platform.system() == "Darwin":
        return uninstall_macos()
    return uninstall_linux()


def status(cfg: Config) -> str:
    if platform.system() == "Darwin":
        return status_macos(cfg)
    return status_linux(cfg)
