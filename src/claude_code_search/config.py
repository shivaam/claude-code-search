"""TOML config loader with dataclass-typed settings."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class PathsConfig:
    root: str = "~/.claude/projects"
    db: str = "data/index.db"


@dataclass
class EmbeddingConfig:
    model: str = "BAAI/bge-small-en-v1.5"
    # "auto" picks mps on Apple Silicon, cuda on NVIDIA, else cpu. Explicit
    # values "mps" | "cuda" | "cpu" force a specific backend.
    device: str = "auto"
    batch_size: int = 64


@dataclass
class ChunkingConfig:
    max_chars: int = 1500
    overlap: int = 200
    min_chars: int = 10
    include_sidechains: bool = False


@dataclass
class SummarizationConfig:
    enabled: bool = True
    model: str = "gemma2:9b"
    ollama_url: str = "http://localhost:11434"
    max_bullets: int = 8
    time_gap_min_for_bullet: int = 60
    re_summarize_on_growth: bool = True
    re_summarize_threshold: int = 10  # only re-summarize when >= this many new msgs


@dataclass
class SearchConfig:
    top_n: int = 10
    # k_chunks is how many raw chunk hits we fetch before grouping by session.
    # sqlite-vec hard-caps this at 4096; defaulting to the max keeps recall
    # high on large indexes without meaningfully affecting latency on small
    # ones (KNN scan is fast relative to Python overhead either way).
    k_chunks: int = 4096
    snippet_chars: int = 240


@dataclass
class ShowConfig:
    default_context: int = 4


@dataclass
class OutputConfig:
    strip_project_prefix: str = ""


@dataclass
class ScheduleConfig:
    hour: int = 2           # run daily at this hour (0-23)
    minute: int = 0
    summarize: bool = True  # also run summarize --all after indexing


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    summarization: SummarizationConfig = field(default_factory=SummarizationConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    show: ShowConfig = field(default_factory=ShowConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)


def _merge_section(target: Any, values: dict[str, Any]) -> None:
    for key, val in values.items():
        if hasattr(target, key):
            setattr(target, key, val)
        # Unknown keys are silently ignored so config files can tolerate
        # forward-compatible options.


def load_config(path: Path, project_root: Path | None = None) -> Config:
    """Load a Config from a TOML file. Missing file → all defaults."""
    cfg = Config()
    if path.exists():
        data = tomllib.loads(path.read_text())
        for section_name in [f.name for f in fields(Config)]:
            if section_name in data and isinstance(data[section_name], dict):
                _merge_section(getattr(cfg, section_name), data[section_name])

    # Expand root.
    cfg.paths.root = str(Path(cfg.paths.root).expanduser())

    # Resolve db path relative to project_root or config file dir.
    db_path = Path(cfg.paths.db).expanduser()
    if not db_path.is_absolute():
        base = project_root if project_root else path.parent
        db_path = (base / db_path).resolve()
    cfg.paths.db = str(db_path)

    return cfg


def write_default_config(path: Path) -> None:
    """Write a commented default config to path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_DEFAULT_TOML)


_DEFAULT_TOML = """\
# claude-code-search config. All values here are the defaults.

[paths]
root = "~/.claude/projects"
db   = "data/index.db"

[embedding]
model      = "BAAI/bge-small-en-v1.5"
device     = "auto"              # auto | mps | cuda | cpu   (auto picks best available)
batch_size = 64

[chunking]
max_chars          = 1500
overlap            = 200
min_chars          = 10
include_sidechains = false

[summarization]
enabled                 = true
model                   = "gemma2:9b"
ollama_url              = "http://localhost:11434"
max_bullets             = 8
time_gap_min_for_bullet = 60
re_summarize_on_growth  = true
re_summarize_threshold  = 10       # only re-summarize after this many new messages

[search]
top_n         = 10
k_chunks      = 4096           # raw chunks scanned before session grouping (sqlite-vec max)
snippet_chars = 240

[show]
default_context = 4

[output]
strip_project_prefix = ""

[schedule]
hour      = 2              # run daily at this hour (0-23, local time)
minute    = 0
summarize = true           # also run summarize --all after indexing
"""
