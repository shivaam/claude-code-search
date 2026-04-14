# claude-code-search

Local semantic search over your Claude Code conversations. Find past sessions by meaning, get the ID, resume them.

```
$ ccsearch "scheduler bug"

 1. 0.39  2026-04-10  airflow  940abfde-...  (754 msgs)
    Investigate concurrency bugs in Airflow scheduler
      • Debugged scheduler re-queueing bug
      • Traced it to DAG parse retry loop
    ↳ Scheduler hasn't picked up the task yet...
    cd ~/workspace/airflow && claude -r 940abfde-...
```

## Install

```bash
pip install claude-code-search
```

Or isolated (recommended):

```bash
pipx install claude-code-search    # or
uv tool install claude-code-search # fastest
```

## Usage

```bash
ccsearch index                  # build the index (~5 min first run)
ccsearch "your query"           # semantic search
ccsearch -g "#64827"            # exact/literal search (instant, no model)
ccsearch show <id> --query "x"  # see messages around a match
ccsearch resume <id>            # cd + claude -r, ready to go
```

That's it. The first `ccsearch index` downloads a ~130 MB embedding model and indexes `~/.claude/projects/`. After that, searches take ~3 seconds (semantic) or ~0.2 seconds (grep).

## Optional: conversation outlines

If you have [ollama](https://ollama.com) installed, you can generate timestamped outlines for each conversation:

```bash
ollama pull gemma2:9b
ccsearch summarize --daemon     # runs in background, ~30s per conversation
ccsearch summarize --stop       # pause anytime, resume with --daemon
ccsearch summarize --status     # check progress
```

Outlines appear in search results once generated. Everything works fine without them.

## Keep it up to date

```bash
ccsearch index                  # re-run manually (incremental, seconds)
ccsearch schedule --install     # or auto-run daily (macOS LaunchAgent / Linux cron)
ccsearch schedule --uninstall   # remove the schedule
```

## Configuration

```bash
ccsearch config                 # see current settings
ccsearch config --edit          # open config.toml in $EDITOR
```

Everything is in one TOML file, auto-generated on first run. Key settings:

| Setting | Default | What it does |
|---|---|---|
| `[embedding] model` | `BAAI/bge-small-en-v1.5` | Embedding model (384-dim, ~130 MB) |
| `[embedding] device` | `auto` | `auto` picks mps/cuda/cpu |
| `[summarization] model` | `gemma2:9b` | Ollama model for outlines |
| `[summarization] re_summarize_threshold` | `10` | Re-summarize after N new messages |
| `[search] top_n` | `10` | Results per query |
| `[schedule] hour` | `2` | Daily auto-index hour (0-23) |

To swap the embedding model, edit config and `ccsearch index --rebuild`. To swap the summary model, edit config and `ccsearch summarize --daemon` (picks up new model on next run).

## All commands

| Command | What |
|---|---|
| `ccsearch index [--rebuild]` | Index conversations (incremental by default) |
| `ccsearch "query" [-n N]` | Semantic search |
| `ccsearch -g "text" [-n N]` | Literal grep search (no model, instant) |
| `ccsearch show ID [--query Q] [--context N] [--full]` | View conversation around a match |
| `ccsearch resume ID` | Resume a conversation in Claude Code |
| `ccsearch summarize --daemon/--stop/--status/--all` | Background outline generation via ollama |
| `ccsearch schedule --install/--uninstall` | Daily auto-index (macOS/Linux) |
| `ccsearch config [--edit] [--path]` | View/edit configuration |
| `ccsearch stats` | Index health: files, sessions, chunks, summaries |

Add `--compact` or `--json` to any search for one-line or machine-readable output.

## How it works

1. **Parses** `.jsonl` files from `~/.claude/projects/` — keeps user/assistant text + tool output, drops tool calls, images, thinking blocks
2. **Chunks** long messages into ~1500-char windows with overlap
3. **Embeds** with sentence-transformers (bge-small-en-v1.5, runs on MPS/CUDA/CPU)
4. **Stores** chunks + 384-dim vectors in a single SQLite file using [sqlite-vec](https://github.com/asg017/sqlite-vec)
5. **Searches** via KNN, groups hits by session, enriches with titles and outlines
6. **Grep mode** (`-g`) bypasses all of the above — just `LIKE '%query%'` on the chunks table

## Tradeoffs

**Good at:** finding conversations by topic, resuming past work, exact identifier lookup (`-g`), staying local and private.

**Less good at:** non-English text (bge-small is English-biased), very large corpora (sqlite-vec caps KNN at k=4096 — fine for ~50K chunks, may need partitioning past ~250K), first-invocation speed (~3s to load torch).

**Requires:** Python 3.11+, ~300 MB disk (model + index). Ollama only needed for outlines (optional). Tested on macOS (Apple Silicon + Intel) and should work on Linux. Windows untested (daemon uses `os.fork`).

## License

MIT
