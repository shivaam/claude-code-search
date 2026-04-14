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

## Install + setup

```bash
pip install claude-code-search    # or: pipx install / uv tool install
ccsearch init                     # indexes, sets up daily auto-refresh, detects ollama
```

That's it. `init` walks you through everything:

```
  claude-code-search setup

[1/3] Indexing your conversations...
      found 937 files; 937 need indexing
      done in 5m12s: 937 files, 50999 chunks, 303 sessions.

[2/3] Setting up daily auto-index...
      installed: runs daily at 02:00

[3/3] Conversation outlines (optional, requires ollama)...
      ollama found, model gemma2:9b available.
      starting background summarizer (303 conversations to process)...

  Ready. Try: ccsearch "what was I working on last week"
```

No ollama? No problem — everything works without it. You just won't get bullet outlines in results.

## Usage

```bash
ccsearch "your query"               # semantic search (~3s)
ccsearch -g "#64827"                # exact text search (~0.2s, no model)
ccsearch show <id> --query "text"   # view messages around a match
ccsearch resume <id>                # cd + claude -r, back in the conversation
ccsearch stats                      # index health
```

`--compact` for one-line-per-result, `--json` for scripting, `-n 20` for more results.

## Configuration

```bash
ccsearch config                     # see current settings
ccsearch config --edit              # open in $EDITOR
```

Key settings in `config.toml` (auto-generated, all have sensible defaults):

| Setting | Default | What |
|---|---|---|
| `[embedding] model` | `BAAI/bge-small-en-v1.5` | Embedding model (~130 MB) |
| `[embedding] device` | `auto` | auto-detects mps/cuda/cpu |
| `[summarization] model` | `gemma2:9b` | Ollama model for outlines |
| `[search] top_n` | `10` | Results per query |
| `[schedule] hour` | `2` | Daily auto-index hour |

Swap embedding model: edit config, run `ccsearch index --rebuild`.
Swap summary model: edit config, `ollama pull <model>`, run `ccsearch summarize --daemon`.

## Commands

```
ccsearch init                        setup everything (run once)
ccsearch "query"                     semantic search
ccsearch -g "text"                   literal grep search (instant)
ccsearch show ID [--query Q]         view conversation context
ccsearch resume ID                   resume in Claude Code
ccsearch summarize --daemon/--stop   background outlines via ollama
ccsearch schedule --install/--uninstall  daily auto-index
ccsearch config [--edit]             view/edit settings
ccsearch stats                       index health
ccsearch --help                      full help
```

## How it works

Parses `.jsonl` from `~/.claude/projects/` → chunks messages → embeds with sentence-transformers → stores in SQLite + [sqlite-vec](https://github.com/asg017/sqlite-vec) → KNN search grouped by session. Grep mode (`-g`) skips all of that and does `LIKE '%query%'` directly on the chunks table.

## Tradeoffs

**Good at:** finding conversations by topic, exact ID lookup (`-g`), resuming past work, staying fully local.

**Limitations:** ~3s startup (torch import), English-biased embeddings, sqlite-vec caps KNN at k=4096 (fine for ~50K chunks), ollama needed for outlines only. macOS + Linux tested; Windows untested (`os.fork` in daemon).

## License

MIT
