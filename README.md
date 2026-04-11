# claude-code-search

**Local semantic search over your Claude Code conversation history.** Find past
conversations by meaning, not keyword. Returns the session IDs you need to
resume them in Claude Code.

Built because the existing solution for this (episodic-memory) returned zero
results for real queries on a 50,000-chunk corpus. This is ~1,000 lines of
Python you fully own, with a single-file SQLite index you can inspect with
any tool.

```
$ ccsearch "scheduler bug"

 1. 0.39  2026-04-10  airflow     aaaaaaaa-bbbb-cccc-dddd-111111111111  (754 msgs)
    Fix gaps and market.
      • Debugged Airflow scheduler re-queueing bug
      • Traced it to a DAG parse retry loop
      • Switched to provider package test failures
    ↳ Scheduler hasn't picked up the task yet. Wait a bit longer:
    $ (cd /Users/you/workspace/airflow && claude -r aaaaaaaa-bbbb-cccc-dddd-111111111111)
    ──────────────────────────────────────────────────────────────────────────────
 2. 0.37  2026-04-10  airflow     aaaaaaaa-bbbb-cccc-dddd-222222222222  (176 msgs)
    Build local semantic search for Claude conversations
    ↳ for example can you find my conversation using the mcp related to scheduler bug
    $ (cd /Users/you/workspace/airflow && claude -r aaaaaaaa-bbbb-cccc-dddd-222222222222)
```

## What it does

- **Indexes** every `.jsonl` file under `~/.claude/projects/` (where Claude
  Code stores conversation history) with a local embedding model.
- **Searches** semantically across all of them — not just keyword matches.
  Results are grouped by session, ranked by the best-matching chunk.
- **Shows context** around a match so you can scan it before resuming.
- **Resumes** a conversation by `cd`ing into the original working directory
  and running `claude -r <session_id>`.
- **Summarizes** long conversations with a local Ollama model into
  time-segmented outlines, rendered inline in search results.
- **Respects your manual renames.** If you've given a session a custom title
  in Claude Code, that title takes priority in search results.

## Requirements

- **Python 3.11+**
- **Claude Code** installed and in use (the tool reads `~/.claude/projects/`).
- **Ollama** (optional, for conversation outlines). If you don't want outlines,
  it's fine to skip — search works without them.
- **~200 MB disk** for the initial index + ~130 MB for the embedding model.
  Summarization adds a few GB for the Ollama model (5.4 GB for the default
  `gemma2:9b`).

## Install

```bash
git clone https://github.com/<you>/claude-code-search.git ~/workspace/claude-code-search
cd ~/workspace/claude-code-search
python3 -m venv .venv
.venv/bin/pip install -e .
```

Then put `ccsearch` on your PATH — either symlink:

```bash
ln -s ~/workspace/claude-code-search/.venv/bin/ccsearch ~/.local/bin/ccsearch
```

or activate the venv when you need it:

```bash
source ~/workspace/claude-code-search/.venv/bin/activate
```

First run of `ccsearch index` will download the embedding model
(`BAAI/bge-small-en-v1.5`, ~130 MB) into `~/.cache/huggingface/`.

## Quickstart

```bash
ccsearch index                       # build the index (first run: ~5-10 min)
ccsearch "scheduler bug"             # search
ccsearch show <session-id> --query "scheduler bug" --context 3
ccsearch resume <session-id>         # cd + claude -r, ready to go
ccsearch stats                       # counts and disk usage
```

Optional outlines via Ollama:

```bash
ollama pull gemma2:9b                # or a smaller model, see below
ccsearch summarize --daemon          # backgrounds, resumable, ~30 sec/session
ccsearch summarize --status          # progress check
ccsearch summarize --stop            # pause; resume anytime with --daemon
```

## Commands

| Command | Purpose |
|---|---|
| `ccsearch index` | Walk `~/.claude/projects/` and incrementally index new/changed `.jsonl` files. Incremental by `mtime`. |
| `ccsearch "query"` | Shortcut for `ccsearch search "query"`. Returns top 10 conversations by default. |
| `ccsearch search QUERY [-n N] [--compact] [--project SUBSTR] [--json]` | Full search. `--compact` for one-line-per-result, `--json` for scripting. |
| `ccsearch show SESSION_ID [--query Q] [--context N] [--full]` | Print messages around a matching chunk, or the whole conversation. |
| `ccsearch resume SESSION_ID` | `cd` into the original cwd and `exec claude -r <session_id>`. |
| `ccsearch summarize --daemon` | Start a background daemon that generates timestamped outlines via Ollama. Pausable, resumable, stateless. |
| `ccsearch summarize --status` / `--stop` / `--session ID` / `--all` | Status / clean stop / one-shot / foreground bulk. |
| `ccsearch stats` | Index size, sessions, chunks, summary count, daemon state. |

## How the search works

1. **Parser** reads each `.jsonl`, keeping only `user` and `assistant` turns
   plus `tool_result` text. Tool call blobs, thinking traces, images,
   file-history snapshots, and subagent sidechains are dropped.
2. **Chunker** splits long messages into ~1500-character windows with
   200-character overlap, preferring paragraph/sentence boundaries.
3. **Embedder** turns each chunk into a 384-dimensional vector using
   `BAAI/bge-small-en-v1.5` (sentence-transformers, MPS/CUDA/CPU
   auto-detected, L2-normalized).
4. **Store** writes the chunks to a regular SQLite table and the vectors to
   a `sqlite-vec` virtual table, both living in
   `data/index.db` (one file).
5. **Searcher** embeds your query with the same model, runs a KNN scan
   against `vec_chunks` (default `k = 4096`, the sqlite-vec maximum), groups
   the hits by `session_id`, and returns the top-N conversations each with
   their best-scoring chunk as representative.
6. **Summarizer** (optional) pulls each conversation's full transcript, sends
   it to a local Ollama model with a prompt asking for a bulleted,
   timestamped outline, and writes the result back to the `sessions` table.
   Runs as a detached daemon with a pid file, signal-safe, fully resumable.

## Config

Everything that matters is in a single TOML file at
`~/workspace/claude-code-search/config.toml`, generated with sensible defaults
on first run. CLI flags override file values for one-off usage (e.g. `-n 50`,
`--project airflow`, etc.).

```toml
[paths]
root = "~/.claude/projects"       # where Claude Code stores conversations
db   = "data/index.db"             # index location (single SQLite file)

[embedding]
model      = "BAAI/bge-small-en-v1.5"
device     = "auto"                # auto | mps | cuda | cpu
batch_size = 64

[chunking]
max_chars          = 1500          # window size per chunk
overlap            = 200           # sliding overlap between chunks
min_chars          = 10            # drop sub-10-char messages (tool ACKs)
include_sidechains = false         # include subagent trace files

[summarization]
enabled                 = true
model                   = "gemma2:9b"
ollama_url              = "http://localhost:11434"
max_bullets             = 8
time_gap_min_for_bullet = 60
re_summarize_on_growth  = true

[search]
top_n         = 10                 # top N conversations returned per query
k_chunks      = 4096               # raw chunks scanned before session grouping
snippet_chars = 240

[show]
default_context = 4                # messages before/after match for `show`

[output]
strip_project_prefix = "-Users-yourname-workspace-"
```

### Default models — why these, and swapping them

**Embedding: `BAAI/bge-small-en-v1.5`.** Small (~130 MB, 384-dim), fast on CPU,
excellent on MPS, and one of the best small retrieval models available today.
Good default for every machine. If you want slightly better quality and don't
mind 2x the storage/compute, try `BAAI/bge-base-en-v1.5` (768-dim) — edit
`config.toml` and run `ccsearch index --rebuild` (embeddings are not
interoperable between models).

**Summarization: `gemma2:9b`.** Chosen because it actually follows the
bulleted-timestamp output format that `ccsearch` asks for. Most 3B models
drift into prose. Needs ~5.4 GB and ~10–15 seconds per summary on M4.

If you'd rather use a smaller model, edit `config.toml`:

```toml
[summarization]
model = "llama3.2:3b"      # 2 GB, ~3-5 sec/summary, worse format adherence
# or
model = "qwen2.5:7b"       # 4.7 GB, good balance
```

Then pull it (`ollama pull <model>`) and run `ccsearch summarize --daemon`.
The daemon picks up where any previous run left off.

**If you don't want outlines at all**, set `[summarization].enabled = false`
or just never run `ccsearch summarize`. Search still works perfectly — the
title fallback is your own custom titles (if set), Claude Code's
auto-generated `ai_title`, or the first user message. `ccsearch` will print
a one-line hint at the bottom of search results reminding you that outlines
are available.

### What gets indexed, what doesn't

| Content | Included? | Why |
|---|---|---|
| User text messages | Yes | Your prompts |
| Assistant text blocks | Yes | Claude's replies |
| Tool result text | Yes | Real debugging discussion lives here |
| Tool invocation args | No | Big JSON blobs, not conversational |
| Thinking blocks | No | Internal reasoning, not dialogue |
| Images | No | Not text |
| Attachment records | No | Metadata wrapper, not content |
| File-history snapshots | No | Internal state, not content |
| Queue operations | No | Internal |
| Subagent sidechains | Off by default, configurable | Derivative of main thread |

## Pros

- **Actually finds things.** 79% recall on a realistic query (`"scheduler
  bug"`) on a 50K-chunk corpus. Top-10 results include the conversations you
  would have manually picked.
- **Local only.** No API keys. No network after the initial embedding model
  download. No daemon except the one you opt into for summaries. No cloud
  storage.
- **One file.** The entire index is a single SQLite file you can `cp` back
  up, `sqlite3` inspect, or `rm` to start over.
- **Sub-second search** on 50K vectors (brute-force KNN from sqlite-vec;
  doesn't need an ANN index at this scale).
- **Incremental indexing.** Re-running `ccsearch index` only processes files
  with a newer `mtime` than the last run. New conversations add in seconds.
- **Respects your own titles.** If you've renamed sessions in Claude Code,
  those names take priority over Claude's auto-generated ones.
- **Resumable background summarization.** Stop anytime, resume anytime,
  safe across crashes. The DB is the state; no bookmark file.
- **Composable output.** Pretty vertical cards by default (with bullet
  outlines and colors in a TTY), `--compact` one-line-per-result for quick
  scanning, `--json` for scripting. Respects `NO_COLOR`.
- **Short code.** ~1,000 lines of Python. You can read the whole thing in an
  hour. No framework, no ORM, no dependency injection, no plugin system.
  Just argparse + sqlite-vec + sentence-transformers.
- **Tested on real data.** Acceptance test is "does `scheduler bug` find
  conversations about scheduler bugs on this specific user's 50K-chunk
  corpus" — not a synthetic benchmark.

## Cons and limitations

- **Startup cost per invocation is ~3 seconds** because `sentence-transformers`
  and `torch` are heavy imports. Fine for interactive use, bad for scripting
  tight loops. If you need faster, the JSON output lets you batch queries
  via a small wrapper.
- **No ANN index.** KNN is brute-force. At this scale (50K vectors) it's
  sub-second. At 250K it's a few seconds. Past ~1M vectors (or roughly
  ~20× your current corpus), recall will degrade because `sqlite-vec` caps
  `k` at 4096 per query — you'd be scanning <2% of candidates. At that
  point, partition by date or switch to a store with a real ANN index.
- **Embedding model is English-biased.** `bge-small-en-v1.5` is trained
  primarily on English. Non-English conversations will still search but with
  reduced quality.
- **Summaries drift off-format ~20% of the time.** Even with `gemma2:9b`,
  local models occasionally answer the prompt with prose or clarifying
  questions instead of bullets. The display layer filters non-bullet lines
  out, which masks most of the damage, but occasionally a session shows no
  outline even though one was generated.
- **No automated re-indexing.** You have to manually run `ccsearch index`
  after accumulating new Claude Code conversations. A launchd agent or git
  hook is easy to add (not shipped by default).
- **macOS/Linux only (probably).** Tested on macOS (Apple Silicon and
  Intel), should work on Linux. Windows is untested — the daemon uses
  `os.fork()` and `os.setsid()` which aren't available on Windows. If
  someone wants to port it, the fix is ~20 lines (use `multiprocessing` or
  a Windows Service shim).
- **Extended-thinking blocks aren't searched.** Claude's internal reasoning
  is intentionally excluded from the index because it's not dialogue. If
  you find yourself needing to search across reasoning content, that's a
  config knob worth adding.
- **Resume assumes `claude` is on your PATH.** `ccsearch resume` just
  `cd`s and `exec`s `claude -r <session_id>`. If your Claude Code binary is
  somewhere else, add an alias or shim.

## Known failure modes and how to diagnose

**Search returns nothing reasonable.** Check:
```bash
ccsearch stats
```
Make sure `files indexed`, `sessions`, and `chunks` are non-zero. If they
are, try bumping `k_chunks` in config.toml to 4096 (the sqlite-vec max) and
re-run without re-indexing.

**`ccsearch index` hangs on first run.** The embedding model is downloading.
Check `~/.cache/huggingface/hub/`. If your network is slow, give it ~5
minutes for ~130 MB.

**`ccsearch summarize --daemon` errors with "model not found".** Either
pull the model (`ollama pull <model>`) or edit `config.toml` to point at a
model you already have (`ollama list`).

**`ccsearch "query"` prints `Warning: unauthenticated requests to HF Hub`.**
This is benign — we force offline mode via `HF_HUB_OFFLINE=1` once the
model is cached, so this warning shouldn't appear. If it does, the model
isn't cached yet; run `CCSEARCH_HF_ONLINE=1 ccsearch index` once to allow
the download.

**Daemon exits unexpectedly.** Check `data/summarize.log` — it logs signal
arrivals, errors with tracebacks, retry counts, and exit reasons. The
daemon is resumable; just run `ccsearch summarize --daemon` again and it
picks up where it left off (the DB tracks which sessions are summarized).

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  ccsearch CLI  (Python, argparse)                                    │
│                                                                      │
│  index   search   show   resume   summarize   stats                  │
│     │       │        │       │         │         │                  │
│     ▼       ▼        ▼       ▼         ▼         ▼                  │
│  Indexer → Searcher → Viewer → os.exec → Summarizer → (read-only)    │
│     │       │                              │                        │
│     ▼       ▼                              ▼                        │
│  ┌──────────────────────────────────────────────────────┐            │
│  │  data/index.db     (SQLite + sqlite-vec)             │            │
│  │    - files         (path → mtime for incremental)    │            │
│  │    - sessions      (one row per conversation)        │            │
│  │    - chunks        (metadata + raw text)             │            │
│  │    - vec_chunks    (embeddings, vec0 virtual table)  │            │
│  └──────────────────────────────────────────────────────┘            │
└──────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ ~/.claude/projects  │  (read-only input)
                 │     **/*.jsonl      │
                 └────────────────────┘
```

Source layout:

```
src/claude_code_search/
  cli.py              argparse entrypoint, subcommand dispatch, output formatters
  config.py           TOML loader, typed dataclasses, defaults
  models.py           Message / SessionMeta / Chunk / Hit dataclasses
  parser.py           .jsonl → list[Message] + SessionMeta
  chunker.py          Message → list[Chunk] with sentence-aware cuts
  embedder.py         sentence-transformers wrapper, device auto-detection
  store.py            SQLite connection, schema, idempotent migrations
  walker.py           file discovery + mtime-based incremental decision
  indexer.py          orchestrates parse → chunk → embed → write
  searcher.py         KNN + session grouping + metadata enrichment
  viewer.py           pretty-printer for `ccsearch show`
  ollama_client.py    httpx wrapper for /api/generate
  summarizer.py       summarize_session + run_daemon + pid file mgmt
```

## Design decisions worth knowing

All captured in the spec at
[docs/superpowers/specs/](docs/superpowers/specs/).

- **Why sqlite-vec and not chromadb/lancedb/faiss:** one loadable extension,
  zero daemons, co-located with metadata, single file. Trade-off: flat KNN
  only, with a hard cap of `k=4096`.
- **Why brute-force KNN:** at 50k vectors the math is sub-second; adding an
  ANN index is engineering overhead we don't need.
- **Why bge-small not MiniLM:** bge-small strictly dominates MiniLM on MTEB
  retrieval at the same 384 dimensions. Same speed, better recall.
- **Why per-message chunking:** keeps session/role/timestamp metadata clean
  per chunk and makes snippets read naturally. Alternative was fixed sliding
  windows ignoring message boundaries.
- **Why a background daemon for summaries:** summarization is ~10 sec/session
  with a 9B model × 300 sessions = an hour of compute we don't want to
  block `ccsearch index` on. The daemon is resumable so you can pause
  whenever.
- **Why TOML config and not env vars or flags-only:** flags get tedious,
  env vars are invisible to users, a single TOML file is discoverable and
  editable by humans.

## Contributing / development

```bash
git clone https://github.com/<you>/claude-code-search
cd claude-code-search
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q           # 31 tests, <1 sec
```

The test suite uses a tiny fixture `.jsonl` file and a fake embedder (pure
numpy), so it runs offline and in under a second. The only slow test is a
smoke test that downloads the real bge model; set `CCSEARCH_SKIP_SLOW=1` to
skip it.

There's a spec (`docs/superpowers/specs/`) and an implementation plan
(`docs/superpowers/plans/`) checked in, produced by the brainstorming and
writing-plans workflows. They're the authoritative record of what this
thing is, why, and how.

## License

MIT. Do what you want.
