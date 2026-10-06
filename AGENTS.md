# AGENTS.md

Guidelines for AI coding agents (Gemini, Claude, Copilot, etc.) working in this repo.

---

## Repo in one sentence

Local RAG pipeline over ETSI GS MEC PDFs: ingest → tri-vector Qdrant (dense + ColBERT + sparse) → hybrid search (server-side RRF fusion) → OpenRouter LLM answer.

---

## Package manager — `uv` only

```powershell
uv add <pkg>          # add dependency
uv run <cmd>          # run in venv
uv sync               # install all deps
```

Never use `pip` directly. Do not touch `.venv/`.

---

## Project layout

```
src/etsi_mec_agent/
  config.py          # frozen Settings dataclass — all env vars here
  store.py           # Qdrant client + ensure_collection()
  ingest.py          # PDF → chunks → embed → upsert  (CLI: uv run python -m etsi_mec_agent.ingest)
  search.py          # hybrid search + LLM answer       (CLI: uv run python -m etsi_mec_agent.search)
  dedup.py           # rank shaping (stitch + dedup) — must stay stdlib-only, so scripts/check_dedup_stitch.py runs without ONNX
  agent.py           # monitor entrypoint
  tools/
    clip_embed.py    # CLIP image embedder (CPU-safe, lazy-loads)
    exa_search.py    # Exa API search (direct HTTP, no SDK)
    monitor_deliver.py  # polls ETSI deliver for new PDFs
    spec_sync.py     # ETSI Forge sync

scripts/
  backfill_clip.py          # add 'clip' vectors to existing collection (no re-ingest)
  deduplicate_collection.py # remove duplicate chunks in-place (no re-embed)
  audit_specs.py            # report data/specs files and manifest keys naming the wrong spec
  check_dedup_stitch.py     # asserts for dedup.py; no models, no Qdrant
  eval_rag.py               # golden-question recall@k, both paths + the --answer budget
  migrate_add_sparse.py     # copy points into a new collection that has 'sparse' (no re-embed)
  graph_visualize.py        # force-directed spec relationship graph

data/
  specs/      # downloaded PDFs (MECxxx.pdf)
  diagrams/   # extracted PNG diagrams (MECxxx.pdf-PPPP-NN.png)
```

---

## Environment

Copy `.env.example` → `.env` and fill in:

| Variable | Required | Notes |
|----------|----------|-------|
| `OPENROUTER_API_KEY` | For `--answer` | Free tier: `openrouter/free` |
| `EXA_API_KEY` | For monitor | Exa search API |
| `QDRANT_HOST` | No | Default `localhost` |
| `QDRANT_PORT` | No | Default `6333` |

Qdrant must be running in Docker before any ingest or search:

```powershell
docker run -d -p 6333:6333 -p 6334:6334 `
  -v ${PWD}/qdrant_data:/qdrant/storage `
  qdrant/qdrant
```

---

## Common commands

```powershell
# Ingest all PDFs in data/specs/
uv run python -m etsi_mec_agent.ingest

# Ingest only new PDFs (skip already-indexed)
uv run python -m etsi_mec_agent.ingest --skip-existing

# Full re-ingest (wipes collection first)
uv run python -m etsi_mec_agent.ingest --recreate-index

# Search
uv run python -m etsi_mec_agent.search "What is Mp1?"
uv run python -m etsi_mec_agent.search "Mm4 role" --use-bm25 --answer
uv run python -m etsi_mec_agent.search "Mm4 role" --use-bm25 --answer --show-reasoning

# Clean duplicate chunks in-place (no re-embedding, ~10 min)
uv run python scripts/deduplicate_collection.py --dry-run
uv run python scripts/deduplicate_collection.py

# Golden-question retrieval eval (recall@k for both paths + the --answer budget)
uv run python scripts/eval_rag.py --top-k 5

# Visualise spec relationships
uv run python scripts/graph_visualize.py --limit 3000 --threshold 0.55
```

---

## Search flags

| Flag | Effect |
|------|--------|
| `--use-bm25` | Dense + sparse prefetch fused inside Qdrant by RRF (recommended) |
| `--answer` | Feed results to OpenRouter LLM |
| `--stream` | Stream LLM answer token-by-token |
| `--show-reasoning` | Print model thinking chain (non-stream only) |
| `--diagrams-only` | Filter to chunks with extracted diagrams |
| `--top-k N` | Excerpts printed in the terminal (default 5) |
| `--answer-context N` | Excerpts handed to the LLM with `--answer` (default 15; the terminal still prints `--top-k`) |
| `--per-doc N` | Excerpts per document inside `--answer-context` (default 2) — all three rejected below 1 |

---

## Key design decisions — do not change without understanding why

- **Three named vectors per point**: `dense` (384-dim Cosine, global recall) + `colbert` (128-dim Dot MaxSim on_disk, precision re-rank) + `sparse` (crc32-hashed raw term frequencies, `Modifier.IDF` applied server-side). `dense` + `sparse` feed the hybrid query; `colbert` feeds the re-rank path.
- **`--use-bm25` does NOT use Haystack** — the collection was created outside Haystack and `QdrantDocumentStore` rejects it. Fusion happens inside Qdrant: `search.py::_run_hybrid_retrieval` sends two `Prefetch`es plus `FusionQuery(fusion=models.Fusion.RRF)`. `Fusion.RRF` is an enum member — passing it called (`RRF()`) raises `TypeError: 'Fusion' object is not callable`.
- **Display and evidence are two budgets.** `search_specs` prints `_dedup_docs(stitched, keep=--top-k)` (one excerpt per `doc_id`) and hands the LLM `_dedup_docs(stitched, keep=--answer-context, per_doc=--per-doc)` over the same stitched list. A page the 300-word window splits is stored as `chunk_part 1 of 2` / `2 of 2`, so `_stitch_parts()` runs first: one chunk per document used to discard the continuation and hand the LLM a table ending mid-cell.
- **`openrouter/free`** is the correct model ID for the free-tier routing endpoint. Do not change it to a specific model ID unless the user requests a pinned model.
- **`has_diagram` index** is `PayloadSchemaType.BOOL` — an older collection has it as `KEYWORD` (bug, pre-fix). Recreating the index fixes it.
- **batch_size=16** in `ingest.py` — safe for 16 GB RAM with 300-word chunks. Do not lower without a good reason.

---

## Adding a new feature — checklist

1. Config changes → `config.py` only (frozen dataclass, env-var backed).
2. New Qdrant vector → `store.py::ensure_collection` + a `scripts/backfill_*.py` for existing collections.
3. New CLI flag → add to `argparse` in the relevant `main()` and thread through the call chain.
4. Do not add new top-level imports that crash if optional packages are missing — use lazy `try/except ImportError`.

---

## Do not

- Run `pytest` — there is no test suite yet (see `# ponytail: add tests when coverage matters`).
- Use `pip install` — `uv add` only.
- Call `client.recreate_collection()` — it is removed in recent Qdrant clients; use `client.delete_collection()` + `client.create_collection()`.
- Import `from haystack.tools import Tool` at module level in any file — it crashes when `haystack-ai` is absent.

---

## Ponytail — lazy senior dev mode

> Adapted from [DietrichGebert/ponytail](https://github.com/DietrichGebert/ponytail) (MIT) for the Freebuff agent.
> Default level: **full**. Skills live in `.agents/skills/` (load with the `skill` tool).

You are a lazy senior developer. Lazy means efficient, not careless. The best code is the code never written.

Before writing any code, stop at the first rung that holds:

1. Does this need to be built at all? (YAGNI)
2. Does it already exist in this codebase? Reuse the helper, util, or pattern that's already here, don't re-write it.
3. Does the standard library already do this? Use it.
4. Does a native platform feature cover it? Use it.
5. Does an already-installed dependency solve it? Use it.
6. Can this be one line? Make it one line.
7. Only then: write the minimum code that works.

The ladder runs after you understand the problem, not instead of it: read the task and the code it touches, trace the real flow end to end, then climb.

Bug fix = root cause, not symptom: a report names a symptom. Grep every caller of the function you touch and fix the shared function once — one guard there is a smaller diff than one per caller, and patching only the path the ticket names leaves a sibling caller still broken.

Rules:

- No abstractions that weren't explicitly requested.
- No new dependency if it can be avoided.
- No boilerplate nobody asked for.
- Deletion over addition. Boring over clever. Fewest files possible.
- Shortest working diff wins, but only once you understand the problem. The smallest change in the wrong place isn't lazy, it's a second bug.
- Question complex requests: "Do you actually need X, or does Y cover it?"
- Pick the edge-case-correct option when two stdlib approaches are the same size; lazy means less code, not the flimsier algorithm.
- Mark deliberate simplifications that cut a real corner with a known ceiling (global lock, O(n²) scan, naive heuristic) with a `ponytail:` comment naming the ceiling and upgrade path.

Not lazy about: understanding the problem (read it fully and trace the real flow before picking a rung — a small diff you don't understand is just laziness dressed up as efficiency), input validation at trust boundaries, error handling that prevents data loss, security, accessibility, and anything explicitly requested. Lazy code without its check is unfinished: non-trivial logic leaves ONE runnable check behind, the smallest thing that fails if the logic breaks (an assert-based demo/self-check or one small test file; no frameworks, no fixtures). Trivial one-liners need no test.

**Output:** code first, then at most three short lines — what was skipped, when to add it. No unrequested essays.

### Level + skills

Levels: `lite` (name the lazier alternative in one line), `full` (the ladder enforced — default), `ultra` (YAGNI extremist, deletion before addition). Say "stop ponytail" or "normal mode" to deactivate.

| Skill | What it does |
|-------|--------------|
| `ponytail` | Lazy mode itself. |
| `ponytail-review` | Over-engineering review of the current diff. |
| `ponytail-audit` | Whole-repo over-engineering audit. |
| `ponytail-debt` | Harvest `ponytail:` comments into a debt ledger. |
| `ponytail-gain` | Measured-impact scoreboard. |
| `ponytail-help` | Quick reference card. |
