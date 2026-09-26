# ETSI MEC Agent — Architecture

Local RAG pipeline over ETSI GS MEC PDFs:
ingest → dual-vector Qdrant (dense + ColBERT) → hybrid search (BM25 re-rank + RRF) → OpenRouter LLM answer.

## Pipeline & data flow

1. **Ingest** (`src/etsi_mec_agent/ingest.py`): PDFs from `data/specs/` → PyMuPDF4LLM chunks (~300 words) → FastEmbed → upsert to Qdrant. Also extracts diagram PNGs to `data/diagrams/` (`MECxxx.pdf-PPPP-NN.png`).
2. **Store** (`src/etsi_mec_agent/store.py`): single Qdrant collection `etsi_mec_specs`, **two named vectors per point**:
   - `dense` — 384-dim, Cosine, BAAI/bge-small-en-v1.5 (global recall)
   - `colbert` — 128-dim, Dot, colbertv2.0, `on_disk` (MaxSim precision re-rank)
   - Payload index `has_diagram` is `PayloadSchemaType.BOOL` (an older collection had it as KEYWORD — bug, pre-fix; recreating the index fixes it).
3. **Search** (`src/etsi_mec_agent/search.py`): dense prefetch → optional BM25 re-rank → RRF fusion → optional LLM answer. **`--use-bm25` does NOT use Haystack** — the collection was created outside Haystack and `QdrantDocumentStore` rejects it; BM25 re-ranking is hand-rolled with stdlib `math` + `re` in `_run_hybrid_retrieval`.
4. **Answer**: OpenRouter, model ID `openrouter/free` (free-tier routing endpoint — do not pin a specific model unless the user asks).

## Layout

```
src/etsi_mec_agent/
  config.py    frozen Settings dataclass — ALL env vars live here (OPENROUTER_API_KEY,
               EXA_API_KEY, QDRANT_HOST=localhost, QDRANT_PORT=6333)
  store.py     Qdrant client + ensure_collection()
  ingest.py    CLI: uv run python -m etsi_mec_agent.ingest [--skip-existing|--recreate-index]
  search.py    CLI: uv run python -m etsi_mec_agent.search "query" [--use-bm25] [--answer]
               [--stream] [--show-reasoning] [--diagrams-only] [--top-k N]
  agent.py     monitor entrypoint
  tools/       clip_embed.py (CLIP image embedder, CPU-safe lazy-loads),
               exa_search.py (direct HTTP, no SDK), monitor_deliver.py,
               spec_sync.py, etsi_forge.py, local_search.py
scripts/
  backfill_clip.py           add 'clip' vectors to existing collection (no re-ingest)
  deduplicate_collection.py  remove duplicate chunks in-place (no re-embed)
  graph_visualize.py         force-directed spec graph → etsi_mec_graph.png
data/
  specs/     downloaded PDFs (~120 MB, gitignored)
  diagrams/  extracted PNGs (~190 MB, gitignored)
```

## Hard invariants (break these and things break subtly)

- **uv only** — `uv add` / `uv run` / `uv sync`. Never `pip`. Never touch `.venv/`.
- **batch_size=16** in `ingest.py` — sized for 16 GB RAM with 300-word chunks. Don't lower without reason.
- No top-level imports that crash on optional deps — use lazy `try/except ImportError`. Especially never import `from haystack.tools import Tool` at module level.
- Never call `client.recreate_collection()` (removed in recent Qdrant clients) — use `delete_collection()` + `create_collection()`.
- Config changes go in `config.py` only; new Qdrant vectors need `store.py::ensure_collection` + a `scripts/backfill_*.py`; new CLI flags go in the relevant `main()` argparse and thread through the call chain.
- Qdrant must run in Docker (`qdrant/qdrant`, ports 6333/6334, volume `./qdrant_data`) before ingest or search.

## Tooling state

- **No test suite** (`# ponytail: add tests when coverage matters`) — do not run pytest.
- **Ponytail** (lazy-senior-dev mode, full level) always-on via `AGENTS.md`; six skills in `.agents/skills/`.
- **Serena MCP** enabled for semantic code navigation (`.agents/mcp.json`, `--context agent --project-from-cwd`); project registered as `etsi-mec-agent`, language server `python`.
- Package metadata: `pyproject.toml` + `uv.lock`; `.python-version` pins the interpreter.