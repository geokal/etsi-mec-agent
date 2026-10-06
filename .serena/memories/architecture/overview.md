# ETSI MEC Agent — Architecture

Local RAG pipeline over ETSI GS MEC PDFs:
ingest → tri-vector Qdrant (dense + ColBERT + sparse) → hybrid search (dense+sparse prefetch, RRF
fusion inside Qdrant) → OpenRouter LLM answer.

## Pipeline & data flow

1. **Ingest** (`src/etsi_mec_agent/ingest.py`): PDFs from `data/specs/` → PyMuPDF4LLM chunks (~300 words) → FastEmbed → upsert to Qdrant. Also extracts diagram PNGs to `data/diagrams/` (`MECxxx.pdf-PPPP-NN.png`).
2. **Store** (`src/etsi_mec_agent/store.py`): single Qdrant collection `etsi_mec_specs` (an **alias** to
   `etsi_mec_specs_v2` since the sparse migration), **three named vectors per point**:
   - `dense` — 384-dim, Cosine, BAAI/bge-small-en-v1.5 (global recall)
   - `colbert` — 128-dim, Dot, colbertv2.0, `on_disk` (MaxSim precision re-rank)
   - `sparse` — raw term frequencies over crc32-hashed token indices; `Modifier.IDF` is declared on
     the vector so Qdrant weights rarity corpus-wide
   - Payload index `has_diagram` is `PayloadSchemaType.BOOL` (an older collection had it as KEYWORD —
     bug, pre-fix; recreating the index fixes it).
3. **Search** (`src/etsi_mec_agent/search.py`): two retrieval paths — `dense` prefetch rescored by
   `colbert` MaxSim, and `--use-bm25` which sends a `dense` + `sparse` prefetch pair and lets Qdrant
   fuse them with `FusionQuery(fusion=models.Fusion.RRF)` (pass the enum member; `RRF()` raises
   `TypeError`). The old hand-rolled stdlib BM25 re-ranker in `_run_hybrid_retrieval` is **deleted**.
   **`--use-bm25` still does not use Haystack** — the collection was created outside Haystack and
   `QdrantDocumentStore` rejects it.
4. **Answer**: OpenRouter, model ID `openrouter/free` (free-tier routing endpoint — do not pin a specific model unless the user asks).

## Layout

```
src/etsi_mec_agent/
  config.py    frozen Settings dataclass — Qdrant connection (QDRANT_HOST=localhost,
               QDRANT_PORT=6333, QDRANT_INDEX=etsi_mec_specs, QDRANT_PATH), DENSE_MODEL,
               COLBERT_MODEL, DIAGRAMS_DIR. API keys are NOT here: OPENROUTER_API_KEY is read
               by os.getenv in search.py::generate_answer, EXA_API_KEY inside tools/exa_search.py.
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
  eval_rag.py                golden-question recall@k for both retrieval paths
  migrate_add_sparse.py      copy points into a collection that has 'sparse' (no re-embed)
  audit_specs.py             manifest key vs URL/PDF content identity + coverage gaps
  check_diagram_filter.py    asserts the blank/tiny/duplicate image gate
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
- **Serena MCP** enabled for semantic code navigation (`.agents/mcp.json`, `--context agent --project-from-cwd`); project registered as `etsi-mec-agent`, language server `python`. Memories: `architecture/overview.md` (this file) and `architecture/qdrant-schema.md`.
- Package metadata: `pyproject.toml` + `uv.lock`; `.python-version` pins the interpreter.

### Recent additions (2026-09-26)

- `ingest --dry-run` — parse + chunk preview, no embeddings, no upsert, no `ensure_collection`. Honors `--skip-existing` (per-doc scroll check like a real run) and previews `--recreate-index` wipes (`[DRY-RUN] A real run would DELETE ...`); with `--recreate-index`, `skip_existing` is normalized to False (wipe empties the collection, nothing to skip).
- Shadowing duplicate definition of `ingest_pdfs` in `ingest.py` removed (dead code, found via Serena reference analysis).

### Recent additions (2026-10-06)

- **Diagram capture**: `ingest.py::render_vector_figures` renders vector drawings (ETSI's box-and-arrow
  figures are not embedded rasters) at 200 dpi via `page.cluster_drawings()`, merges stacked raster
  tiles into whole figures, and `_image_is_worth_keeping` drops blank / sub-200px / duplicate-MD5
  images plus their markdown refs. Cross-document dedup is global, so figures reprinted in overview
  decks are dropped from the later document (known limitation; hash→path map pending).
- **Sparse vector + server-side hybrid**: see Store/Search above. `scripts/migrate_add_sparse.py`
  exists because Qdrant 1.19 cannot add a named vector to an existing collection.
- **Downloader identity rule**: `tools/monitor_deliver.py` files each PDF under the spec ID its URL
  declares, not the number it was asked for, and requires a `%PDF` body. `scripts/audit_specs.py`
  reports the pre-existing damage: 106 files are 54 documents, 33 spec numbers, 29 in 001–062 never
  fetched, and ~20% of live points carry a `doc_id` belonging to another document.
- **Agreed next work**: `docs/superpowers/specs/2026-10-06-corpus-wide-answering-design.md` —
  design A (query-time part stitching + per-document excerpts, `--answer-context`) then design B
  (content-derived identity, clause/row-group chunks, `is_current` edition filter).