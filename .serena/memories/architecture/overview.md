# ETSI MEC Agent — Architecture

Local RAG pipeline over ETSI GS MEC PDFs:
ingest → tri-vector Qdrant (dense + ColBERT + sparse) → hybrid search (dense+sparse prefetch, RRF
fusion inside Qdrant) → OpenRouter LLM answer.

## Pipeline & data flow

1. **Ingest** (`src/etsi_mec_agent/ingest.py`): PDFs from `data/specs/` → PyMuPDF4LLM page markdown →
   `chunking.chunk_page` row/clause-aware units → FastEmbed → upsert. `identity.stamp(pdf)` runs per
   document so every point carries `spec_id`/`edition`/`pub_date`/`is_current` from the **cover**, plus
   `content_md5`; byte-identical PDFs are skipped within a run (`[SKIP] ... byte-identical`), so 106
   files become 54 documents. Diagram PNGs (raster + rendered vector figures) go to `data/diagrams/`.
2. **Chunking** (`src/etsi_mec_agent/chunking.py`, stdlib-only): whole markdown units are packed, so a
   table row is never cut and every table chunk repeats its header row and clause. Sizes are bounded by
   `estimate_tokens` (pipes and `<br>` count) at 420 vs the embedders' 512 cap, and by `max_words=300`
   for prose. A clause change inside one table only starts a new chunk once the chunk holds
   `min_words=150` — without that floor a dense sub-clause table shatters into one-chunk-per-row
   confetti (measured: 9854 → ~4800 chunks corpus-wide, p90 parts/page 46 → 2). Prose past the limit
   keeps the old 300-word/40-overlap window because `dedup._stitch` trims exactly that overlap.
3. **Store** (`src/etsi_mec_agent/store.py`): collection `QDRANT_INDEX` (default `etsi_mec_specs`, an
   **alias** to `etsi_mec_specs_v2` since the sparse migration), three named vectors per point:
   - `dense` — 384-dim, Cosine, BAAI/bge-small-en-v1.5 (global recall)
   - `colbert` — 128-dim, Dot, colbertv2.0, `on_disk` (MaxSim precision re-rank)
   - `sparse` — raw term frequencies over crc32-hashed token indices; `Modifier.IDF` declared on the
     vector so Qdrant weights rarity corpus-wide
   Payload indexes created by `ensure_collection`: `has_diagram` BOOL (an older collection had KEYWORD —
   bug, pre-fix). Identity indexes (`is_current` BOOL, `spec_id`/`content_md5` KEYWORD) are created by
   `scripts/backfill_spec_identity.py --apply`, not by `ensure_collection`.
4. **Search** (`src/etsi_mec_agent/search.py`): two retrieval paths — `dense` prefetch rescored by
   `colbert` MaxSim, and `--use-bm25` which sends a `dense` + `sparse` prefetch pair and lets Qdrant
   fuse with `FusionQuery(fusion=models.Fusion.RRF)` (pass the enum member; `RRF()` raises `TypeError`).
   **`--use-bm25` still does not use Haystack** — the collection was created outside Haystack and
   `QdrantDocumentStore` rejects it. `build_filter` excludes `is_current=false` unless `--all-editions`.
   Display and evidence are **two budgets** over the same stitched list: `--top-k` excerpts printed,
   `--answer-context` (default 15) handed to the LLM with `--per-doc` (default 2) per document, where
   the per-document quota is spent per `content_md5` so aliases of one spec share it.
   `dedup.py` (stdlib-only) does `_stitch` / `_stitch_parts` / `_dedup_docs`.
5. **Answer**: OpenRouter, model ID `openrouter/free` (free-tier routing — don't pin a model unless asked).

## Layout

```
src/etsi_mec_agent/
  config.py    frozen Settings dataclass — Qdrant connection (QDRANT_HOST=localhost,
               QDRANT_PORT=6333, QDRANT_INDEX=etsi_mec_specs, QDRANT_PATH), DENSE_MODEL,
               COLBERT_MODEL, DIAGRAMS_DIR. OPENROUTER_API_KEY is read by os.getenv in
               search.py::generate_answer, EXA_API_KEY inside tools/exa_search.py.
  store.py     Qdrant client + ensure_collection()
  chunking.py  page markdown → row/clause-aware PageChunks; stdlib-only so checks run without ONNX
  identity.py  cover_identity()/stamp() — spec_id/edition/pub_date via pymupdf, no ONNX
  dedup.py     stitch + per-doc rank shaping; stdlib-only for the same reason
  ingest.py    CLI: uv run python -m etsi_mec_agent.ingest [--skip-existing|--recreate-index|--dry-run]
  search.py    CLI: uv run python -m etsi_mec_agent.search "query" [--use-bm25] [--answer] [--stream]
               [--show-reasoning] [--diagrams-only] [--top-k N] [--answer-context N] [--per-doc N] [--all-editions]
  agent.py     monitor entrypoint
  tools/       clip_embed.py, exa_search.py, monitor_deliver.py, spec_sync.py, etsi_forge.py, local_search.py
scripts/
  backfill_spec_identity.py  report / --apply / --verify identity from PDF covers (no re-ingest)
  backfill_clip.py           add 'clip' vectors (no re-ingest)
  deduplicate_collection.py  remove duplicate chunks in-place
  eval_rag.py                golden-question recall@k, both paths + the aggregate@N answer budget
  migrate_add_sparse.py      copy points into a collection that has 'sparse' (no re-embed)
  audit_specs.py             manifest key vs URL/PDF content identity + coverage gaps
  check_chunking.py          asserts: rows never cut, headers repeated, clause floor; no models
  check_dedup_stitch.py      asserts: overlap trim, content_md5 alias quota; no models, no Qdrant
  graph_visualize.py         force-directed spec graph → etsi_mec_graph.png
data/
  specs/     downloaded PDFs (~120 MB, gitignored)
  diagrams/  extracted PNGs (~190 MB, gitignored)
```

## Hard invariants (break these and things break subtly)

- **uv only** — `uv add` / `uv run` / `uv sync`. Never `pip`. Never touch `.venv/`.
- **There is no `--index` flag.** Target another collection with the env var:
  `$env:QDRANT_INDEX='etsi_mec_prototype'` for ingest, backfill and eval **in the same shell**, since
  everything reads `config.py::Settings.qdrant_index`.
- **Identity never comes from the filename.** 59 of 106 doc_ids hold another spec's content
  (`MEC041.pdf` is GS MEC 040). New code must read `spec_id`/`content_md5`; `doc_id` is a filename.
- **batch_size=16** in `ingest.py` — sized for 16 GB RAM. Don't lower without reason.
- `dedup.py` and `chunking.py` must stay stdlib-only so `scripts/check_*.py` run with ONNX absent —
  that is how retrieval/chunking changes get verified on this host without loading models.
- No top-level imports that crash on optional deps — lazy `try/except ImportError`; never import
  `from haystack.tools import Tool` at module level.
- Never call `client.recreate_collection()` (removed) — `delete_collection()` + `create_collection()`.
- Config changes go in `config.py` only; new vectors need `ensure_collection` + a `scripts/backfill_*.py`;
  new CLI flags go in the relevant `main()` argparse and thread through the call chain.
- Qdrant must run in Docker (`qdrant/qdrant`, 6333/6334, volume `./qdrant_data`) before ingest or search.
- **After any ingest, run `scripts/backfill_spec_identity.py --apply`** — which edition is newest is
  only knowable once the whole corpus is in; `identity.stamp()` sets `is_current=True` for everything.

## Tooling state

- **No test suite** (`# ponytail: add tests when coverage matters`) — do not run pytest. The runnable
  checks are `scripts/check_chunking.py`, `scripts/check_dedup_stitch.py`, `backfill_spec_identity.py
  --self-check/--verify`.
- **Ponytail** (lazy-senior-dev, full level) always-on via `AGENTS.md`; skills in `.agents/skills/`.
- **Serena MCP** is registered **globally** in `~/.qoder/settings.json` (`mcpServers.serena`,
  `uvx -q -p 3.13 --from git+https://github.com/oraios/serena serena start-mcp-server --context ide`),
  not in a repo `.agents/mcp.json`. A global registration passes no `--project`: every session must
  call `activate_project etsi-mec-agent` first. Under `--context ide`, `create_text_file`/`read_file`/
  `execute_shell_command`/`find_file`/`list_dir` are excluded, so brand-new files go through the IDE's
  own Write tool and all edits go through Serena. Line numbers are 0-based.
- **code-review-graph MCP** is registered the same way; `get_minimal_context_tool` is the cheap entry
  point and reports `head_matches_build`, so check it instead of rebuilding.
- Package metadata: `pyproject.toml` + `uv.lock`; `.python-version` pins the interpreter.

### Recent additions (2026-09-26)

- `ingest --dry-run` — parse + chunk preview, no embeddings, no upsert, no `ensure_collection`; honors
  `--skip-existing`, previews `--recreate-index` wipes, and `--show-chunks N` prints candidate chunks.
- Shadowing duplicate definition of `ingest_pdfs` removed (dead code, found via Serena reference analysis).

### Recent additions (2026-10-06)

- **Diagram capture**: `ingest.py::render_vector_figures` renders ETSI's vector box-and-arrow figures at
  200 dpi via `page.cluster_drawings()`, merges stacked raster tiles, and `_image_is_worth_keeping`
  drops blank / sub-200px / duplicate-MD5 images plus their markdown refs. Cross-document dedup is
  global, so a figure reprinted in an overview deck is dropped from the later document (known
  limitation; hash→path map pending — see `mem:architecture/qdrant-schema`).
- **Sparse vector + server-side hybrid**: `scripts/migrate_add_sparse.py` exists because Qdrant 1.19
  cannot add a named vector to an existing collection.
- **Downloader identity rule**: `tools/monitor_deliver.py` files each PDF under the spec ID its URL
  declares, not the number it was asked for, and requires a `%PDF` body.

### Recent additions (2026-10-08)

- **Spec identity landed**: `identity.py` + `scripts/backfill_spec_identity.py` (report / `--apply` /
  `--verify`), `build_filter` edition filtering, `--all-editions`, and `content_md5` per-doc quota.
  Unnumbered docs (white papers, decks) get `spec_id = min(filename stems)` rather than null.
- **B2 chunking** (`chunking.py`) with the `min_words` clause floor; ingest embeds
  `spec_id edition clause heading + body` while storing the page markdown body (asymmetric on purpose).
- **Measured**: recall@5 colbert/hybrid went 10/10/11 → 11/11/12 with identity (union 14/16), and the
  prototype with row/clause chunks scored 11/10/12 @ctx15 and 11/10/13 @ctx25 *before* the floor fix.
  q08 (User app LCM proxy) and q14 (MEC platform service discovery) are missed by all three paths and
  are provably retrieval failures, not ranking ones. The rerun + cutover (`rename_alias` then
  `create_alias`) is the open work — see `docs/superpowers/specs/2026-10-06-corpus-wide-answering-design.md`.
