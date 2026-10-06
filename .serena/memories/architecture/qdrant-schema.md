# Qdrant Collection Schema — `etsi_mec_specs`

Created by `store.py::ensure_collection()`. Collection name from `QDRANT_INDEX` (default `etsi_mec_specs`). Connection: `QDRANT_HOST`/`QDRANT_PORT`, or `QDRANT_PATH` for local file mode (default empty → server mode).

## Named vectors (three; each retrieval path uses two of them)

| Vector | Dim | Distance | Config | Model (FastEmbed) |
|--------|-----|----------|--------|-------------------|
| `dense` | 384 | COSINE | default | `BAAI/bge-small-en-v1.5` (env `DENSE_MODEL`) |
| `colbert` | 128 | DOT | `MultiVectorConfig(comparator=MAX_SIM)`, `on_disk=True` (RAM saver) | `colbert-ir/colbertv2.0` (env `COLBERT_MODEL`) |
| `sparse` | n/a | DOT + server-side `Modifier.IDF` | `SparseVectorParams(modifier=IDF)` | none — raw TF over crc32-hashed token indices (`search.py::_token_sparse`) |

Each retrieval path uses two of the three: `--use-bm25` fuses `dense` + `sparse` with RRF inside
Qdrant; the default path prefetches `dense` and rescores with `colbert`. Both models are
`max_position_embeddings: 512`, which is why chunks stay small.

`colbert` is a **multi-vector**: upserts pass a list of per-token 128-d vectors (`passage_embed(...).tolist()`), not one vector. `dense` is a single 384-d vector per point.

## Payload fields (written in `ingest.py::_process_pdf`)

| Field | Type | Content / how it's produced |
|-------|------|------------------------------|
| `text` | str | Chunk text, ≤300 words, 40-word overlap (`chunk_page_text`). Cap exists because both embedders are 512-token models and long pages blow ONNX memory. `chunk_page_text` slices by **word count**, so a boundary can land inside a table row — the rest of that row lives in the next `chunk_part` of the same page. |
| `doc_id` | str | Doc name derived from the PDF **filename** (e.g. `MEC003`). Unreliable for anything ingested before 2026-10-06: ~20% of points carry content belonging to a different document, because the downloader saved lookups under the requested number |
| `filename` | str | `pdf_path.name`, e.g. `MEC003.pdf` |
| `page` | int | 1-based page number; from pymupdf4llm `page_number`, fallback `page + 1` |
| `chunk_part` | int | 1-based sub-chunk index within the page |
| `total_parts` | int | Sub-chunk count for that page |
| `heading` | str | First markdown heading in the chunk (`extract_primary_heading`), may be `""` |
| `has_diagram` | bool | True iff markdown image links present in the chunk text |
| `diagram_paths` | list[str] | Paths from markdown image syntax `![...](...)` |

Point ID: `str(uuid.uuid4())`.

## Payload index

- **`has_diagram` → `PayloadSchemaType.BOOL`** — the only indexed payload field (used by `--diagrams-only`, `search.py` filter `must=[FieldCondition(key="has_diagram", match=True)]`).
- An older collection had it as `KEYWORD` (booleans never matched — bug, pre-fix). Fix is recreating the index/collection; `ensure_collection` creates it correctly on new collections.

## How fields are consumed (`search.py`, `tools/local_search.py`)

- `text` → result content and LLM context, and the source of the `sparse` vector — `search.py::_token_sparse` builds it at ingest *and* for the query, so the two sides must keep using the same tokenizer
- `filename`/`doc_id` + `page` + `heading` → citation lines, e.g. `(MEC003 p.18)`; `doc_id` is the fallback display if `filename` missing
- `has_diagram` → `--diagrams-only` filter
- `diagram_paths` → resolved to PNGs in `data/diagrams/` and shown for diagram hits

## CLI ingest flags relevant to the collection

- `--skip-existing` — per-doc scroll on `doc_id` before ingesting a PDF.
- `--dry-run` — full parse + chunk preview, writes nothing (skips embedder load and `ensure_collection`). Honors `--skip-existing`; with `--recreate-index` it previews the wipe (`A real run would DELETE ... (N points)`) and, like real runs, treats `--skip-existing` as no-op since a wiped collection has nothing to skip.

## Gotchas

- `ensure_collection(recreate=False)` is idempotent — creates only when missing. `ingest --recreate-index` wipes the collection (never call `client.recreate_collection()`, removed in recent clients).
- Embedding failures are handled per-chunk with an individual fallback; oversized chunks are skipped with `[SKIPPED]` (page number logged in payload `page`).
- `ingest` streams: embed → upsert per batch of 16; a crash mid-run leaves a partial but valid collection (re-run with `--skip-existing` is not point-id-aware — dedupe via `scripts/deduplicate_collection.py` if needed).
- **Qdrant 1.19 cannot add a named vector to an existing collection** — `update_collection` with
  `sparse_vectors_config` (or a new dense name) returns 400. Schema changes therefore go through
  `scripts/migrate_add_sparse.py`: copy points + vectors + payload into `<index>_v2`, verify the count,
  delete the old, repoint the alias via `POST /collections/aliases`
  (`{"actions":[{"create_alias": ...}]}`; the older `POST /aliases` is gone).
- Scroll pages above ~8 vectors hit client buffer errors on this host — page small, or request
  payload only (`with_vector: false`) for corpus scans.
- Collection status may read `yellow` for a while after a copy migration; queries work regardless.
