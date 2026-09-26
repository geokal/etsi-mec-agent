# Qdrant Collection Schema — `etsi_mec_specs`

Created by `store.py::ensure_collection()`. Collection name from `QDRANT_INDEX` (default `etsi_mec_specs`). Connection: `QDRANT_HOST`/`QDRANT_PORT`, or `QDRANT_PATH` for local file mode (default empty → server mode).

## Named vectors (both required — search queries reference both)

| Vector | Dim | Distance | Config | Model (FastEmbed) |
|--------|-----|----------|--------|-------------------|
| `dense` | 384 | COSINE | default | `BAAI/bge-small-en-v1.5` (env `DENSE_MODEL`) |
| `colbert` | 128 | DOT | `MultiVectorConfig(comparator=MAX_SIM)`, `on_disk=True` (RAM saver) | `colbert-ir/colbertv2.0` (env `COLBERT_MODEL`) |

`colbert` is a **multi-vector**: upserts pass a list of per-token 128-d vectors (`passage_embed(...).tolist()`), not one vector. `dense` is a single 384-d vector per point.

## Payload fields (written in `ingest.py::_process_pdf`)

| Field | Type | Content / how it's produced |
|-------|------|------------------------------|
| `text` | str | Chunk text, ≤300 words, 40-word overlap (`chunk_page_text`). Cap exists for ColBERT's 512-token window + ONNX memory blows-ups on long tables/pages. |
| `doc_id` | str | Doc name derived from the PDF (e.g. `MEC003`) |
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

- `text` → BM25 tokenization for re-rank, result content, LLM context
- `filename`/`doc_id` + `page` + `heading` → citation lines, e.g. `(MEC003 p.18)`; `doc_id` is the fallback display if `filename` missing
- `has_diagram` → `--diagrams-only` filter
- `diagram_paths` → resolved to PNGs in `data/diagrams/` and shown for diagram hits

## Gotchas

- `ensure_collection(recreate=False)` is idempotent — creates only when missing. `ingest --recreate-index` wipes the collection (never call `client.recreate_collection()`, removed in recent clients).
- Embedding failures are handled per-chunk with an individual fallback; oversized chunks are skipped with `[SKIPPED]` (page number logged in payload `page`).
- `ingest` streams: embed → upsert per batch of 16; a crash mid-run leaves a partial but valid collection (re-run with `--skip-existing` is not point-id-aware — dedupe via `scripts/deduplicate_collection.py` if needed).
