# Qdrant Collection Schema — `etsi_mec_specs`

Created by `store.py::ensure_collection()`. Collection name = `QDRANT_INDEX` (default `etsi_mec_specs`,
which is an **alias** to `etsi_mec_specs_v2`; `settings.qdrant_index` is the only source, there is no
`--index` CLI flag). Connection: `QDRANT_HOST`/`QDRANT_PORT`, or `QDRANT_PATH` for local file mode.

## Named vectors (three; each retrieval path uses two of them)

| Vector | Dim | Distance | Config | Model (FastEmbed) |
|--------|-----|----------|--------|-------------------|
| `dense` | 384 | COSINE | default | `BAAI/bge-small-en-v1.5` (env `DENSE_MODEL`) |
| `colbert` | 128 | DOT | `MultiVectorConfig(comparator=MAX_SIM)`, `on_disk=True` (RAM saver) | `colbert-ir/colbertv2.0` (env `COLBERT_MODEL`) |
| `sparse` | n/a | DOT + server-side `Modifier.IDF` | `SparseVectorParams(modifier=IDF)` | none — raw TF over crc32-hashed token indices (`search.py::_token_sparse`) |

`--use-bm25` fuses `dense` + `sparse` with RRF inside Qdrant; the default path prefetches `dense` and
rescores with `colbert`. Both models are `max_position_embeddings: 512`, which is why chunks stay small.
`colbert` is a **multi-vector**: upserts pass a list of per-token 128-d vectors, not one vector.

## Payload fields (written in `ingest.py::_process_pdf`)

| Field | Type | Content / how it's produced |
|-------|------|------------------------------|
| `text` | str | The stored chunk body — page markdown, row/clause-packed by `chunking.chunk_page`. A table row is never cut and each table chunk repeats the header row; sizes are bounded by `estimate_tokens` (pipes and `<br>` count) at 420, with `max_words=300` for prose. **What is embedded is not what is stored**: `texts_to_embed` is prefixed with `spec_id edition clause heading`, `metadata['text']` stays the bare body, and query text is never expanded (asymmetric on purpose). |
| `spec_id` | str | From the PDF **cover** title line (`ETSI GS MEC 003 V4.1.1 (2025-05)`) via `identity.stamp()`, e.g. `MEC-003`; an unnumbered document gets `min(filename stems)` instead of null. The authoritative document key. |
| `edition` | str | Cover version, e.g. `V4.1.1`; `None` for unnumbered docs. |
| `pub_date` | str | Cover date `(2025-05)` → `2025-05`; `None` when absent. |
| `content_md5` | str | md5 of the PDF bytes. Two points with the same `content_md5` are the same document under different filenames. |
| `is_current` | bool | True for the newest edition of each `spec_id`. `identity.stamp()` starts everything True — only `scripts/backfill_spec_identity.py --apply`, which sees the whole corpus, sets the superseded ones False. |
| `doc_id` | str | Filename stem (e.g. `MEC003`). A filename, and filenames lie here: 59 of 106 stems hold another spec's content. Never key new logic on it alone. |
| `filename` | str | `pdf_path.name`, e.g. `MEC003.pdf` — used to show the user what the PDF was *stored as*. |
| `page` | int | 1-based page number (pymupdf4llm `page_number`, fallback `page + 1`). |
| `chunk_part` / `total_parts` | int | 1-based sub-chunk index within the page / sub-chunk count. `dedup._stitch_parts` rejoins contiguous parts before ranking, so a table never reaches the LLM ending mid-cell. |
| `heading` | str | First markdown heading in the chunk (`extract_primary_heading`), may be `""`. |
| `clause` | str | Clause the chunk belongs to (`7.2.1`); for a mixed chunk it is the **first** table clause, so a chunk labelled 7.2.2 holds only 7.2.2 rows. `""` when nothing matched. |
| `block_kind` | str | `table` or `prose`. |
| `has_diagram` | bool | True iff markdown image links (or page-level vector renders, attached at `chunk_part == 1`) present. |
| `diagram_paths` | list[str] | POSIX paths (`Path.as_posix`) so a Linux consumer resolves the same file. |

Point ID: `str(uuid.uuid4())`.

## Payload indexes

| Field | Type | Created by |
|-------|------|------------|
| `has_diagram` | BOOL | `store.py::ensure_collection` (an older collection had KEYWORD — booleans never matched; recreating the index fixes it) |
| `is_current` | BOOL | `scripts/backfill_spec_identity.py --apply` |
| `spec_id` | KEYWORD | same |
| `content_md5` | KEYWORD | same |

`edition`, `pub_date`, `clause`, `block_kind`, `heading` and `page` are payload-only — readable in
results, not filterable without a client-side scan. `create_payload_index` takes `field_schema=`, not
`schema=`.

## How fields are consumed (`search.py`, `tools/local_search.py`, `scripts/eval_rag.py`)

- `text` → result content and LLM context, and the source of the `sparse` vector: `search.py::_token_sparse`
  builds it at ingest *and* for the query, so both sides must keep the same tokenizer.
- `build_filter(current_only=True)` adds `must_not` on `is_current=false`; `--all-editions` turns it off.
  Unstamped points keep `is_current` absent/True, so a partially-backfilled collection still returns them.
- `_dedup_docs` spends `--per-doc` per `content_md5` (falling back to `doc_id`, then `text[:40]`), and
  prints/labels hits as `MEC-003 V4.1.1 (stored as MEC041.pdf)`.
- `clause` + `block_kind` → chunk labelling and the `--show-chunks` preview; `page`/`heading` → citation lines.
- `has_diagram` → `--diagrams-only`; `diagram_paths` → PNGs under `data/diagrams/`.
- `eval_rag.py` measures the corpus and all three paths **behind `build_filter()`**, and its golden docs
  are spec ids, not filenames.

## CLI / env relevant to the collection

- `QDRANT_INDEX=etsi_mec_prototype` — the only way to point a run at another collection (ingest, backfill
  and eval must share one shell so they agree).
- `--skip-existing` — per-doc scroll on `doc_id`; ingest additionally refuses to upsert a PDF whose
  `content_md5` already landed in the same run (106 files are 54 documents).
- `--recreate-index` — wipes the target collection. `--dry-run` previews parse + chunking, writes
  nothing, and with `--show-chunks N` prints candidate chunks; with `--recreate-index` it prints the
  wipe it would perform and treats `--skip-existing` as a no-op.

## Gotchas

- `ensure_collection(recreate=False)` is idempotent. Never call `client.recreate_collection()` (removed).
- Oversized chunks are skipped with `[SKIPPED]` per chunk; a giant single table row is kept whole rather
  than sliced, because cutting a row is cutting a cell.
- A crash mid-run leaves a partial but valid collection; a re-run with `--skip-existing` is not
  point-id-aware, so dedupe with `scripts/deduplicate_collection.py` if needed.
- **Qdrant 1.19 cannot add a named vector to an existing collection** — `update_collection` with
  `sparse_vectors_config` (or a new dense name) returns 400 `Not existing vector name error`. Schema
  changes go through `scripts/migrate_add_sparse.py`: scroll old points *with* vectors, upsert into
  `<index>_v2`, verify the count, delete the old, repoint via `POST /collections/aliases` with
  `{"actions":[{"create_alias":{"collection_name":...,"alias_name":...}}]}` — the field is `alias_name`,
  `alias` fails with "did not match any variant of untagged enum AliasOperations", and `POST /aliases`
  is gone. On the Python client there is no `client.create_alias`; use
  `update_collection_aliases(change_aliases_operations=[CreateAliasOperation(CreateAlias(...))])`.
- Filter-scoped `set_payload` re-reads the collection and times out at ~4.7k points — scroll the point
  ids and update in batches of 400.
- Scrolling many points *with vectors* at `limit=64` hits a numpy output-buffer error on this host; use
  small limits or `with_vector: false`.
- The snapshot API returns 200 with an empty body here; backups are volume tars into `backups/`, and
  they matter — the container was once deleted by another agent.
- A freshly copied collection reports `yellow` while it optimizes; queries work regardless.
