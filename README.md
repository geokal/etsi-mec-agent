# etsi-mec-agent

An intelligent indexing and RAG agent for ETSI MEC (Multi-access Edge Computing) specifications using **FastEmbed** (local ONNX embeddings), **Qdrant** in Docker, and hybrid retrieval — a `dense` + `sparse` prefetch pair fused by RRF **inside Qdrant**, plus a separate `dense` → `colbert` MaxSim re-rank path — with optional LLM answers via OpenRouter.

Haystack is deliberately not used: the collection is created and queried through the
plain `qdrant_client`, and `QdrantDocumentStore` rejects a collection it did not create.

---

## Architecture Overview

```
 [ETSI MEC PDFs] (e.g., GS MEC 003, MEC 011)
         │
         ▼
  [PyMuPDF4LLM] (layout-aware Markdown, tables, embedded raster export)
         │
         ├──── render_vector_figures(): vector drawings → PNG @200 dpi,
         │       stacked raster tiles merged, blank/tiny/duplicate images dropped
         ▼
 [chunk_page_text()] (300-word sliding window, 40-word overlap)
         │
         ├──── Dense embedding (BAAI/bge-small-en-v1.5, 384-dim)
         ├──── ColBERT embedding (colbert-ir/colbertv2.0, N×128-dim)
         └──── Sparse term frequencies (crc32-hashed token indices, raw counts)
                          │
                          ▼
                [Qdrant] (Docker: Port 6333)
              ┌──────────────────────────┐
              │  named vector "dense"    │  384-dim Cosine
              │  named vector "colbert"  │  128-dim Dot MaxSim (on_disk)
              │  named vector "sparse"   │  raw TF + Modifier.IDF (server-side)
              │  payload: text, heading, │
              │    page, has_diagram,    │
              │    diagram_paths …       │
              └──────────────────────────┘
```

---

## How the Three Vectors Work Together

All three vectors encode **text only**. Diagrams are rendered to PNG files and referenced
by path in the payload — no image is embedded as a vector.

```
Query: "Explain the role of Mp1 in MEC"
             │
     ┌───────┴────────────────────┐
     │ PATH A (default)           │ PATH B (--use-bm25)
     ▼                            ▼
┌──────────────────┐   ┌──────────────────────────────┐
│ dense 384-dim    │   │ dense  → prefetch 100        │
│   ↓ prefetch 25  │   │ sparse → prefetch 100        │
│ colbert MaxSim   │   │  (crc32 token → raw TF,      │
│   re-score per   │   │   IDF weighted server-side)  │
│   query token    │   │  ↓ RRF fusion inside Qdrant  │
└────────┬─────────┘   └──────────────┬───────────────┘
         ▼                             ▼
    fetch_k chunks          fetch_k×6 candidates (30 without --answer,
                                                    90 with it, --top-k 5)
        fetch_k = max(--top-k, --answer-context) — 5 or 15 by default
                                    │
                                    ▼   (PATH A takes the same two steps)
                    _stitch_parts(): contiguous parts of one page —
                    "chunk_part 1 of 2" + "2 of 2" come back as one excerpt
                                    │
                                    ▼
             _dedup_docs() twice over that stitched list: text fingerprint,
             then keep=--top-k (one excerpt per doc_id) for the terminal
             print, and with --answer keep=--answer-context,
             per_doc=--per-doc for the prompt
```

| | `dense` | `colbert` | `sparse` | diagrams |
|--|---------|-----------|---------|---------|
| **What it encodes** | Whole chunk text | Token-by-token text | Term frequencies per chunk | Not encoded (metadata only) |
| **Vectors per chunk** | 1 × 384-dim | N × 128-dim (one per token) | 1 sparse vector | 0 |
| **Role in search** | Candidate retrieval (PATH A and B) | Precision re-ranking (PATH A) | Exact identifiers: `Mm3`, `Mp1`, clause numbers | Payload filter / display |
| **Scoring** | Cosine | Dot product (MaxSim) | Dot with corpus-wide IDF (`Modifier.IDF`) | N/A |
| **Speed** | Very fast | Slower (MaxSim) | Very fast | N/A |

> **Why three vectors?**
> `dense` is fast but works at the chunk level ("roughly about this topic").
> `colbert` re-ranks at the token level ("do the specific words match in context?").
> `sparse` handles what neither embedding reliably does: rare ETSI identifiers and
> clause numbers, where a single exact token is the whole signal. IDF is applied by
> Qdrant over the full corpus, so a token that appears in 3 of 4,669 chunks is weighted
> as rare instead of being scored inside whichever candidate set happened to be fetched.

---

## Components

- **Document Parser**: **PyMuPDF4LLM** extracts layout-aware Markdown, tables, and embedded raster images.
- **Diagram Capture**: `render_vector_figures()` renders the vector drawings ETSI specs actually use (box-and-arrow figures are not embedded rasters) at 200 dpi and merges adjacent raster tiles into a single whole figure; `_image_is_worth_keeping()` drops blank, sub-200px, and duplicate images.
- **Chunking**: `chunk_page_text()` uses a 300-word sliding window (40-word overlap) to stay within ColBERT's 512-token context limit.
- **Embedding Engine**: [FastEmbed](https://github.com/qdrant/fastembed) runs locally via ONNX Runtime (zero API costs, runs on CPU).
- **Vector Database**: [Qdrant](https://qdrant.tech/) running via official Docker container.
- **PDF Monitor**: `tools/monitor_deliver.py` polls Exa for new ETSI PDF releases and downloads them automatically.
- **Exa Search**: `tools/exa_search.py` calls the Exa API directly (no third-party SDK) to locate official ETSI deliver PDFs.
- **Rank Shaping**: `src/etsi_mec_agent/dedup.py` — `_stitch_parts()` merges contiguous parts of one page, `_dedup_docs()` applies the text fingerprint and the per-document cap. Stdlib-only, so `scripts/check_dedup_stitch.py` asserts on it without loading the ONNX embedders.
- **Retrieval Eval**: `scripts/eval_rag.py` scores recall@k for both retrieval paths against golden questions, with a corpus evidence pre-check, and reports the rank inside the wider `--answer` evidence budget as an `aggregate` column.
- **Corpus Audit**: `scripts/audit_specs.py` hashes `data/specs/*.pdf`, reads each PDF's own title page, and reports manifest keys that do not name the document they point at, URLs it cannot verify, and spec numbers never fetched.
- **Schema Migration**: `scripts/migrate_add_sparse.py` adds a named vector to an existing collection by copying points, because Qdrant 1.19 cannot add named vectors in place.
- **LLM Answers**: OpenRouter's OpenAI-compatible endpoint, model ID `openrouter/free`, reasoning chain optional.
- **Package Manager**: [uv](https://github.com/astral-sh/uv) on Python 3.12.

---

## Getting Started

### 1. Prerequisites
- [uv](https://docs.astral.sh/uv/) installed.
- [Docker](https://www.docker.com/) running Qdrant.
- An [Exa API key](https://exa.ai) for the PDF monitor.

### 2. Start Qdrant in Docker
```bash
docker run -d -p 6333:6333 -p 6334:6334 \
    -v qdrant_storage:/qdrant/storage:z \
    --name qdrant \
    qdrant/qdrant
```
Verify Qdrant is accessible at `http://localhost:6333/dashboard`.

### 3. Add ETSI MEC PDFs
Drop your ETSI MEC PDF files into the `data/specs/` directory:
- `data/specs/MEC003.pdf` (MEC Architecture)
- `data/specs/MEC011.pdf` (Edge Platform Application Enablement)
- etc.

### 4. Run PDF Ingestion
```bash
uv run python -m etsi_mec_agent.ingest
```

Options:
- To ingest a specific PDF or directory:
  ```bash
  uv run python -m etsi_mec_agent.ingest path/to/my_spec.pdf
  ```
- To wipe and recreate the Qdrant index before ingestion:
  ```bash
  uv run python -m etsi_mec_agent.ingest --recreate-index
  ```
- To skip PDFs whose `doc_id` is already indexed (safe incremental updates):
  ```bash
  uv run python -m etsi_mec_agent.ingest --skip-existing
  ```
- To preview a run without writing anything (no embeddings, no upserts; honors `--skip-existing` and previews `--recreate-index` wipes):
  ```bash
  uv run python -m etsi_mec_agent.ingest --dry-run --skip-existing
  ```

### 5. Query / Search Specifications
```bash
uv run python -m etsi_mec_agent.search "What is the Mp1 reference point?"
```

Other flags: `--use-bm25` (dense + sparse hybrid fused inside Qdrant), `--answer`
(LLM synthesis via OpenRouter), `--stream`, `--show-reasoning`, `--diagrams-only`,
`--top-k N` (excerpts printed in the terminal), `--answer-context N` (excerpts the LLM reads,
default 15), `--per-doc N` (excerpts per document inside that budget, default 2),
`--prefetch N`. `--top-k`, `--answer-context` and `--per-doc` are rejected below 1. Retrieval
alone loads only the local ONNX embedders; no API call leaves the machine unless `--answer` is used.

---

## Usage Guidelines

### 1️⃣ Clean previous data (optional)
```powershell
# Remove old PDFs and manifest
Remove-Item -Recurse -Force "data/specs"
New-Item -ItemType Directory -Path "data/specs"
# Drop existing Qdrant collection (if you want a fresh index)
Remove-Item -Recurse -Force "qdrant_data/collection/etsi_mec_specs"
```

### 2️⃣ Fetch latest PDFs via the monitor
```powershell
$env:EXA_API_KEY = "<your-exa-key>"
uv run python -m etsi_mec_agent.agent --watch-deliver 5   # press Ctrl‑C after a few cycles
```
PDFs are saved as `data/specs/MECxxx.pdf`. The `manifest.json` tracks the last known version URL per spec so only newer versions are re-downloaded.

### 3️⃣ Ingest PDFs into Qdrant
```powershell
uv run python -m etsi_mec_agent.ingest data/specs --recreate-index
```

### 4️⃣ Basic search (dense + ColBERT)
```powershell
uv run python -m etsi_mec_agent.search "What is Mp1?"
```

### 5️⃣ Hybrid retrieval (dense + sparse, fused server-side by RRF)
```powershell
uv run python -m etsi_mec_agent.search "MEC004 section 5.1.2" --use-bm25
```

No extra dependencies needed: the `sparse` vector holds raw term frequencies and the collection declares
`Modifier.IDF`, so rarity is computed by Qdrant over the whole corpus and fused with the dense
prefetch via `FusionQuery(fusion=models.Fusion.RRF)` — pass the enum member, do not call it.
Haystack is intentionally **not** used — the collection was created outside Haystack and
`QdrantDocumentStore` rejects it.

### 6️⃣ LLM answer generation (OpenRouter free model)
```powershell
$env:OPENROUTER_API_KEY = "<YOUR_KEY>"
uv run python -m etsi_mec_agent.search "Explain the role of Mp1 in MEC" --answer
```

#### Streaming answer (token‑by‑token)
```powershell
uv run python -m etsi_mec_agent.search "Summarize security requirements in MEC002" --answer --stream
```

### 7️⃣ Measure retrieval quality (golden-question eval)
```powershell
uv run python scripts/eval_rag.py --top-k 5
```
Prints the rank at which each golden question is answered under both retrieval paths, plus the
`aggregate` column — the same hybrid path shaped like the `--answer` evidence budget (stitched
parts, up to `--per-doc` excerpts per doc, read over `--answer-context`, defaults 2 and 15).
Recall is reported for the two baseline paths at `--top-k` and for the aggregate column over that
budget.
A question flagged `EVIDENCE-MISSING` is bad golden data — its keyword does not occur in the
expected document — and is excluded from the denominator, so recall reflects retrieval only.

### 8️⃣ Add a named vector without re-embedding
Qdrant 1.19 cannot add a named vector to an existing collection, so schema changes in
`store.py::ensure_collection` only apply to a fresh index. For an index you already have:
```powershell
uv run python scripts/migrate_add_sparse.py
```
Copies every point's vectors and payload into `<index>_v2`, verifies the count, deletes the old
collection and repoints the `etsi_mec_specs` alias. No re-embedding.

### 9️⃣ Semantic relationship graph (all specs)
```powershell
uv add networkx matplotlib numpy
uv run python scripts/graph_visualize.py --limit 3000 --threshold 0.55
```
Produces `etsi_mec_graph.png` — a force-directed graph where each node is a MEC spec and edge thickness reflects semantic similarity between specs.

---

## How an answer is assembled (and its current limits)

`generate_answer()` builds one system prompt plus a single context block: every retrieved chunk
gets a `[Source i] filename p.page — heading` header, its text, and any diagram paths listed as
text. One request, one answer.

Worth knowing before trusting an aggregation-style question:

- **Two budgets, not one.** The terminal prints at most `--top-k` excerpts (default 5), one per
  `doc_id` — the rule that stops one spec monopolising the slots. `--answer` reads a separate
  evidence set over the same stitched list: up to `--answer-context` excerpts (default 15), with
  up to `--per-doc` (default 2) from any one document. Parts of a page that the 300-word window
  split are merged before either budget, so a table reaches the prompt whole instead of losing its
  continuation to the one-slot-per-document display rule.
- **Paths, not pixels.** Diagram filenames go into the prompt; no image bytes are sent, so the
  model can cite a figure but cannot read it.
- **One query, one pass.** No sub-question decomposition and no second retrieval round over what
  the first pass missed.
- **Document labels are approximate.** `data/specs` holds byte-identical PDFs saved under several
  spec numbers (106 files, 54 unique) and several editions of the same spec (GS MEC 003
  V2.2.1 / V3.1.1 / V4.1.1) are indexed together, so a citation can name a spec the text does not
  belong to, or quote a superseded edition.

Raising `--top-k` buys more printed excerpts, not better synthesis; `--answer-context` and
`--per-doc` buy evidence. Cross-document aggregation still needs row/clause-aware chunking and a
map-reduce pass — the multi-chunk-per-document allowance this section used to ask for is
`--per-doc` today.

---

## Qdrant Dashboard Visualization

Open `http://localhost:6333/dashboard`, click your collection, then **Visualize** and paste:

```json
{
  "limit": 500,
  "using": "dense",
  "color_by": { "payload": "doc_id" },
  "algorithm": "UMAP",
  "n_neighbors": 15
}
```

> **`"using": "dense"` is required** — the collection has three named vectors (`dense`, `colbert`, `sparse`)
> and Qdrant needs to know which one to project.

Click a dot to see its payload (spec name, page, heading, text excerpt, diagram paths).

---

## Repo Tooling (AI agents)

This repo is wired for AI coding agents:

- `AGENTS.md` — always-on *ponytail* minimal-code ruleset (loaded by Freebuff and compatible agents).
- `.agents/skills/` — the six ponytail skills (`ponytail`, `ponytail-review`, `ponytail-audit`, `ponytail-debt`, `ponytail-gain`, `ponytail-help`).
- `.agents/mcp.json` — Serena MCP server registration for the Freebuff CLI (semantic symbol-level code navigation).
- `.serena/` — Serena project config plus architecture memories (pipeline overview, Qdrant schema).

---

## Configuration

Settings can be customized via `.env` (copy from `.env.example`):
```bash
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_PORT` | `6333` | Qdrant HTTP port |
| `QDRANT_INDEX` | `etsi_mec_specs` | Collection name in Qdrant |
| `QDRANT_PATH` | *(empty)* | Local file path for embedded Qdrant mode (instead of host/port) |
| `DENSE_MODEL` | `BAAI/bge-small-en-v1.5` | FastEmbed dense model |
| `COLBERT_MODEL` | `colbert-ir/colbertv2.0` | FastEmbed ColBERT model |
| `DIAGRAMS_DIR` | `data/diagrams` | Where extracted diagram PNGs are stored |
| `EXA_API_KEY` | *(required for monitor)* | Exa search API key |
| `OPENROUTER_API_KEY` | *(required for --answer)* | Read directly in `search.py`, not in `config.Settings` |
| `FASTEMBED_CACHE_PATH` | OS cache dir | Point at `models` (gitignored) to reuse downloaded ONNX weights instead of re-fetching |
