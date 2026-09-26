# etsi-mec-agent

An intelligent indexing and RAG agent for ETSI MEC (Multi-access Edge Computing) specifications using **FastEmbed** (local ONNX embeddings), **Qdrant** in Docker, and optional **Haystack 2.x** hybrid retrieval.

---

## Architecture Overview

```
 [ETSI MEC PDFs] (e.g., GS MEC 003, MEC 011)
         │
         ▼
 [PyMuPDF4LLM] (Layout-aware Markdown extraction, table & diagram detection)
         │
         ▼
 [chunk_page_text()] (300-word sliding window, 40-word overlap)
         │
         ├──── Dense embedding (BAAI/bge-small-en-v1.5, 384-dim)
         │                │
         └──── ColBERT embedding (colbert-ir/colbertv2.0, N×128-dim)
                          │
                          ▼
              [QdrantDocumentStore] (Docker: Port 6333)
              ┌──────────────────────────┐
              │  named vector "dense"    │  384-dim Cosine
              │  named vector "colbert"  │  128-dim Dot MaxSim (on_disk)
              │  payload: text, heading, │
              │    page, has_diagram,    │
              │    diagram_paths …       │
              └──────────────────────────┘
```

---

## How the Two Vectors Work Together

Both `dense` and `colbert` encode **text only**. Diagrams are stored as file paths
in the payload — not as vectors.

```
Query: "Explain the role of Mp1 in MEC"
             │
             ▼
    ┌─────────────────┐
    │  dense embed    │  → single 384-dim query vector
    └────────┬────────┘
             │  PREFETCH: fast cosine scan over all 10k+ chunks
             │  → top 25 candidates
             ▼
    ┌─────────────────┐
    │  colbert embed  │  → one 128-dim vector per query token
    └────────┬────────┘
             │  RESCORE (MaxSim): for each query token, find
             │    the best-matching chunk token → precise ranking
             ▼
         Top 5 results ✅
```

| | `dense` | `colbert` | diagrams |
|--|---------|-----------|---------|
| **What it encodes** | Whole chunk text | Token-by-token text | Not encoded (metadata only) |
| **Vectors per chunk** | 1 × 384-dim | N × 128-dim (one per token) | 0 |
| **Role in search** | Fast candidate retrieval | Precise re-ranking | Payload filter / display |
| **Speed** | Very fast | Slower (MaxSim) | N/A |
| **Distance metric** | Cosine | Dot product (MaxSim) | N/A |

> **Why two vectors?**
> `dense` is fast but works at the chunk level ("roughly about this topic").
> `colbert` re-ranks at the token level ("do the specific words match in context?").
> Together they give you speed **and** precision.

---

## Components

- **Document Parser**: **PyMuPDF4LLM** extracts layout-aware Markdown, tables, and diagram image files.
- **Chunking**: `chunk_page_text()` uses a 300-word sliding window (40-word overlap) to stay within ColBERT's 512-token context limit.
- **Embedding Engine**: [FastEmbed](https://github.com/qdrant/fastembed) runs locally via ONNX Runtime (zero API costs, runs on CPU).
- **Vector Database**: [Qdrant](https://qdrant.tech/) running via official Docker container.
- **PDF Monitor**: `tools/monitor_deliver.py` polls Exa for new ETSI PDF releases and downloads them automatically.
- **Exa Search**: `tools/exa_search.py` calls the Exa API directly (no third-party SDK) to locate official ETSI deliver PDFs.
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

### 5. Query / Search Specifications
```bash
uv run python -m etsi_mec_agent.search "What is the Mp1 reference point?"
```

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

### 5️⃣ Hybrid retrieval (BM25 + Vector, via Haystack)
```powershell
uv add haystack-ai qdrant-haystack sentence-transformers
uv run python -m etsi_mec_agent.search "MEC004 section 5.1.2" --use-bm25
```

### 6️⃣ LLM answer generation (OpenRouter free model)
```powershell
$env:OPENROUTER_API_KEY = "<YOUR_KEY>"
uv run python -m etsi_mec_agent.search "Explain the role of Mp1 in MEC" --answer
```

#### Streaming answer (token‑by‑token)
```powershell
uv run python -m etsi_mec_agent.search "Summarize security requirements in MEC002" --answer --stream
```

### 7️⃣ Semantic relationship graph (all specs)
```powershell
uv add networkx matplotlib numpy
uv run python scripts/graph_visualize.py --limit 3000 --threshold 0.55
```
Produces `etsi_mec_graph.png` — a force-directed graph where each node is a MEC spec and edge thickness reflects semantic similarity between specs.

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

> **`"using": "dense"` is required** — the collection has two named vectors (`dense` and `colbert`) and Qdrant needs to know which to project.

Click a dot to see its payload (spec name, page, heading, text excerpt, diagram paths).

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
| `DENSE_MODEL` | `BAAI/bge-small-en-v1.5` | FastEmbed dense model |
| `EMBEDDING_DIM` | `384` | Dense vector dimensions |
| `COLBERT_MODEL` | `colbert-ir/colbertv2.0` | FastEmbed ColBERT model |
| `DIAGRAMS_DIR` | `data/diagrams` | Where extracted diagram PNGs are stored |
| `EXA_API_KEY` | *(required for monitor)* | Exa search API key |
| `OPENROUTER_API_KEY` | *(required for --answer)* | OpenRouter LLM key |
