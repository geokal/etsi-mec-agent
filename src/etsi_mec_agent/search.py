import argparse
import os
import time

from fastembed import LateInteractionTextEmbedding, TextEmbedding
from qdrant_client import models

from etsi_mec_agent.config import settings
from etsi_mec_agent.store import get_qdrant_client


def get_embedders():
    dense_model = TextEmbedding(settings.dense_model)
    colbert_model = LateInteractionTextEmbedding(settings.colbert_model)
    return dense_model, colbert_model


# ---------------------------------------------------------------------------
# Hybrid retrieval — Dense prefetch + BM25 re-rank (no Haystack required)
# ---------------------------------------------------------------------------

import math
import re as _re


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split into word tokens."""
    return _re.findall(r"[a-z0-9]+", text.lower())


def _bm25_score(query_tokens: list[str], doc_tokens: list[str],
                avg_dl: float, k1: float = 1.5, b: float = 0.75,
                corpus_size: int = 100) -> float:
    """
    Score one document against a tokenised query using BM25.
    IDF is approximated as log((N - df + 0.5) / (df + 0.5) + 1)
    where N = corpus_size and df = 1 (single-doc IDF estimate).
    """
    tf_map: dict[str, int] = {}
    for tok in doc_tokens:
        tf_map[tok] = tf_map.get(tok, 0) + 1

    dl = len(doc_tokens)
    score = 0.0
    for qt in set(query_tokens):
        tf = tf_map.get(qt, 0)
        if tf == 0:
            continue
        idf = math.log((corpus_size - 1 + 0.5) / (1 + 0.5) + 1)
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * dl / max(avg_dl, 1))
        score += idf * (numerator / denominator)
    return score


def _run_hybrid_retrieval(
    client,
    query_text: str,
    query_dense: list,
    top_k: int = 5,
    prefetch_limit: int = 50,
    query_filter=None,
) -> list:
    """
    True hybrid retrieval over our existing Qdrant collection:

    Stage 1 — Dense prefetch:
        Fetch `prefetch_limit` candidates using the pre-computed dense vector.
        This gives broad semantic recall.

    Stage 2 — BM25 re-rank:
        Score each candidate against the query using BM25 on its stored text.
        BM25 rewards exact keyword matches that dense embeddings might miss
        (e.g. "Mm4", "Mp1", specific clause numbers, RFC identifiers).

    Stage 3 — RRF (Reciprocal Rank Fusion):
        Combine the dense rank and BM25 rank into a single score so neither
        signal dominates.  Returns the top_k highest-fused results as simple
        namespace objects with .content and .meta for compatibility with
        generate_answer().
    """
    from qdrant_client import models as _qm

    # ── Stage 1: dense prefetch ────────────────────────────────────────────
    results = client.query_points(
        collection_name=settings.qdrant_index,
        prefetch=_qm.Prefetch(
            query=query_dense,
            using="dense",
            limit=prefetch_limit,
            filter=query_filter,
        ),
        query=query_dense,          # score by dense for initial ordering
        using="dense",
        limit=prefetch_limit,
        query_filter=query_filter,
    )
    hits = results.points
    if not hits:
        return []

    # ── Stage 2: BM25 scoring ──────────────────────────────────────────────
    query_tokens = _tokenize(query_text)
    doc_token_lists = [_tokenize(h.payload.get("text", "")) for h in hits]
    avg_dl = sum(len(tl) for tl in doc_token_lists) / max(len(doc_token_lists), 1)

    bm25_scores = [
        _bm25_score(query_tokens, tl, avg_dl, corpus_size=len(hits))
        for tl in doc_token_lists
    ]

    # ── Stage 3: RRF fusion ────────────────────────────────────────────────
    # Rank by dense score (already ordered) and by BM25 score separately,
    # then fuse with RRF constant k=60.
    k = 60
    dense_ranks  = {h.id: r + 1 for r, h in enumerate(hits)}
    bm25_sorted  = sorted(range(len(hits)), key=lambda i: bm25_scores[i], reverse=True)
    bm25_ranks   = {hits[i].id: r + 1 for r, i in enumerate(bm25_sorted)}

    def rrf(hit_id):
        return 1 / (k + dense_ranks[hit_id]) + 1 / (k + bm25_ranks[hit_id])

    fused = sorted(hits, key=lambda h: rrf(h.id), reverse=True)[:top_k]

    # ── Wrap as simple doc objects for generate_answer() ──────────────────
    class _Doc:
        def __init__(self, text, meta):
            self.content = text
            self.meta = meta

    return [
        _Doc(
            text=h.payload.get("text", ""),
            meta=h.payload,
        )
        for h in fused
    ]



# ---------------------------------------------------------------------------
# Deduplication helper shared by both search paths
# ---------------------------------------------------------------------------

_IMG_RE = _re.compile(r'!\[.*?\]\(.*?\)')   # strip markdown image refs before fingerprinting


def _dedup_docs(docs: list, keep: int) -> list:
    """
    Two-pass deduplication:

    Pass 1 — text fingerprint (first 400 chars, images stripped):
        Catches the same chunk stored under different doc_ids because
        each PDF extraction embeds a different image path in the markdown,
        making a naive first-300-chars fingerprint fail.

    Pass 2 — doc_id uniqueness (keep best chunk per document):
        Prevents the same spec from dominating all result slots.
        Within each doc_id we keep the first (highest-ranked) chunk.

    Returns at most `keep` results.
    """
    # Pass 1 — text fingerprint
    seen_text: set = set()
    after_text: list = []
    for doc in docs:
        raw = doc.content if hasattr(doc, "content") else (doc.meta or {}).get("text", "")
        fp = _IMG_RE.sub("", raw or "")[:400].strip()
        if fp not in seen_text:
            seen_text.add(fp)
            after_text.append(doc)

    # Pass 2 — one chunk per doc_id
    seen_doc: set = set()
    unique: list = []
    for doc in after_text:
        meta = doc.meta if hasattr(doc, "meta") else {}
        doc_id = (meta or {}).get("doc_id", "") or (doc.content or "")[:40]
        if doc_id not in seen_doc:
            seen_doc.add(doc_id)
            unique.append(doc)
        if len(unique) >= keep:
            break

    return unique


# ---------------------------------------------------------------------------
# LLM answer generation (OpenRouter free model, OpenAI-compatible SDK)
# ---------------------------------------------------------------------------

def _make_simple_doc(text: str, meta: dict):
    """Tiny stand-in for a Document object – just a namespace with .content and .meta."""
    class _Doc:
        def __init__(self, c, m):
            self.content = c
            self.meta = m
    return _Doc(text, meta)


def generate_answer(query: str, documents: list, stream: bool = False, show_reasoning: bool = False) -> str:
    """
    Call OpenRouter (openrouter/free) with the retrieved ETSI MEC chunks as context.

    model="openrouter/free"   — confirmed correct per OpenRouter docs
    reasoning.enabled=True    — model returns reasoning_details (thinking chain)
                                 alongside the final .content answer
    stream=True               — streams final answer token-by-token
    show_reasoning=True       — also prints the reasoning_details thinking block
    """
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "openai package is required for --answer. Install with:\n"
            "  uv add openai"
        ) from exc

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    # Build rich context: each chunk gets a header with its source + diagram paths
    context_parts = []
    for i, doc in enumerate(documents, 1):
        if hasattr(doc, "content"):
            text = doc.content or ""
            meta = doc.meta or {}
        elif hasattr(doc, "meta"):
            text = doc.meta.get("text", "")
            meta = doc.meta or {}
        else:
            text = str(doc)
            meta = {}

        src_label = (
            f"[Source {i}] {meta.get('filename', 'unknown')} "
            f"p.{meta.get('page', '?')} — {meta.get('heading', '')}"
        ).strip(" —")
        diagrams = meta.get("diagram_paths", [])
        diagram_note = (
            "\n  📐 Diagram(s) available: " + ", ".join(diagrams)
            if diagrams else ""
        )
        context_parts.append(f"--- {src_label}{diagram_note}\n{text}")

    context_text = "\n\n".join(context_parts)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a technical writer specialising in ETSI MEC (Multi-access Edge Computing) "
                "specifications. When answering questions:\n"
                "• Write a comprehensive, well-structured answer of at least 3–5 paragraphs.\n"
                "• Cite the source document and page for every claim, e.g. (MEC003 p.18).\n"
                "• If a diagram is listed in the context (📐 Diagram(s) available: …), "
                "reference it explicitly by filename in your answer.\n"
                "• Use the ETSI standard terminology (reference points, functional entities, etc.).\n"
                "• Do NOT invent information not present in the provided context excerpts."
            ),
        },
        {
            "role": "user",
            "content": f"Context excerpts from ETSI MEC specifications:\n\n{context_text}\n\nQuestion: {query}",
        },
    ]


    response = client.chat.completions.create(
        model="openrouter/free",
        messages=messages,
        extra_body={"reasoning": {"enabled": True}},
        stream=stream,
    )

    if stream:
        # Streaming: content tokens arrive chunk-by-chunk.
        # reasoning_details are not available in streaming mode — they only
        # come back in a non-streaming response.
        print("\n[LLM ANSWER] ", end="", flush=True)
        full = ""
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                print(delta, end="", flush=True)
                full += delta
        print()
        return full
    else:
        msg = response.choices[0].message

        # Optionally print the model's internal reasoning chain (thinking blocks)
        # This is the "reasoning_details" field documented on openrouter.ai/openrouter/free
        if show_reasoning:
            reasoning = getattr(msg, "reasoning_details", None)
            if reasoning:
                print("\n[REASONING]")
                for block in reasoning:
                    # blocks are typically {"type": "thinking", "thinking": "…"}
                    thinking = (
                        block.get("thinking")
                        if isinstance(block, dict)
                        else getattr(block, "thinking", str(block))
                    )
                    if thinking:
                        print(thinking)
                print("-" * 65)

        print("\n[LLM ANSWER]", msg.content)
        return msg.content



# ---------------------------------------------------------------------------
# Core search function
# ---------------------------------------------------------------------------

def search_specs(
    query_text: str,
    top_k: int = 5,
    prefetch_limit: int = 25,
    diagrams_only: bool = False,
    use_bm25: bool = False,
    generate: bool = False,
    stream: bool = False,
    show_reasoning: bool = False,
):
    """
    Execute Hybrid Search (Dense + ColBERT) with optional BM25+Vector hybrid
    and optional LLM answer generation.

    * Without --use-bm25    : original dense + ColBERT Qdrant query (unchanged).
    * With    --use-bm25    : Haystack 2.x QdrantEmbeddingRetriever (pre-computed vector).
    * With    --answer      : feeds the retrieved chunks to the OpenRouter LLM.
    * With    --show-reasoning : also prints the model's reasoning_details thinking chain.
    """
    client = get_qdrant_client()
    dense_model, colbert_model = get_embedders()

    t0 = time.time()
    query_dense = list(dense_model.embed([query_text]))[0].tolist()
    query_colbert = list(colbert_model.query_embed(query_text))[0].tolist()

    query_filter = None
    if diagrams_only:
        query_filter = models.Filter(
            must=[models.FieldCondition(key="has_diagram", match=models.MatchValue(value=True))]
        )

    # ------------------------------------------------------------------
    # 1️⃣  Hybrid retrieval (Dense prefetch + BM25 re-rank + RRF fusion)
    # ------------------------------------------------------------------
    if use_bm25:
        print(f"\n[QUERY] '{query_text}' [HYBRID: Dense prefetch → BM25 re-rank → RRF]")
        hybrid_docs = _run_hybrid_retrieval(
            client=client,
            query_text=query_text,
            query_dense=query_dense,
            top_k=top_k * 6,          # fetch 6× more so dedup still yields top_k unique docs
            prefetch_limit=max(prefetch_limit * 4, 100),
            query_filter=query_filter,
        )

        unique_docs = _dedup_docs(hybrid_docs, keep=top_k)

        if len(unique_docs) < len(hybrid_docs):
            print(f"[DEDUP] {len(hybrid_docs)} → {len(unique_docs)} unique document(s) (removed {len(hybrid_docs)-len(unique_docs)} duplicates).\n")

        print(f"[HYBRID] Retrieved {len(unique_docs)} unique document(s).\n")
        for i, doc in enumerate(unique_docs, 1):
            meta          = doc.meta or {}
            fname         = meta.get("filename", "<unknown>")
            page          = meta.get("page", "N/A")
            heading       = meta.get("heading", "")
            has_diagram   = meta.get("has_diagram", False)
            diagram_paths = meta.get("diagram_paths", [])

            print(f"=== [Hybrid {i}] Page: {page} | Doc: {fname} ===")
            if heading:
                print(f"🔖 Section: {heading}")
            if has_diagram and diagram_paths:
                print("🖼️  Architecture Diagram(s) on Disk:")
                for dp in diagram_paths:
                    print(f"   -> {dp}")
            print("-" * 65)
            print(doc.content or "")
            print("=" * 65 + "\n")

        if generate:
            generate_answer(query_text, unique_docs, stream=stream, show_reasoning=show_reasoning)
        return []



    # ------------------------------------------------------------------
    # 2️⃣  Original dense + ColBERT query (unchanged)
    # ------------------------------------------------------------------
    results = client.query_points(
        collection_name=settings.qdrant_index,
        prefetch=models.Prefetch(
            query=query_dense,
            using="dense",
            limit=prefetch_limit,
            filter=query_filter,
        ),
        query=query_colbert,
        using="colbert",
        limit=top_k,
        query_filter=query_filter,
    )
    elapsed = (time.time() - t0) * 1000

    filter_info = " [FILTER: Diagrams Only]" if diagrams_only else ""
    print(f"\n[QUERY] '{query_text}'{filter_info}")
    print(f"[SEARCH] Dense Prefetch ({prefetch_limit}) + ColBERT MaxSim Rescore -> Top {top_k} ({elapsed:.1f}ms)\n")

    if not results.points:
        print("No matching documents found.")
        return []

    # Wrap hits as _Doc objects so _dedup_docs can handle them uniformly
    class _HitDoc:
        def __init__(self, hit):
            self.content = hit.payload.get("text", "")
            self.meta    = hit.payload
            self._hit    = hit

    hit_docs  = [_HitDoc(h) for h in results.points]
    deduped   = _dedup_docs(hit_docs, keep=top_k)
    unique_points = [d._hit for d in deduped]

    if len(unique_points) < len(results.points):
        removed = len(results.points) - len(unique_points)
        print(f"[DEDUP] {len(results.points)} → {len(unique_points)} unique results (removed {removed} duplicates).\n")


    for i, hit in enumerate(unique_points, 1):
        payload = hit.payload
        doc_id        = payload.get("doc_id", "Unknown")
        filename      = payload.get("filename", "")
        page          = payload.get("page", "N/A")
        heading       = payload.get("heading", "")
        text          = payload.get("text", "").strip()
        has_diagram   = payload.get("has_diagram", False)
        diagram_paths = payload.get("diagram_paths", [])

        print(f"=== [Result {i}] Score: {hit.score:.4f} | Page: {page} | Doc: {filename or doc_id} ===")
        if heading:
            print(f"🔖 Section: {heading}")
        if has_diagram and diagram_paths:
            print("🖼️  Architecture Diagram(s) on Disk:")
            for dp in diagram_paths:
                print(f"   -> {dp}")
        print("-" * 65)
        print(text)   # full text — no truncation
        print("=" * 65 + "\n")

    # ------------------------------------------------------------------
    # 3️⃣  Optional LLM answer generation (dense+ColBERT path)
    # ------------------------------------------------------------------
    if generate:
        docs = [_make_simple_doc(hit.payload.get("text", ""), hit.payload) for hit in unique_points]
        generate_answer(query_text, docs, stream=stream, show_reasoning=show_reasoning)

    return unique_points



# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Hybrid (Dense + ColBERT) search for ETSI MEC specifications."
    )
    parser.add_argument("query", type=str, help="Search query (e.g., 'What is Mp1 reference point?')")
    parser.add_argument("--top-k",    type=int, default=5,  help="Number of results to return (default: 5)")
    parser.add_argument("--prefetch", type=int, default=25, help="Dense candidates to prefetch (default: 25)")
    parser.add_argument(
        "--diagrams-only", action="store_true",
        help="Filter results to only passages containing architectural diagrams/charts",
    )
    parser.add_argument(
        "--use-bm25", action="store_true",
        help="Enable Haystack hybrid retrieval (requires haystack-ai + qdrant-haystack)",
    )
    parser.add_argument(
        "--answer", action="store_true",
        help="Generate an LLM answer via OpenRouter free model (requires OPENROUTER_API_KEY)",
    )
    parser.add_argument(
        "--stream", action="store_true",
        help="Stream the LLM answer token-by-token (only has effect with --answer)",
    )
    parser.add_argument(
        "--show-reasoning", action="store_true",
        help="Print the model's internal reasoning_details thinking chain before the answer (non-streaming only)",
    )

    args = parser.parse_args()
    search_specs(
        args.query,
        top_k=args.top_k,
        prefetch_limit=args.prefetch,
        diagrams_only=args.diagrams_only,
        use_bm25=args.use_bm25,
        generate=args.answer,
        stream=args.stream,
        show_reasoning=args.show_reasoning,
    )



if __name__ == "__main__":
    main()
