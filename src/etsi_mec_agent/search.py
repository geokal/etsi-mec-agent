import argparse
import os
import time
import zlib

from fastembed import LateInteractionTextEmbedding, TextEmbedding
from qdrant_client import models

from etsi_mec_agent.config import settings
from etsi_mec_agent.dedup import SimpleDoc, _dedup_docs, _stitch_parts
from etsi_mec_agent.store import get_qdrant_client


def get_embedders():
    dense_model = TextEmbedding(settings.dense_model)
    colbert_model = LateInteractionTextEmbedding(settings.colbert_model)
    return dense_model, colbert_model


# ---------------------------------------------------------------------------
# Hybrid retrieval — dense + server-side sparse BM25 (no Haystack required)
# ---------------------------------------------------------------------------

import re as _re


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split into word tokens."""
    return _re.findall(r"[a-z0-9]+", text.lower())

def _token_sparse(text: str) -> models.SparseVector:
    """Raw term-frequency sparse vector; token ids are crc32 hashes so no shared
    vocabulary file is needed between ingest and query time. Qdrant applies IDF
    weighting at query time via Modifier.IDF on the collection config."""
    tf: dict[int, int] = {}
    for tok in _tokenize(text):
        i = zlib.crc32(tok.encode()) & 0x7FFFFFFF
        tf[i] = tf.get(i, 0) + 1
    return models.SparseVector(indices=list(tf), values=[float(v) for v in tf.values()])


def _run_hybrid_retrieval(
    client,
    query_text: str,
    query_dense: list,
    top_k: int = 5,
    prefetch_limit: int = 50,
    query_filter=None,
) -> list:
    """
    Server-side hybrid retrieval over the existing Qdrant collection:

    Two prefetches — dense (semantic recall) and sparse BM25-style term
    frequencies (exact identifiers like "Mm4", "Mp1", clause numbers) —
    fused with RRF inside Qdrant. The collection declares Modifier.IDF on
    the sparse vector, so rarity is computed corpus-wide instead of the
    old per-candidate approximation.

    Returns up to top_k SimpleDocs — .content is the stored payload text, .meta is the
    payload — the shape generate_answer() and _dedup_docs read.
    """
    results = client.query_points(
        collection_name=settings.qdrant_index,
        prefetch=[
            models.Prefetch(
                query=query_dense, using="dense", limit=prefetch_limit, filter=query_filter
            ),
            models.Prefetch(
                query=_token_sparse(query_text), using="sparse",
                limit=prefetch_limit, filter=query_filter,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
        query_filter=query_filter,
    )

    return [SimpleDoc(h.payload.get("text", ""), h.payload) for h in results.points]


# ---------------------------------------------------------------------------
# LLM answer generation (OpenRouter free model, OpenAI-compatible SDK)
# ---------------------------------------------------------------------------

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
    answer_context: int = 15,
    per_doc: int = 2,
    generate: bool = False,
    stream: bool = False,
    show_reasoning: bool = False,
):
    """
    Execute Hybrid Search (Dense + ColBERT) with optional BM25+Vector hybrid
    and optional LLM answer generation.

    Display and evidence are two budgets: the terminal prints at most `top_k` excerpts,
    while an LLM answer reads up to `answer_context` excerpts with at most `per_doc` per
    document. That is what keeps the continuation of a long table in the prompt instead of
    dropping it under the one-excerpt-per-document display rule.

    * Without --use-bm25    : original dense + ColBERT Qdrant query (unchanged).
    * With    --use-bm25    : Qdrant server-side dense+sparse hybrid with RRF fusion.
    * With    --answer      : feeds the retrieved chunks to the OpenRouter LLM.
    * With    --show-reasoning : also prints the model's reasoning_details thinking chain.
    """
    client = get_qdrant_client()
    # The LLM reads more than the terminal prints: retrieval must fetch enough
    # candidates to fill the evidence budget after stitching and dedup.
    fetch_k = max(top_k, answer_context) if generate else top_k
    # The dense prefetch is the rerank pool and therefore the ceiling on how many excerpts
    # ColBERT can rescore: a bigger evidence budget needs a bigger pool.
    prefetch_limit = max(prefetch_limit, fetch_k)
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
    # 1️⃣  Hybrid retrieval (dense + sparse prefetch, fused server-side by RRF)
    # ------------------------------------------------------------------
    if use_bm25:
        print(f"\n[QUERY] '{query_text}' [HYBRID: Dense + sparse prefetch → server-side RRF]")
        stitched = _stitch_parts(_run_hybrid_retrieval(
            client=client,
            query_text=query_text,
            query_dense=query_dense,
            top_k=fetch_k * 6,        # fetch 6× more so dedup still fills the evidence budget
            prefetch_limit=max(prefetch_limit * 4, 100),
            query_filter=query_filter,
        ))

        unique_docs = _dedup_docs(stitched, keep=top_k)
        answer_docs = _dedup_docs(stitched, keep=answer_context, per_doc=per_doc) if generate else []

        if len(unique_docs) < len(stitched):
            print(f"[DEDUP] {len(stitched)} → {len(unique_docs)} unique document(s) (removed {len(stitched)-len(unique_docs)} duplicates).\n")

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
            print(f"[ANSWER CONTEXT] {len(answer_docs)} excerpts, up to {per_doc} per document "
                  f"(~{sum(len(d.content.split()) for d in answer_docs)} words).")
            generate_answer(query_text, answer_docs, stream=stream, show_reasoning=show_reasoning)
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
        limit=fetch_k,
        query_filter=query_filter,
    )
    elapsed = (time.time() - t0) * 1000

    filter_info = " [FILTER: Diagrams Only]" if diagrams_only else ""
    print(f"\n[QUERY] '{query_text}'{filter_info}")
    print(f"[SEARCH] Dense Prefetch ({prefetch_limit}) + ColBERT MaxSim Rescore -> "
          f"Top {fetch_k} fetched, {top_k} displayed ({elapsed:.1f}ms)\n")

    if not results.points:
        print("No matching documents found.")
        return []

    # Wrap hits as _HitDoc objects so _dedup_docs can handle them uniformly
    class _HitDoc:
        def __init__(self, hit):
            self.content = hit.payload.get("text", "")
            self.meta    = hit.payload

    # Stitch first, then shape: parts 1+2 of one page come back as one excerpt, so the
    # merged text — not half a table row — is what dedup, printing and the LLM all see.
    hit_docs  = _stitch_parts([_HitDoc(h) for h in results.points])
    deduped   = _dedup_docs(hit_docs, keep=top_k)
    answer_docs = _dedup_docs(hit_docs, keep=answer_context, per_doc=per_doc) if generate else []

    if len(deduped) < len(hit_docs):
        removed = len(hit_docs) - len(deduped)
        print(f"[DEDUP] {len(hit_docs)} stitched excerpts → {len(deduped)} unique results "
              f"(page parts were merged first, then {removed} duplicate excerpt(s) dropped).\n")


    # Merged runs are SimpleDocs without a backing hit, so the loop reads doc.meta and
    # drops the per-hit score: a stitched excerpt is several hits and has no single score.
    for i, doc in enumerate(deduped, 1):
        payload = doc.meta or {}
        doc_id        = payload.get("doc_id", "Unknown")
        filename      = payload.get("filename", "")
        page          = payload.get("page", "N/A")
        heading       = payload.get("heading", "")
        text          = (doc.content or "").strip()
        has_diagram   = payload.get("has_diagram", False)
        diagram_paths = payload.get("diagram_paths", [])

        print(f"=== [Result {i}] Page: {page} | Doc: {filename or doc_id} ===")
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
        print(f"[ANSWER CONTEXT] {len(answer_docs)} excerpts, up to {per_doc} per document "
              f"(~{sum(len(d.content.split()) for d in answer_docs)} words).")
        generate_answer(query_text, answer_docs, stream=stream, show_reasoning=show_reasoning)

    return deduped



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
    parser.add_argument("--answer-context", type=int, default=15,
                        help="Excerpts handed to the LLM with --answer (terminal still shows --top-k)")
    parser.add_argument("--per-doc", type=int, default=2,
                        help="Excerpts per document inside --answer-context (>= 1)")
    parser.add_argument(
        "--diagrams-only", action="store_true",
        help="Filter results to only passages containing architectural diagrams/charts",
    )
    parser.add_argument(
        "--use-bm25", action="store_true",
        help="Hybrid retrieval: dense + sparse BM25 fused server-side by Qdrant (RRF)",
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
    # _dedup_docs clamps for programmatic callers; someone typing --per-doc 0 means "no
    # limit", so say so instead of silently handing them one excerpt per document.
    if args.per_doc < 1:
        parser.error("--per-doc must be >= 1")
    if args.answer_context < 1:
        parser.error("--answer-context must be >= 1")
    search_specs(
        args.query,
        top_k=args.top_k,
        prefetch_limit=args.prefetch,
        diagrams_only=args.diagrams_only,
        use_bm25=args.use_bm25,
        answer_context=args.answer_context,
        per_doc=args.per_doc,
        generate=args.answer,
        stream=args.stream,
        show_reasoning=args.show_reasoning,
    )



if __name__ == "__main__":
    main()
