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


def build_filter(current_only: bool = True, diagrams_only: bool = False):
    """The payload filter search_specs sends, shared with scripts/eval_rag.py.

    must_not is_current=false rather than must is_current=true: chunks written before
    scripts/backfill_spec_identity.py carry no is_current field at all, and a missing field
    only survives the must_not form.
    """
    must, must_not = [], []
    if diagrams_only:
        must.append(models.FieldCondition(key="has_diagram", match=models.MatchValue(value=True)))
    if current_only:
        must_not.append(
            models.FieldCondition(key="is_current", match=models.MatchValue(value=False))
        )
    return models.Filter(must=must, must_not=must_not) if (must or must_not) else None


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

# ---------------------------------------------------------------------------
# Answer grounding — a citation must trace back to the excerpt it came from
# ---------------------------------------------------------------------------

# A window, not a bracket parser: ETSI citations nest parentheses, as in
# "(ETSI GS MEC 003 V4.1.1 (2025-05) p.15)", so a page is attributed to the nearest spec code
# around it — which balanced-bracket matching gets wrong.
_CITE_WINDOW = 120
# Models answer typographically, not in ASCII: the live run cited "MEC‑059" with a non-breaking
# hyphen (U+2011) and "MEC 003" with a non-breaking space, and an ASCII-only matcher called those
# two correct citations unsupported. Flatten before matching.
_SPACE_RE = _re.compile("[\\u00a0\\u1680\\u2000-\\u200a\\u202f\\u205f\\u3000\\u200b]")
_DASH_RE = _re.compile("[\\u2010-\\u2015\\u2212\\uff0d]")
# "p.15", "pp. 15" and the prose form "page 6" are all citations.
_PAGE_RE = _re.compile(r"\b(?:pp?\s*\.?\s*(\d{1,4})|pages?\s+(\d{1,4}))", _re.IGNORECASE)
_SPEC_RE = _re.compile(r"\bMEC(?:[- ]?DEC)?[- ]?\d{3}(?:[- ]?\d)?", _re.IGNORECASE)


def _plain(text: str) -> str:
    """ASCII spaces and hyphens, so the matchers see what a citation means rather than how it was typed."""
    return _DASH_RE.sub("-", _SPACE_RE.sub(" ", text))


def _norm_spec(code: str) -> str:
    """`MEC003`, `MEC 003`, `MEC-DEC 032-2` all become the one form `spec_id` stores."""
    s = _re.sub(r"\s+", "-", code.upper())
    return _re.sub(r"(\d)([A-Z])", r"\1-\2", _re.sub(r"([A-Z])(\d)", r"\1-\2", s))


def _source_label(meta: dict) -> str:
    """The citation an excerpt is known by: the identity printed on the PDF, never its filename."""
    ident = meta.get("spec_id") or meta.get("filename") or "unknown"
    page = meta.get("page", "?")
    edition = meta.get("edition")
    return f"{ident} {edition} p.{page}" if edition else f"{ident} p.{page}"


def _build_context(documents: list) -> tuple:
    """(the context block for the prompt, one citation label per excerpt in the same order)."""
    parts, labels = [], []
    for i, doc in enumerate(documents, 1):
        if hasattr(doc, "meta"):
            meta = doc.meta or {}
            text = getattr(doc, "content", None) or meta.get("text", "")
        else:
            meta, text = {}, str(doc)
        label = _source_label(meta)
        labels.append(label)
        header = f"--- [Source {i}] {label}"
        if meta.get("heading"):
            header += f" — {meta['heading']}"
        diagrams = meta.get("diagram_paths", [])
        if diagrams:
            header += " — 📐 Diagram(s) available: " + ", ".join(diagrams)
        parts.append(f"{header}\n{text}")
    return "\n\n".join(parts), labels


def _pages_by_spec(documents: list) -> dict:
    """{canonical spec: {pages}} over the excerpts that carry a stamped identity."""
    pages = {}
    for doc in documents:
        meta = getattr(doc, "meta", None) or {}
        if meta.get("spec_id") and isinstance(meta.get("page"), int):
            pages.setdefault(_norm_spec(meta["spec_id"]), set()).add(meta["page"])
    return pages


def _ungrounded_citations(answer: str, documents: list) -> list:
    """Page citations the excerpts cannot support, in words a reader can act on.

    A spec that was never retrieved and a known spec cited at a page none of its excerpts
    sit on are different faults: the first is a retrieval gap, the second means the claim
    came from somewhere other than where the answer says it did. A page is attributed to the
    nearest specification code on either side of it, because both "(MEC-016 V3.1.1 p.6)" and
    "page 6 of MEC-016" occur in real answers.
    """
    pages = _pages_by_spec(documents)
    if not pages:
        return []
    answer = _plain(answer)
    # Documents with no ETSI number — white papers and decks — are cited by the name stamped on them,
    # which the MEC pattern cannot see, so the audit learns whatever names the context actually has.
    others = []
    for doc in documents:
        spec = (getattr(doc, "meta", None) or {}).get("spec_id") or ""
        esc = _re.escape(spec)
        if spec and esc not in others and not _SPEC_RE.fullmatch(_norm_spec(spec)):
            others.append(esc)
    finder = _re.compile("|".join([_SPEC_RE.pattern] + sorted(others, key=len, reverse=True)),
                         _re.IGNORECASE) if others else _SPEC_RE
    codes = [(_norm_spec(m.group()), m.start(), m.end()) for m in finder.finditer(answer)]
    complaints = []
    for hit in _PAGE_RE.finditer(answer):
        page = int(hit.group(1) or hit.group(2))
        near = min(codes, key=lambda c: min(abs(c[1] - hit.start()), abs(c[2] - hit.start())),
                   default=None)
        if near is None or min(abs(near[1] - hit.start()), abs(near[2] - hit.start())) > _CITE_WINDOW:
            complaints.append(f"p.{page} cited with no specification named")
        elif near[0] not in pages:
            complaints.append(f"{near[0]} p.{page} — no excerpt from this specification was retrieved")
        elif page not in pages[near[0]]:
            have = ", ".join(f"p.{p}" for p in sorted(pages[near[0]]))
            complaints.append(f"{near[0]} p.{page} — not in that spec's excerpts (they are {have})")
    return list(dict.fromkeys(complaints))


def _report_grounding(answer: str, documents: list, model: str = "") -> None:
    """Print who answered, then whether what they said traces back to the excerpts."""
    print()
    if model:
        print(f"[ANSWER MODEL] {model}")
    if not _pages_by_spec(documents):
        print("[CITATION CHECK] skipped — the excerpts carry no stamped spec identity "
              "(run scripts/backfill_spec_identity.py --apply).")
        return
    suspects = _ungrounded_citations(answer, documents)
    cited = len(_PAGE_RE.findall(_plain(answer)))
    words = len(answer.split())
    if suspects:
        print(f"[CITATION CHECK] {cited} page citation(s), {len(suspects)} unsupported:")
        for line in suspects:
            print(f"  ⚠ {line}")
    elif cited:
        print(f"[CITATION CHECK] {cited} page citation(s), all present in the excerpts above.")
    elif words < 40:
        # The free router sometimes answers with a classifier line instead of a generation; printed
        # as bare text it looks like an answer. Measured 2026-10-08: "User Safety: safe".
        print(f"[CITATION CHECK] {words} word(s) and no citation — that is a canned response from "
              f"the router, not an answer. Re-run: openrouter/free is non-deterministic.")
    else:
        print("[CITATION CHECK] the answer cites no page numbers.")


def generate_answer(query: str, documents: list, stream: bool = False, show_reasoning: bool = False) -> str:
    """
    Call OpenRouter (openrouter/free) with the retrieved ETSI MEC chunks as context.

    model="openrouter/free"   — confirmed correct per OpenRouter docs
    reasoning.enabled=True    — model returns reasoning_details (thinking chain)
                                 alongside the final .content answer
    stream=True               — streams final answer token-by-token
    show_reasoning=True       — also prints the reasoning_details thinking block

    The prompt is labelled with the identity printed on each PDF and the finished answer is
    audited against those labels, because the terminal prints only --top-k of the
    --answer-context excerpts the model actually read.
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

    context_text, source_labels = _build_context(documents)
    print(f"\n[ANSWER SOURCES] {len(source_labels)} excerpts fed to the LLM "
          f"(only --top-k of them are printed above):")
    for i, label in enumerate(source_labels, 1):
        print(f"  [{i}] {label}")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a technical writer specialising in ETSI MEC (Multi-access Edge Computing) "
                "specifications. When answering questions:\n"
                "• Write a comprehensive, well-structured answer of at least 3–5 paragraphs.\n"
                "• Back each factual claim with the specification, edition and page copied from the "
                "[Source …] line it came from, e.g. (MEC-003 V4.1.1 p.18). Cite only pages that "
                "appear on those lines; if no excerpt supports a claim, leave it out.\n"
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
        served = ""
        for chunk in response:
            # Which model the free router actually picked is on the stream chunks, not a header.
            served = served or getattr(chunk, "model", "") or ""
            delta = chunk.choices[0].delta.content
            if delta:
                print(delta, end="", flush=True)
                full += delta
        print()
        _report_grounding(full, documents, served)
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
        _report_grounding(msg.content or "", documents, getattr(response, "model", "") or "")
        return msg.content



# ---------------------------------------------------------------------------
# Core search function
# ---------------------------------------------------------------------------

def search_specs(
    query_text: str,
    top_k: int = 5,
    prefetch_limit: int = 25,
    diagrams_only: bool = False,
    current_only: bool = True,
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
    dropping it under the one-excerpt-per-document display rule. When `generate` is set,
    retrieval widens to `max(top_k, answer_context)` so the evidence budget can fill.

    * Without --use-bm25    : dense prefetch + ColBERT MaxSim rescore, then stitch and dedup.
    * With    --use-bm25    : Qdrant server-side dense+sparse hybrid fused by RRF, then the
                              same stitch and dedup.
    * With    --answer      : feeds the `answer_context` excerpts to the OpenRouter LLM.
    * With    --show-reasoning : also prints the model's reasoning_details thinking chain.
    * By    --all-editions    : keep chunks stamped is_current=false; by default a superseded
                              edition is filtered server-side. Excerpts are bucketed by
                              content_md5, so the doc_ids that are byte-identical copies of
                              another spec share one per-doc quota instead of each claiming
                              its own.

    Both paths print and return the same at-most-`top_k` display excerpts (SimpleDocs with
    .content / .meta), or an empty list when nothing matched.
    """
    client = get_qdrant_client()
    # The LLM reads more than the terminal prints: retrieval must fetch enough
    # candidates to fill the evidence budget after stitching and dedup.
    fetch_k = max(top_k, answer_context) if generate else top_k
    # A page's sibling part has to be in the list at all before it can merge, and ColBERT
    # rescore returns exactly `limit` results with no post-filter slack: over-fetch first,
    # stitch, then let dedup pick the display and evidence budgets (as the hybrid branch does).
    colbert_k = fetch_k * 3 if generate else fetch_k
    # The dense prefetch is the rerank pool and therefore the ceiling on how many excerpts
    # ColBERT can rescore: a bigger evidence budget needs a bigger pool. The budget is 15
    # excerpts while the display is 5, and _stitch_parts can only merge parts that were
    # retrieved at all, so the pool must be 3x the evidence budget rather than its width.
    prefetch_limit = max(prefetch_limit, fetch_k * 3)
    dense_model, colbert_model = get_embedders()

    t0 = time.time()
    query_dense = list(dense_model.embed([query_text]))[0].tolist()
    query_colbert = list(colbert_model.query_embed(query_text))[0].tolist()

    query_filter = build_filter(current_only=current_only, diagrams_only=diagrams_only)
    notes = (["current editions only"] if current_only else []) + \
            (["diagrams only"] if diagrams_only else [])
    filter_info = f" [FILTER: {', '.join(notes)}]" if notes else ""

    # ------------------------------------------------------------------
    # 1️⃣  Hybrid retrieval (dense + sparse prefetch, fused server-side by RRF)
    # ------------------------------------------------------------------
    if use_bm25:
        print(f"\n[QUERY] '{query_text}' [HYBRID: Dense + sparse prefetch → server-side RRF]"
              f"{filter_info}")
        stitched = _stitch_parts(_run_hybrid_retrieval(
            client=client,
            query_text=query_text,
            query_dense=query_dense,
            top_k=fetch_k * 6,        # fetch 6× more so dedup still fills the evidence budget
            prefetch_limit=max(prefetch_limit * 4, 100),
            query_filter=query_filter,
        ))

        if not stitched:
            print("No matching documents found.")
            return []

        unique_docs = _dedup_docs(stitched, keep=top_k)
        answer_docs = _dedup_docs(stitched, keep=answer_context, per_doc=per_doc) if generate else []

        if len(unique_docs) < len(stitched):
            print(f"[DEDUP] {len(stitched)} fetched → {len(unique_docs)} shown "
                  f"(page parts merged first; the rest were duplicates or beyond --top-k).")

        print(f"[HYBRID] Retrieved {len(unique_docs)} unique document(s).\n")
        for i, doc in enumerate(unique_docs, 1):
            meta          = doc.meta or {}
            fname         = meta.get("filename", "<unknown>")
            spec          = meta.get("spec_id")
            label         = f"{fname} ({spec} {meta.get('edition')})" if spec else fname
            page          = meta.get("page", "N/A")
            heading       = meta.get("heading", "")
            has_diagram   = meta.get("has_diagram", False)
            diagram_paths = meta.get("diagram_paths", [])

            print(f"=== [Hybrid {i}] Page: {page} | Doc: {label} ===")
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
                  f"(~{sum(len(d.content.split()) for d in answer_docs)} words), "
                  f"{sum(1 for d in stitched if (d.meta or {}).get('stitched_parts', 1) > 1)} page-part group(s) merged.")
            generate_answer(query_text, answer_docs, stream=stream, show_reasoning=show_reasoning)
        return unique_docs



    # ------------------------------------------------------------------
    # 2️⃣  Dense prefetch + ColBERT MaxSim rescore
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
        limit=colbert_k,
        query_filter=query_filter,
    )
    elapsed = (time.time() - t0) * 1000

    print(f"\n[QUERY] '{query_text}'{filter_info}")
    print(f"[SEARCH] Dense Prefetch ({prefetch_limit}) + ColBERT MaxSim Rescore -> "
          f"Top {colbert_k} fetched, {top_k} displayed ({elapsed:.1f}ms)\n")

    if not results.points:
        print("No matching documents found.")
        return []

    # Stitch first, then shape: parts 1+2 of one page come back as one excerpt, so the
    # merged text — not half a table row — is what dedup, printing and the LLM all see.
    hit_docs  = _stitch_parts([SimpleDoc(h.payload.get("text", ""), h.payload) for h in results.points])
    deduped   = _dedup_docs(hit_docs, keep=top_k)
    answer_docs = _dedup_docs(hit_docs, keep=answer_context, per_doc=per_doc) if generate else []

    if len(deduped) < len(hit_docs):
        print(f"[DEDUP] {len(hit_docs)} fetched → {len(deduped)} shown "
              f"(page parts merged first; the rest were duplicates or beyond --top-k).")


    # Merged runs are SimpleDocs without a backing hit, so the loop reads doc.meta and
    # drops the per-hit score: a stitched excerpt is several hits and has no single score.
    for i, doc in enumerate(deduped, 1):
        payload = doc.meta or {}
        doc_id        = payload.get("doc_id", "Unknown")
        filename      = payload.get("filename", "")
        spec          = payload.get("spec_id")
        label         = (f"{filename or doc_id} ({spec} {payload.get('edition')})"
                         if spec else (filename or doc_id))
        page          = payload.get("page", "N/A")
        heading       = payload.get("heading", "")
        text          = (doc.content or "").strip()
        has_diagram   = payload.get("has_diagram", False)
        diagram_paths = payload.get("diagram_paths", [])

        print(f"=== [Result {i}] Page: {page} | Doc: {label} ===")
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
              f"(~{sum(len(d.content.split()) for d in answer_docs)} words), "
              f"{sum(1 for d in hit_docs if (d.meta or {}).get('stitched_parts', 1) > 1)} page-part group(s) merged.")
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
    parser.add_argument("--top-k",    type=int, default=5,  help="Excerpts to print in the terminal (default: 5); --answer reads --answer-context")
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
        "--all-editions", action="store_true",
        help="Also retrieve superseded editions of a spec (default: current edition only)",
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
    if args.top_k < 1:
        parser.error("--top-k must be >= 1")
    search_specs(
        args.query,
        top_k=args.top_k,
        prefetch_limit=args.prefetch,
        diagrams_only=args.diagrams_only,
        current_only=not args.all_editions,
        use_bm25=args.use_bm25,
        answer_context=args.answer_context,
        per_doc=args.per_doc,
        generate=args.answer,
        stream=args.stream,
        show_reasoning=args.show_reasoning,
    )



if __name__ == "__main__":
    main()
