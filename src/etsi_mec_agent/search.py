import argparse
import os
import time

from qdrant_client import models

from etsi_mec_agent.config import settings
from etsi_mec_agent.dedup import SimpleDoc, _dedup_docs, _stitch_parts
from etsi_mec_agent.embeddings import get_provider
from etsi_mec_agent.store import get_qdrant_client, verify_collection_identity


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


def _run_hybrid_retrieval(
    client,
    query_dense: list,
    query_sparse,
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
                query=query_sparse, using="sparse",
                limit=prefetch_limit, filter=query_filter,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
        query_filter=query_filter,
    )

    # The fused RRF score rides along in meta so the UI can display it; a stitched run keeps
    # the first part's score, and dedup/grounding ignore the extra key.
    return [SimpleDoc(h.payload.get("text", ""), {**h.payload, "score": h.score}) for h in results.points]


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
_SPEC_RE = _re.compile(r"\bMEC(?:[- ]?DEC)?[- ]?\d{3}(?:[- ]?\d+)?", _re.IGNORECASE)
# A part number is canonicalised, not just matched: models write "MEC 010-02" where the cover says
# "MEC 010-2", and the two must resolve to the same specification.
_PART_RE = _re.compile(r"^(MEC(?:-DEC)?-\d{3})-0*(\d+)$")
# An explicit edition in a citation ("V4.1.1") is checked against the stamped editions,
# because p.15 of one edition is not p.15 of another.
_EDITION_RE = r"\b[Vv]\d+(?:\.\d+){1,3}\b"


def _plain(text: str) -> str:
    """ASCII spaces and hyphens, so the matchers see what a citation means rather than how it was typed."""
    return _DASH_RE.sub("-", _SPACE_RE.sub(" ", text))


def _norm_spec(code: str) -> str:
    """`MEC003`, `MEC 003`, `MEC-DEC 032-2`, `MEC 010-02` all become the one form `spec_id` stores."""
    s = _re.sub(r"\s+", "-", code.upper())
    s = _re.sub(r"(\d)([A-Z])", r"\1-\2", _re.sub(r"([A-Z])(\d)", r"\1-\2", s))
    return _PART_RE.sub(r"\1-\2", s)


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
    """{canonical spec: (pages, editions)} over the excerpts that carry a stamped identity."""
    pages = {}
    for doc in documents:
        meta = getattr(doc, "meta", None) or {}
        spec = meta.get("spec_id")
        if spec and isinstance(meta.get("page"), int):
            entry = pages.setdefault(_norm_spec(spec), (set(), set()))
            entry[0].add(meta["page"])
            if meta.get("edition"):
                entry[1].add(meta["edition"])
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
        # The endpoint of a range is a claim too; the middle pages of "pp.15-17" are prose slack
        # that no single excerpt has to cover.
        span = _re.match(r"\s*[-–—]\s*(\d{1,4})", answer[hit.end():])
        claimed = [page] + ([int(span.group(1))] if span else [])
        # Nearest gap wins; when two codes tie at zero distance the one on the page's own
        # side claims it — "p.15, MEC-016" must not read the gap left of MEC-016 as p.15's.
        # Prose form ("page 6 of MEC-016") belongs to the code that follows it.
        page_compact = hit.group(1) is not None
        near = min(codes,
                   key=lambda c: (max(c[1] - hit.end(), 0) if page_compact
                                  else max(hit.start() - c[2], 0),
                                  min(abs(c[1] - hit.start()), abs(c[2] - hit.start()))),
                   default=None)
        if near is None or min(abs(near[1] - hit.start()), abs(near[2] - hit.start())) > _CITE_WINDOW:
            complaints.append(f"p.{page} cited with no specification named")
            continue
        if near[0] not in pages:
            for p in claimed:
                complaints.append(f"{near[0]} p.{p} — no excerpt from this specification was retrieved")
            continue
        have_pages, have_editions = pages[near[0]]
        for p in claimed:
            if p not in have_pages:
                have = ", ".join(f"p.{x}" for x in sorted(have_pages))
                complaints.append(f"{near[0]} p.{p} — not in that spec's excerpts (they are {have})")
        # Prefer the edition written between the spec code and the page; the forward window is
        # for "p.15 of MEC-016 V3.1.1" and must stop at the next citation's spec code, or the
        # next citation's edition would be attributed to this page.
        tail = answer[hit.end():hit.end() + 24]
        later = [c[1] - hit.end() for c in codes if c is not near and c[1] >= hit.end()]
        if later:
            tail = tail[:min(later)]
        cited = (_re.search(_EDITION_RE, answer[max(near[1], hit.start() - 60):hit.start()])
                 or _re.search(_EDITION_RE, tail))
        if cited and have_editions:
            num = _re.search(r"\d+(?:\.\d+)+", cited.group())
            # A version match, not a substring: "1.1" is not the retrieved "4.1.1".
            if num and not any(e.lstrip("Vv") == num.group()
                               or e.lstrip("Vv").startswith(num.group() + ".")
                               for e in have_editions):
                complaints.append(
                    f"{near[0]} p.{page} — cites edition {cited.group()}, retrieved excerpts of "
                    f"this spec are {', '.join(sorted(have_editions))}")
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
              f"the router, not an answer. Re-run, or pin a model with LLM_MODEL.")
    else:
        print("[CITATION CHECK] the answer cites no page numbers.")


def _llm_client():
    """OpenAI-compatible client for the configured endpoint (default: OpenRouter free router)."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "openai package is required for --answer. Install with:\n"
            "  uv add openai"
        ) from exc

    # OPENROUTER_API_KEY is a fallback for the OpenRouter route only — a key must never be
    # carried to an unrelated base_url.
    api_key = settings.llm_api_key
    if not api_key and "openrouter" in settings.llm_base_url:
        api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            f"No API key for {settings.llm_base_url} — set LLM_API_KEY "
            "(OPENROUTER_API_KEY also works for the OpenRouter route)."
        )
    return OpenAI(base_url=settings.llm_base_url, api_key=api_key)


_SYSTEM_PROMPT = (
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
)


def _build_messages(query: str, documents: list) -> tuple:
    """(chat messages, one citation label per excerpt) — shared by generate_answer and stream_answer."""
    context_text, source_labels = _build_context(documents)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context excerpts from ETSI MEC specifications:\n\n{context_text}\n\nQuestion: {query}",
        },
    ]
    return messages, source_labels


def _extra_body():
    """Extra request body sent with every completion for the configured endpoint."""
    # reasoning.enabled is an OpenRouter extension; other OpenAI-compatible endpoints may reject it.
    return {"reasoning": {"enabled": True}} if "openrouter" in settings.llm_base_url else None


def stream_answer(query: str, documents: list, meta_out: dict | None = None):
    """Yield answer deltas from the configured endpoint — the shape st.write_stream wants.

    meta_out (when given) receives "model" — which model a free router actually served, since
    openrouter/free is a moving target — and "answer", the reassembled full text.
    """
    client = _llm_client()
    messages, _ = _build_messages(query, documents)
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        extra_body=_extra_body(),
        stream=True,
    )
    full = ""
    if meta_out is not None:
        meta_out["model"] = ""  # per-call provenance; never inherited from an earlier stream
    try:
        for chunk in response:
            if meta_out is not None:
                meta_out["model"] = meta_out.get("model") or getattr(chunk, "model", "") or ""
            if not chunk.choices:
                continue  # usage-only / content-filter chunks carry no choices
            delta = chunk.choices[0].delta.content
            if delta:
                full += delta
                yield delta
    finally:
        response.close()  # an abandoned stream (browser stop) must not hold the HTTP connection
        if meta_out is not None:
            meta_out["answer"] = full


def generate_answer(query: str, documents: list, stream: bool = False, show_reasoning: bool = False) -> str:
    """
    Call the configured OpenAI-compatible endpoint (settings.llm_*, default openrouter/free)
    with the retrieved ETSI MEC chunks as context.

    model=settings.llm_model   — LLM_MODEL env var; default confirmed correct per OpenRouter docs
    reasoning.enabled=True     — OpenRouter models return reasoning_details (thinking chain)
                                 alongside the final .content answer
    stream=True                — streams final answer token-by-token
    show_reasoning=True        — also prints the reasoning_details thinking block

    The prompt is labelled with the identity printed on each PDF and the finished answer is
    audited against those labels, because the terminal prints only --top-k of the
    --answer-context excerpts the model actually read.
    """
    messages, source_labels = _build_messages(query, documents)
    print(f"\n[ANSWER SOURCES] {len(source_labels)} excerpts fed to the LLM "
          f"(only --top-k of them are printed above):")
    for i, label in enumerate(source_labels, 1):
        print(f"  [{i}] {label}")

    if stream:
        # Streaming: content tokens arrive chunk-by-chunk.
        # reasoning_details are not available in streaming mode — they only
        # come back in a non-streaming response.
        print("\n[LLM ANSWER] ", end="", flush=True)
        meta_out = {}
        for delta in stream_answer(query, documents, meta_out):
            print(delta, end="", flush=True)
        print()
        full = meta_out.get("answer", "")
        _report_grounding(full, documents, meta_out.get("model", ""))
        return full

    client = _llm_client()
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        extra_body=_extra_body(),
        stream=False,
    )
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
    full_result: bool = False,
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

    With `full_result=True` (the Streamlit UI's shape) nothing is generated and the return is
    `{"display": [...], "answer_docs": [...], "info": {...}}` instead of the bare display list;
    the prints still happen, so a caller that wants them redirects stdout.
    """
    if generate and full_result:
        raise ValueError("full_result is the UI shape and never generates — pass generate=False")
    client = get_qdrant_client()
    # Before the embedders load (cold ONNX costs 30-60s): if the stored vectors came from another
    # provider, every score below is noise, so abort naming the mismatch instead.
    verify_collection_identity(client)
    want_answer = generate or full_result
    # The LLM reads more than the terminal prints: retrieval must fetch enough
    # candidates to fill the evidence budget after stitching and dedup.
    fetch_k = max(top_k, answer_context) if want_answer else top_k
    # A page's sibling part has to be in the list at all before it can merge, and ColBERT
    # rescore returns exactly `limit` results with no post-filter slack: over-fetch first,
    # stitch, then let dedup pick the display and evidence budgets (as the hybrid branch does).
    colbert_k = fetch_k * 3 if want_answer else fetch_k
    # The dense prefetch is the rerank pool and therefore the ceiling on how many excerpts
    # ColBERT can rescore: a bigger evidence budget needs a bigger pool. The budget is 15
    # excerpts while the display is 5, and _stitch_parts can only merge parts that were
    # retrieved at all, so the pool must be 3x the evidence budget rather than its width.
    prefetch_limit = max(prefetch_limit, fetch_k * 3)
    qv = get_provider().query_vectors(query_text)
    query_dense, query_colbert = qv["dense"], qv["colbert"]

    t0 = time.time()

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
        hybrid_hits = _run_hybrid_retrieval(
            client=client,
            query_dense=query_dense,
            query_sparse=qv["sparse"],
            top_k=fetch_k * 6,        # fetch 6× more so dedup still fills the evidence budget
            prefetch_limit=max(prefetch_limit * 4, 100),
            query_filter=query_filter,
        )
        stitched = _stitch_parts(hybrid_hits)

        if not stitched:
            print("No matching documents found.")
            if full_result:
                return {"display": [], "answer_docs": [], "info": {
                    "path": "Hybrid RRF (dense + sparse, fused server-side in Qdrant)",
                    "elapsed_ms": (time.time() - t0) * 1000, "fetched": 0, "stitched": 0, "shown": 0,
                    "filter_info": filter_info.strip(), "merged_groups": 0, "answer_words": 0}}
            return []

        unique_docs = _dedup_docs(stitched, keep=top_k)
        answer_docs = _dedup_docs(stitched, keep=answer_context, per_doc=per_doc) if want_answer else []

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
        if full_result:
            return {"display": unique_docs, "answer_docs": answer_docs, "info": {
                "path": "Hybrid RRF (dense + sparse, fused server-side in Qdrant)",
                "elapsed_ms": (time.time() - t0) * 1000,
                "fetched": len(hybrid_hits),
                "stitched": len(stitched), "shown": len(unique_docs),
                "answer_context": answer_context, "per_doc": per_doc,
                "filter_info": filter_info.strip(),
                "merged_groups": sum(1 for d in stitched if (d.meta or {}).get("stitched_parts", 1) > 1),
                "answer_words": sum(len(d.content.split()) for d in answer_docs),
            }}
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
        if full_result:
            return {"display": [], "answer_docs": [], "info": {
                "path": f"Dense prefetch ({prefetch_limit}) + ColBERT MaxSim re-rank (top {colbert_k})",
                "elapsed_ms": (time.time() - t0) * 1000, "fetched": 0,
                "stitched": 0, "shown": 0, "filter_info": filter_info.strip(),
                "merged_groups": 0, "answer_words": 0}}
        return []

    # Stitch first, then shape: parts 1+2 of one page come back as one excerpt, so the
    # merged text — not half a table row — is what dedup, printing and the LLM all see.
    # The MaxSim score rides along in meta so the UI can display it; a stitched run keeps
    # the first part's score, and dedup/grounding ignore the extra key.
    hit_docs  = _stitch_parts([SimpleDoc(h.payload.get("text", ""), {**h.payload, "score": h.score})
                               for h in results.points])
    deduped   = _dedup_docs(hit_docs, keep=top_k)
    answer_docs = _dedup_docs(hit_docs, keep=answer_context, per_doc=per_doc) if want_answer else []

    if len(deduped) < len(hit_docs):
        print(f"[DEDUP] {len(hit_docs)} fetched → {len(deduped)} shown "
              f"(page parts merged first; the rest were duplicates or beyond --top-k).")


    # The loop reads doc.meta: a stitched excerpt is several hits and shows the first part's score.
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

    if full_result:
        return {"display": deduped, "answer_docs": answer_docs, "info": {
            "path": f"Dense prefetch ({prefetch_limit}) + ColBERT MaxSim re-rank (top {colbert_k})",
            "elapsed_ms": elapsed,
            "fetched": len(results.points),
            "stitched": len(hit_docs), "shown": len(deduped),
            "answer_context": answer_context, "per_doc": per_doc,
            "filter_info": filter_info.strip(),
            "merged_groups": sum(1 for d in hit_docs if (d.meta or {}).get("stitched_parts", 1) > 1),
            "answer_words": sum(len(d.content.split()) for d in answer_docs),
        }}
    return deduped



# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    """Parse the CLI flags below and hand them to search_specs."""
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
        help="Generate an LLM answer via the configured OpenAI-compatible endpoint "
             "(LLM_BASE_URL/LLM_MODEL, key: LLM_API_KEY or OPENROUTER_API_KEY; default: OpenRouter free)",
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
