"""Rank shaping for retrieval results: stitching passage parts and dedup.

Stdlib-only by design. `search.py` imports fastembed at module level, so anything
that lives here would otherwise drag ONNX runtime into every check script.
"""
import re
from dataclasses import dataclass

# strip markdown image refs before fingerprinting
_IMG_RE = re.compile(r"!\[.*?\]\(.*?\)")


@dataclass(frozen=True)
class SimpleDoc:
    """A retrieved excerpt: its text plus the Qdrant payload it came with."""

    content: str
    meta: dict


def _stitch(a: str, b: str, max_overlap: int = 60) -> str:
    """Append b to a, dropping the tokens b repeats from the end of a.

    Only for contiguous parts of one page: sliced parts are `" ".join`ed words, so
    token comparison is exact, but a page short enough to be one chunk keeps raw
    markdown newlines and this function would flatten them.
    """
    # ponytail: must stay >= ingest chunk_page_text overlap (40); a smaller window silently
    # keeps the duplicate instead of trimming. Upgrade path: pass the overlap in from the caller.
    aw, bw = a.split(), b.split()
    n = min(len(aw), len(bw), max_overlap)
    # k stops at 2: one shared token isn't the 40-word overlap; show the duplicate instead
    for k in range(n, 1, -1):
        if aw[-k:] == bw[:k]:
            return " ".join(aw + bw[k:])
    return " ".join(aw + bw)


def _dedup_docs(docs: list, keep: int, per_doc: int = 1) -> list:
    """
    Two-pass deduplication of retrieved hits. Caller contract: every item in `docs` must
    expose `.content` (str) and `.meta` (dict; falsy counts as empty).

    Pass 1 — text fingerprint: markdown images are stripped from the content first and
    the first 400 chars of what remains are compared, in that order, which is why an
    image path inside the window cannot break a match. This catches the same chunk
    stored under different doc_ids, which is common in this corpus because each PDF
    extraction embeds a different image path in the markdown.

    Pass 2 — up to `per_doc` excerpts per doc_id, `keep` results overall. A hit with no
    doc_id is bucketed by the first 40 chars of its text instead, so two different
    documents sharing such a prefix consume one `per_doc` quota together. per_doc=1
    reproduces the historical one-chunk-per-document rule; larger values let a
    document contribute more than its single best-ranked page.
    """
    # per_doc <= 0 would drop every excerpt (the per-doc count starts at 0 and is
    # already >= it); clamp here rather than in argparse for every caller.
    per_doc = max(1, per_doc)
    # keep < 1 must mean "no results": the loop below appends before it checks the cap,
    # so without this guard it hands back one excerpt. search.py's CLI rejects --top-k < 1, but
    # scripts/eval_rag.py still passes its own --top-k straight through as `keep`.
    if keep < 1:
        return []
    seen_text = set()
    after_text = []
    for doc in docs:
        raw = doc.content or ""
        fp = _IMG_RE.sub("", raw)[:400].strip()
        if fp not in seen_text:
            seen_text.add(fp)
            after_text.append(doc)

    # ponytail: pass 1 runs first, so an excerpt it dropped is gone before per_doc is
    # consulted; per_doc cannot rescue it. Upgrade path: a single pass in rank order that
    # applies both filters and continues past rejects until keep fills.
    counts = {}
    unique = []
    for doc in after_text:
        meta = doc.meta or {}
        doc_id = meta.get("doc_id", "") or (doc.content or "")[:40]
        n = counts.get(doc_id, 0)
        if n >= per_doc:
            continue
        counts[doc_id] = n + 1
        unique.append(doc)
        if len(unique) >= keep:
            break

    return unique


def _stitch_parts(docs: list) -> list:
    """Merge ascending runs of `chunk_part`s sharing a doc_id+page into one excerpt.

    A part joins the run when its number is one above the last part accepted for that
    doc_id+page, tracked across the whole ranked list, so an unrelated excerpt sitting
    between part 1 and part 2 does not break the run. Reversed (2 then 1) or gapped
    (1 then 3) parts open a new run rather than fabricating text that was never
    retrieved; a new run may start at part 3, so a merged excerpt can begin mid-page
    but never invents a middle.

    A run that ends up holding one excerpt passes through as the caller's own object:
    points with no usable part metadata, whole pages (total_parts <= 1), and a part whose
    siblings were simply never retrieved. Merged runs come back as SimpleDoc, so the
    returned list mixes both types — read `.content` and `.meta`, which every element
    provides, rather than assuming one class.
    """
    runs = []
    open_run = {}              # (doc_id, page) -> run that still accepts the next part

    for doc in docs:
        meta = doc.meta or {}
        part, total = meta.get("chunk_part"), meta.get("total_parts")
        # total <= 1 is the common case: a whole page (median chunk 243 words) has no
        # siblings to merge, so it never needs run tracking.
        if not isinstance(part, int) or not isinstance(total, int) or total <= 1:
            runs.append([doc])
            continue
        key = (meta.get("doc_id", ""), meta.get("page"))
        run = open_run.get(key)
        if run is not None and run[-1].meta["chunk_part"] + 1 == part:
            run.append(doc)
        else:
            runs.append([doc])
            open_run[key] = runs[-1]

    out = []
    for run in runs:
        # Not a fast path, and not for the reason it looks like: _stitch only ever sees a
        # run's 2nd+ part, so dropping this branch would NOT flatten single-part text. What
        # it really costs is the caller's object identity plus a stitched_parts: 1 key on a
        # one-part excerpt — the two things scripts/check_dedup_stitch.py pins by identity.
        if len(run) == 1:
            out.append(run[0])
            continue
        text = run[0].content
        for nxt in run[1:]:
            text = _stitch(text, nxt.content)
        out.append(SimpleDoc(text, {**run[0].meta, "stitched_parts": len(run)}))
    return out
