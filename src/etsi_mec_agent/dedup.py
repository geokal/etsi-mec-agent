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
    Two-pass deduplication of retrieved hits.

    Pass 1 — text fingerprint (first 400 chars, images stripped): catches the same
    chunk stored under different doc_ids, which is common in this corpus because each
    PDF extraction embeds a different image path in the markdown.

    Pass 2 — up to `per_doc` excerpts per doc_id, `keep` results overall. per_doc=1
    reproduces the historical one-chunk-per-document rule; larger values let a
    document contribute more than its single best-ranked page.
    """
    seen_text: set = set()
    after_text: list = []
    for doc in docs:
        raw = doc.content if hasattr(doc, "content") else (doc.meta or {}).get("text", "")
        fp = _IMG_RE.sub("", raw or "")[:400].strip()
        if fp not in seen_text:
            seen_text.add(fp)
            after_text.append(doc)

    counts: dict = {}
    unique: list = []
    for doc in after_text:
        meta = doc.meta if hasattr(doc, "meta") else {}
        doc_id = (meta or {}).get("doc_id", "") or (doc.content or "")[:40]
        if counts.get(doc_id, 0) >= per_doc:
            continue
        counts[doc_id] = counts.get(doc_id, 0) + 1
        unique.append(doc)
        if len(unique) >= keep:
            break

    return unique
