"""Rank shaping for retrieval results: stitching passage parts and dedup.

Stdlib-only by design. `search.py` imports fastembed at module level, so anything
that lives here would otherwise drag ONNX runtime into every check script.
"""
import re
from dataclasses import dataclass

# strip markdown image refs before fingerprinting; consumed once _dedup_docs moves here
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
