"""Rank shaping for retrieval results: stitching passage parts and dedup.

Stdlib-only by design. `search.py` imports fastembed at module level, so anything
that lives here would otherwise drag ONNX runtime into every check script.
"""
import re

_IMG_RE = re.compile(r"!\[.*?\]\(.*?\)")   # markdown image refs vary between copies of the same text


class SimpleDoc:
    """A retrieved excerpt: its text plus the Qdrant payload it came with."""

    def __init__(self, content: str, meta: dict):
        self.content = content
        self.meta = meta


def _stitch(a: str, b: str, max_overlap: int = 60) -> str:
    """Append b to a, dropping the tokens b repeats from the end of a.

    `chunk_page_text` emits chunks with a 40-word overlap and joins words with single
    spaces, so token-level comparison is lossless and the trim can't corrupt markdown.
    """
    aw, bw = a.split(), b.split()
    n = min(len(aw), len(bw), max_overlap)
    for k in range(n, 0, -1):
        if aw[-k:] == bw[:k]:
            return " ".join(aw + bw[k:])
    return " ".join(aw + bw)
