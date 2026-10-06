"""Checks for the rank-shaping helpers in etsi_mec_agent.dedup.

    uv run python scripts/check_dedup_stitch.py

No models, no Qdrant: dedup.py is stdlib-only on purpose so this runs in ms.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from etsi_mec_agent.dedup import SimpleDoc, _stitch

# The attribute contract `generate_answer` and `_dedup_docs` will code against.
d = SimpleDoc("t", {"doc_id": "MEC003"})
assert d.content == "t" and d.meta["doc_id"] == "MEC003"

# The 40-word sliding window repeats part of a's tail at the head of b; stitch drops it.
a = " ".join(f"alpha{i}" for i in range(40)) + " TAIL0 TAIL1 TAIL2"
b = "TAIL0 TAIL1 TAIL2 beta0 beta1 beta2"
out = _stitch(a, b)
assert out.count("TAIL1") == 1, f"overlap not trimmed: {out}"
assert out.startswith("alpha0") and out.endswith("beta2"), out

# The real ingest contract: chunk_page_text(max_words=300, overlap=40) emits
# words[:300] and words[260:400], so stitching must give the page back exactly.
page = " ".join(f"w{i}" for i in range(400))
pw = page.split()
part1 = " ".join(pw[:300])
part2 = " ".join(pw[260:400])
assert _stitch(part1, part2).split() == pw, "40-word window not reconstructed exactly"

# No overlap: plain concatenation, nothing invented.
assert _stitch("one two", "three four") == "one two three four"

# b entirely contained in a's tail (degenerate window) -> no duplicate text.
assert _stitch("x y z", "y z") == "x y z"

# A single coincidental shared token is not the 40-word overlap: keep both, drop nothing.
assert _stitch("a b c the", "the d e") == "a b c the the d e"

# Empty inputs just pass the other side through.
assert _stitch("", "x") == "x"
assert _stitch("x", "") == "x"

print("[check] _stitch: ingest 40-word round-trip, trim, concat, degenerate tail,")
print("[check]         single-token kept, empty input; SimpleDoc attr contract — OK")
