"""Checks for the rank-shaping helpers in etsi_mec_agent.dedup.

    uv run python scripts/check_dedup_stitch.py

No models, no Qdrant: dedup.py is stdlib-only on purpose so this runs in ms.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from etsi_mec_agent.dedup import SimpleDoc, _stitch

# The 40-word sliding window repeats part of a's tail at the head of b; stitch drops it.
a = " ".join(f"alpha{i}" for i in range(40)) + " TAIL0 TAIL1 TAIL2"
b = "TAIL0 TAIL1 TAIL2 beta0 beta1 beta2"
out = _stitch(a, b)
assert out.count("TAIL1") == 1, f"overlap not trimmed: {out}"
assert out.startswith("alpha0") and out.endswith("beta2"), out

# No overlap: plain concatenation, nothing invented.
assert _stitch("one two", "three four") == "one two three four"

# b entirely contained in a's tail (degenerate window) -> no duplicate text.
assert _stitch("x y z", "y z") == "x y z"

print("[check] _stitch: overlap trim, concat, degenerate tail — OK")
