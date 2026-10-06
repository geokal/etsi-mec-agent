"""Checks for the rank-shaping helpers in etsi_mec_agent.dedup.

    uv run python scripts/check_dedup_stitch.py

No models, no Qdrant: dedup.py is stdlib-only on purpose so this runs in ms.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from etsi_mec_agent.dedup import SimpleDoc, _dedup_docs, _stitch

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


def _docs(specs):
    """specs: list of (doc_id, text) -> SimpleDoc list"""
    return [SimpleDoc(t, {"doc_id": d}) for d, t in specs]


# Baseline behaviour must not change: one excerpt per doc_id, capped at keep.
one = _dedup_docs(_docs([("MEC003", "aa bb cc"), ("MEC003", "dd ee ff"), ("MEC030", "gg")]), keep=5)
assert [d.meta["doc_id"] for d in one] == ["MEC003", "MEC030"], one

# per_doc=2 keeps the continuation of the same document.
two = _dedup_docs(_docs([("MEC003", "aa bb"), ("MEC003", "dd ee"), ("MEC003", "ff gg"),
                         ("MEC030", "hh ii")]), keep=5, per_doc=2)
assert [d.meta["doc_id"] for d in two] == ["MEC003", "MEC003", "MEC030"], two

# keep still wins over per_doc: 4 excerpts of one doc, where per_doc alone would allow 3.
capped = _dedup_docs(_docs([("MEC003", "aa"), ("MEC003", "bb"), ("MEC003", "cc"),
                            ("MEC003", "dd")]), keep=2, per_doc=3)
assert len(capped) == 2, capped

# Identical text under two doc_ids collapses (this is what hides the mislabelled copies).
same = _dedup_docs(_docs([("MEC003", "xx yy zz"), ("MEC070", "xx yy zz")]), keep=5)
assert len(same) == 1, same

# The 400-char fingerprint window, characterised as it behaves today: two excerpts of one
# doc that share a prefix longer than the window collapse in pass 1, so per_doc never gets
# to keep the second one (see the ceiling marked in _dedup_docs).
head = " ".join(f"t{i}" for i in range(120))  # 489 chars, i.e. longer than the window
shared = _dedup_docs(_docs([("MEC003", head + " AAA"), ("MEC003", head + " BBB")]), keep=5, per_doc=2)
assert len(shared) == 1, shared

# No doc_id: the fallback key is the first 40 chars of the text, so two hits sharing that
# prefix but differing later burn a single quota while a distinct text keeps its own.
noid = _dedup_docs([SimpleDoc("F" * 40 + " AAA", {}), SimpleDoc("F" * 40 + " BBB", {}),
                    SimpleDoc("some text", {})], keep=5)
assert [d.content for d in noid] == ["F" * 40 + " AAA", "some text"], noid

# per_doc is clamped at the trust boundary: 0 and negatives behave exactly like 1.
sample = _docs([("MEC003", "aa bb cc"), ("MEC003", "dd ee ff"), ("MEC030", "gg")])
for bad in (0, -5):
    got = _dedup_docs(sample, keep=5, per_doc=bad)
    assert [d.meta["doc_id"] for d in got] == ["MEC003", "MEC030"], (bad, got)

# keep is clamped the same way: 0 means "no results", not "one result" (--top-k 0).
assert _dedup_docs(_docs([("MEC003", "aa")]), keep=0) == []

# Pass 1 strips image refs, then takes the first 400 chars — order matters.
img_a = "![d](MEC003.pdf-0035-02.png)"
img_b = "![d](MEC079.pdf-0035-02.png)"
body = "s" * 380          # 28 + 380 chars in, so an unstripped image path would differ
strip_only = _dedup_docs([SimpleDoc(img_a + body, {"doc_id": "MEC003"}),
                          SimpleDoc(img_b + body, {"doc_id": "MEC070"})], keep=5)
assert len(strip_only) == 1, "differing image paths must not break the match"

# 28 + 372 = exactly 400 chars, so slicing first would hide AAA/BBB and collapse both.
mid = "s" * 372
same_img = _dedup_docs([SimpleDoc(img_a + mid + "AAA", {"doc_id": "MEC003"}),
                        SimpleDoc(img_a + mid + "BBB", {"doc_id": "MEC003"})], keep=5, per_doc=2)
assert len(same_img) == 2, "differing text after the window must keep both excerpts"

print("[check] _dedup_docs: default, per_doc, keep cap, cross-id collapse, 400-char window,")
print("[check]               no-doc_id prefix bucketing, per_doc clamp, keep < 1,")
print("[check]               strip-before-window order — OK")
