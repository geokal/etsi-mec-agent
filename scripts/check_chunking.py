"""Checks for the block-aware page chunker in etsi_mec_agent.chunking.

    uv run python scripts/check_chunking.py

No models, no Qdrant, no PDFs: chunking.py is stdlib-only so this runs in ms. The ingest
contract it pins is B2 of docs/superpowers/specs/2026-10-06-corpus-wide-answering-design.md.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from etsi_mec_agent.chunking import PageChunk, chunk_page, estimate_tokens

# pymupdf4llm emits ETSI tables like this: first row, then the separator, then body rows, and a
# new sub-table introduced by a row whose first cell is a clause number.
TABLE = """#### 7.2 Reference points

##### 7.2.1 Reference points related to the MEC platform

|Mp1:|The Mp1 reference point between the MEC platform and the MEC applications provides<br>service registration.|
|---|---|
|Mp2:|The Mp2 reference point between the MEC platform and the data plane of the Virtualisation<br>infrastructure.|
|Mp3:|The Mp3 reference point between MEC platforms is used for control communication.|

|7.2.2|Reference points related to the MEC management|
|---|---|
|Mm1:|The Mm1 reference point between the MEO and the OSS is used for triggering instantiation.|
|Mm3:|The Mm3 reference point between the MEO and the MEC platform manager is used for the<br>management of the application lifecycle.|
"""

rows = "\n".join(f"|R{n:02d}:|Row {n} of the synthetic table, long enough to force a split.<br>tail|"
                 for n in range(30))
BIG_TABLE = f"|Name:|Description of the column|\n|---|---|\n{rows}\n"
PROSE = " ".join(f"word{i}" for i in range(700))


def cells_are_whole(chunks):
    """No chunk ends inside a cell: every line it holds is a complete markdown row."""
    for c in chunks:
        for line in c.text.splitlines():
            if not line.strip():
                continue
            if line.lstrip().startswith("|") and not line.rstrip().endswith("|"):
                return False, (c.text[:40], line[-60:])
    return True, None


def table_chunk_count(chunks):
    return sum(1 for c in chunks if c.block_kind == "table")


whole, bad = cells_are_whole(chunk_page(BIG_TABLE, max_words=60))
assert whole, f"a row was cut mid-cell: {bad}"
assert table_chunk_count(chunk_page(BIG_TABLE, max_words=60)) > 1, "the 30-row table must split"

# A 30-row table reconstructs completely from its chunks: every row appears, and no row appears
# as a fragment. Headers repeat, so they are dropped before comparing.
pieces = chunk_page(BIG_TABLE, max_words=60)
seen = []
for c in pieces:
    for line in c.text.splitlines():
        s = line.strip()
        if s.startswith("|") and not re.fullmatch(r"\|[\s:|-]+\|", s) and s not in seen:
            seen.append(s)
assert len(seen) == 31, f"expected 30 rows + the column header, got {len(seen)}"
assert all(r in seen for r in
           (f"|R{i:02d}:|Row {i} of the synthetic table, long enough to force a split.<br>tail|"
            for i in range(30))), "a row is missing from the chunks"

# Every table chunk carries its header row, which is what makes the repeated `|Mm3:|...|` cell
# interpretable on its own instead of as an unlabelled fragment.
headered = chunk_page(TABLE, max_words=45)
assert table_chunk_count(headered) >= 2, headered
for c in headered:
    if c.block_kind != "table":
        continue
    body = [ln for ln in c.text.splitlines() if ln.strip().startswith("|")]
    assert re.fullmatch(r"\|[\s:|-]+\|", body[1].strip()), f"no header before the rows: {c.text[:70]}"

# Clause references travel with the block, and a sub-table header re-clauses what follows it —
# once the chunk is past the floor (min_words=20 here forces that; the default is 150).
by_clause = [(c.clause, c.block_kind) for c in chunk_page(TABLE, max_words=45, min_words=20)]
assert any(cl == "7.2.1" for cl, _ in by_clause), by_clause
assert any(cl == "7.2.2" for cl, k in by_clause if k == "table"), by_clause

# The floor itself. A dense annex table re-clauses every few rows; breaking at every one of
# those turned pages into 29-54 parts and left 15-word fragments that rank badly and never
# stitch back. Below the floor, sub-tables share a chunk and it is labelled with its first clause.
confetti = "\n".join(
    f"|{i}.1|Clause {i}.1 of a dense annex table|"
    f"\n|---|---|\n|R{i}:|Row {i} of clause {i}.1, a dozen words or so in it.|"
    for i in range(1, 5)
)
whole = chunk_page(confetti + "\n", max_words=300)
assert len(whole) == 1, f"sub-clauses shattered the table into {len(whole)} chunks"
assert whole[0].clause == "1.1", whole[0].clause
split = chunk_page(confetti + "\n", max_words=300, min_words=10)
assert len(split) == 4, f"the floor is not a knob: {len(split)} chunks"

# Prose keeps the old window (and the 40-word overlap _stitch trims), never a 700-word run.
prose = chunk_page(f"##### 5.1 Intro\n{PROSE}\n", max_words=300, overlap=40)
assert all(len(c.text.split()) <= 380 for c in prose), [len(c.text.split()) for c in prose]
assert all(c.block_kind == "prose" for c in prose), prose
assert prose[0].clause == "5.1", prose[0].clause

# The token bound is the one ColBERT cares about: table punctuation counts.
dense = chunk_page(BIG_TABLE, max_words=60, max_tokens=120)
assert all(estimate_tokens(c.text) <= 120 or c.block_kind == "table" and len(c.text.splitlines()) == 1
           for c in dense), [(estimate_tokens(c.text), c.text[:30]) for c in dense]

# A whole page that fits stays one chunk with its newlines intact — the flattening that produced
# pipe soup is what this replaces.
small = chunk_page("|A:|one|\n|---|---|\n|B:|two|\n")
assert len(small) == 1 and "\n" in small[0].text, small
assert isinstance(small[0], PageChunk)

# ETSI's page furniture alone is not evidence. A page that is only footer/running title/page number
# must yield no chunk at all, while a figure-only page keeps its chunk — that is what --diagrams-only
# searches through.
assert chunk_page("**_ETSI_**") == []
assert chunk_page("**ETSI GS MEC 003 V4.1.1 (2025-05)**\n**15**\n**_ETSI_**") == []
figure = chunk_page("**_ETSI_**\n![](data/diagrams/MEC036.pdf-0034-08.png)\n"
                    "**Figure 6.2.2-4: Configuration option of MEP proxy**")
assert len(figure) == 1 and "![](data/diagrams" in figure[0].text, figure
# Real content keeps its furniture: whole chunks are dropped, chunk text is never edited.
page = chunk_page("**ETSI GS MEC 003 V4.1.1 (2025-05)**\n**15**\n"
                  "MEP stands for Multi-access Edge Platform.\n**_ETSI_**")
assert len(page) == 1 and "_ETSI_" in page[0].text, page

print("[check] chunk_page: rows never cut, headers repeated, 30-row round-trip, clause")
print("[check]             propagation, prose window + overlap, token bound, newlines kept,")
print("[check]             furniture-only pages yield nothing — OK")
