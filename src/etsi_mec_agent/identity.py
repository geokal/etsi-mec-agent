"""Spec identity read off the PDF, not off the filename.

Kept separate from `ingest.py` (which imports fastembed at module level) so tooling like
`scripts/backfill_spec_identity.py` can parse covers without dragging ONNX runtime into the
process. Needs pymupdf, which is a C library and allocates no model.
"""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf

HEADER = re.compile(
    r"ETSI\s+(?:GS|GR|TS|ISG)\s+(MEC(?:-DEC)?\s+\d+(?:-\d+)?)\s+"
    r"V(\d+)\.(\d+)\.(\d+)\s*\((\d{4})-(\d{2})\)"
)


def cover_identity(pdf: Path, pages: int = 3) -> dict | None:
    """{spec_id, edition, pub_date, key} from the cover pages, None if unnumbered.

    The title line repeats on the cover and in the running header, so the first pages are
    enough; slide decks and white papers carry no such line and stay unidentified.
    """
    with pymupdf.open(str(pdf)) as doc:
        head = " ".join(" ".join(doc[i].get_text() for i in range(min(pages, doc.page_count))).split())
    m = HEADER.search(head)
    if not m:
        return None
    edition = f"V{m.group(2)}.{m.group(3)}.{m.group(4)}"
    return {
        "spec_id": re.sub(r"\s+", "-", m.group(1)),
        "edition": edition,
        "pub_date": f"{m.group(5)}-{m.group(6)}",
        "key": tuple(int(p) for p in edition[1:].split(".")),
    }


def stamp(pdf: Path) -> dict:
    """Payload fields every chunk of this PDF carries.

    Unnumbered documents keep the filename as their identity and a null date rather than a
    guessed one. `is_current` starts True for everything: which edition a spec number is on
    is only knowable once the whole corpus is in, so `scripts/backfill_spec_identity.py`
    re-decides it as a post-ingest pass.
    """
    ident = cover_identity(pdf)
    if ident is None:
        return {"spec_id": pdf.stem, "edition": None, "pub_date": None, "is_current": True}
    return {
        "spec_id": ident["spec_id"],
        "edition": ident["edition"],
        "pub_date": ident["pub_date"],
        "is_current": True,
    }
