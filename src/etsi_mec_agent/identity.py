"""Spec identity read off the PDF, not off the filename.

Kept separate from `ingest.py` (which imports fastembed at module level) so tooling like
`scripts/backfill_spec_identity.py` can parse covers without dragging ONNX runtime into the
process. Needs pymupdf, which is a C library and allocates no model.
"""

from __future__ import annotations

import hashlib
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


TITLE_STOP = re.compile(r"\b(?:Disclaimer|Keywords|Reference Number|Series|Edition)\b")


def pdf_title(pdf: Path, pages: int = 2) -> str | None:
    """The document title ETSI prints under the spec header line, or None."""
    with pymupdf.open(str(pdf)) as doc:
        head = " ".join(" ".join(doc[i].get_text() for i in range(min(pages, doc.page_count))).split())
    m = HEADER.search(head)
    if not m:
        return None
    rest = head[m.end():]
    stop = TITLE_STOP.search(rest)
    title = (rest[:stop.start()] if stop else rest[:140]).strip(" ,;.-")
    return title or None


def safe_spec_filename(spec_label: str | None, original_name: str, title: str | None) -> str:
    """`MEC003-Framework-and-Reference-Architecture.pdf`, or the name it arrived with.

    A spec number on the cover earns the prefix; white papers, slide decks and drafts that carry
    none keep their own name rather than get a fabricated identity.
    """
    if not spec_label:
        return original_name
    slug = re.sub(r"[^\w ;().-]", "", title or "")
    slug = re.sub(r"[ ;]+", "-", slug).strip("-.")
    if len(slug) > 60:                      # end on a word boundary, not mid-term
        slug = slug[:60].rsplit("-", 1)[0]
    return f"{spec_label}-{slug}.pdf" if slug else f"{spec_label}.pdf"


def file_md5(pdf: Path) -> str:
    return hashlib.md5(pdf.read_bytes()).hexdigest()


def keep_if_new(tmp: Path, dest: Path, specs_dir: Path) -> str | None:
    """Move tmp onto dest unless those bytes already live in specs_dir under another name.

    ETSI serves one document under several filenames, and naming a download after the URL is
    how 106 files came to hold 54 documents. Returns the existing filename when the fresh copy
    was discarded.

    ponytail: re-md5s every PDF per download (~1 s over 120 MB); upgrade path is an md5 index
    stored in the manifest.
    """
    digest = file_md5(tmp)
    for p in sorted(specs_dir.glob("*.pdf")):
        if p != dest and file_md5(p) == digest:
            tmp.unlink(missing_ok=True)
            return p.name
    tmp.replace(dest)
    return None
