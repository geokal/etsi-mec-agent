"""Runnable check for the downloader identity rules — no models, no network.

    uv run python scripts/check_download_naming.py

Covers the three things that must not regress: a cover-declared spec number earns a
`MECxxx-<title>.pdf` name while an unnumbered document keeps its own; keep_if_new refuses to save
bytes that already exist under another name; and dedupe_spec_pdfs never keeps a file the index
does not point at.
"""
import sys
import tempfile
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from dedupe_spec_pdfs import keepers
from etsi_mec_agent.identity import (cover_identity, file_md5, keep_if_new, pdf_title,
                                     safe_spec_filename)


def make_cover(dir: Path, name: str, header: str, title: str) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 90), header)
    page.insert_text((50, 130), title)
    path = dir / name
    doc.save(str(path))
    doc.close()
    return path


with tempfile.TemporaryDirectory() as td:
    root = Path(td)

    numbered = make_cover(root, "arrived.pdf", "ETSI GS MEC 003 V4.1.1 (2025-05)",
                          "Multi-access Edge Computing (MEC); Framework and Reference Architecture")
    ident = cover_identity(numbered)
    assert ident and ident["spec_id"] == "MEC-003" and ident["edition"] == "V4.1.1", ident
    label = ident["spec_id"].replace("MEC-DEC-", "MEC-DEC").replace("MEC-", "MEC", 1)
    got = safe_spec_filename(label, numbered.name, pdf_title(numbered))
    assert got == "MEC003-Multi-access-Edge-Computing-(MEC)-Framework-and-Reference.pdf", got
    assert not got[: -len(".pdf")].endswith("-"), got          # truncated on a word boundary

    unnumbered = make_cover(root, "public-overview.pdf", "MEC Public Overview", "Slides, no header line")
    assert cover_identity(unnumbered) is None
    assert safe_spec_filename(None, unnumbered.name, pdf_title(unnumbered)) == unnumbered.name

    # keep_if_new: identical bytes are refused, new bytes land on the requested name.
    other = root / "MEC051.pdf"
    other.write_bytes(b"samesame")
    tmp = root / "a.part"; tmp.write_bytes(b"samesame")
    assert keep_if_new(tmp, root / "MEC099.pdf", root) == "MEC051.pdf"
    assert not tmp.exists() and not (root / "MEC099.pdf").exists()
    tmp.write_bytes(b"different")
    assert keep_if_new(tmp, root / "MEC100.pdf", root) is None
    assert (root / "MEC100.pdf").read_bytes() == b"different" and not tmp.exists()

    # keepers: the indexed copy wins even when another name would look tidier.
    dup_a = make_cover(root, "gs_MEC051v010101p.pdf", "MEC Public Overview", "slides, no header line")
    dup_b = root / "MEC052.pdf"
    dup_b.write_bytes(dup_a.read_bytes())
    assert keepers([dup_a, dup_b], {"MEC052.pdf"}) == [dup_b]
    assert keepers([dup_a, dup_b], set()) == [dup_b]        # nothing indexed: shortest name wins
    assert file_md5(dup_a) == file_md5(dup_b)

print("OK: cover numbering, duplicate refusal and dedupe keepers all behave.")
