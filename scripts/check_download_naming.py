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
from etsi_mec_agent.identity import (canon_spec_id, cover_identity, file_md5, keep_if_new, pdf_title,
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
    got = safe_spec_filename(label, numbered.name, pdf_title(numbered), ident["edition"])
    assert got == "MEC003-V4.1.1-Multi-access-Edge-Computing-(MEC)-Framework-and-Reference.pdf", got
    assert not got[: -len(".pdf")].endswith("-"), got          # truncated on a word boundary

    # The edition is in the name because ETSI reissues a document under the same title. Two editions
    # must land as two files: sharing a name let the newer bytes replace the PDF that the older
    # chunks still name in their payload, and the backfill then read the new cover onto old text.
    older = make_cover(root, "old.pdf", "ETSI GS MEC 003 V3.1.1 (2023-04)",
                       "Multi-access Edge Computing (MEC); Framework and Reference Architecture")
    old_ident = cover_identity(older)
    old_name = safe_spec_filename(label, older.name, pdf_title(older), old_ident["edition"])
    assert old_name != got, (old_name, got)
    assert old_name == "MEC003-V3.1.1-Multi-access-Edge-Computing-(MEC)-Framework-and-Reference.pdf", old_name
    # A document whose cover declares no edition still earns the plain prefix.
    assert safe_spec_filename("MEC003", "x.pdf", "Framework", None) == "MEC003-Framework.pdf"

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

# The downloaders' guard rests on this: a body that merely starts with %PDF raises rather than
# returning None, so an unreadable download is skipped instead of filed under a guessed name.
with tempfile.TemporaryDirectory() as td:
    junk = Path(td) / "junk.pdf"
    junk.write_bytes(b"%PDF" + b"\x00" * 300)
    raised = False
    try:
        cover_identity(junk)
    except Exception:
        raised = True
    assert raised, "a corrupt %PDF body must raise, not return None"

# The cover outranks the URL, and the two spell the same spec differently: the deliver file says
# MEC003 while the cover stamps MEC-003, and a part is MEC-010-02 in the URL but MEC 010-2 on the
# cover. Both downloaders compare on canon_spec_id and refuse a document that denies its own number.
assert canon_spec_id("MEC003") == canon_spec_id("MEC-003") == canon_spec_id("MEC 003")
assert canon_spec_id("MEC-010-02") == canon_spec_id("MEC 010-2")
assert canon_spec_id("MEC-DEC025") == canon_spec_id("MEC-DEC-025")
assert canon_spec_id("MEC040") != canon_spec_id("MEC041")

print("OK: cover numbering, duplicate refusal, dedupe keepers, unreadable bodies and "
      "URL-vs-cover disagreement all behave.")
