"""Runnable check for the ingest image quality filter.

    uv run python scripts/check_diagram_filter.py
"""
import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image

from etsi_mec_agent.ingest import _image_is_worth_keeping

with tempfile.TemporaryDirectory() as td:
    def make(name, img):
        p = Path(td) / name
        img.save(p)
        return p

    tiny = make("tiny.png", Image.new("RGB", (9, 44), "white"))
    blank = make("blank.png", Image.new("RGB", (400, 400), (192, 192, 192)))
    real = make("real.png", Image.linear_gradient("L").convert("RGB").resize((400, 300)))
    dup = Path(td) / "dup.png"
    shutil.copyfile(real, dup)

    seen = set()   # one document's worth of digests
    assert not _image_is_worth_keeping(tiny, seen), "sub-200px image should be rejected"
    assert not tiny.exists(), "rejected image should be deleted"
    assert not _image_is_worth_keeping(blank, seen), "flat blank image should be rejected"
    assert not blank.exists(), "rejected image should be deleted"
    assert _image_is_worth_keeping(real, seen), "genuine image should be kept"
    assert real.exists()
    assert not _image_is_worth_keeping(dup, seen), "byte-duplicate image should be rejected"
    assert not dup.exists()

    # ETSI reprints the same reference figure across specs. Those bytes are a duplicate only inside
    # the document that first rendered them; a shared set deleted the reprint and pruned its
    # markdown reference, so the later spec had no diagram left for --diagrams-only to find.
    other_document = set()
    reprint = make("reprint.png", Image.linear_gradient("L").convert("RGB").resize((400, 300)))
    assert hashlib.md5(reprint.read_bytes()).hexdigest() == hashlib.md5(real.read_bytes()).hexdigest()
    assert _image_is_worth_keeping(reprint, other_document), "same bytes in another document must survive"
    assert reprint.exists()

print("OK — tiny/blank/duplicate images filtered correctly")
