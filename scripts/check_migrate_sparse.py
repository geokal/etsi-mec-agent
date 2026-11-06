"""Model-free check of the sparse-vector text the migration rebuilds.

    uv run python scripts/check_migrate_sparse.py

`fastembed` is stubbed, so importing the migration (which pulls in `search`) loads no ONNX model and
no Qdrant connection is made. The migration copies stored points without re-embedding, so the sparse
vector it rebuilds from the payload has to equal the one ingest produced: ingest tokenises
`spec_id edition clause heading\\nbody` (ingest.py:243-244), while the payload stores only the body
under "text". Reading the body alone — what this did before the fix — drops the cover-derived spec
number from the sparse vector.
"""
import sys
import types
import zlib
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root / "scripts"))

stub = types.ModuleType("fastembed")
stub.TextEmbedding = object
stub.LateInteractionTextEmbedding = object
sys.modules.setdefault("fastembed", stub)

from etsi_mec_agent.embeddings import _token_sparse
from migrate_add_sparse import _embed_text


def tok_id(token: str) -> int:
    return zlib.crc32(token.encode()) & 0x7FFFFFFF


# A stored point: identity in its own fields, body under "text" — the shape ingest.py:322 writes.
payload = {
    "spec_id": "MEC-003", "edition": "V4.1.1", "clause": "5.3",
    "heading": "Functional entities",
    "text": "The UALCMP is a system level functional entity.",
}

# The rebuilt text mirrors ingest's `f"{context}\n{body}"`: identity space-joined, then a newline.
assert _embed_text(payload) == (
    "MEC-003 V4.1.1 5.3 Functional entities\n"
    "The UALCMP is a system level functional entity."
), _embed_text(payload)

# No identity -> ingest's `else pc.text` branch: the body alone, no leading newline.
assert _embed_text({"text": "An unnumbered white paper body."}) == "An unnumbered white paper body."

# A missing field is dropped by filter(None), not rendered as the literal string "None".
partial = {"spec_id": "MEC-016", "edition": "V3.1.1", "heading": "Mx2", "text": "body"}
assert _embed_text(partial) == "MEC-016 V3.1.1 Mx2\nbody", _embed_text(partial)
assert "None" not in _embed_text(partial)

# The crux the review named: the spec number is in the sparse vector now, and was not before.
fixed = _token_sparse(_embed_text(payload))
before = _token_sparse(payload["text"])          # the old body-only input
assert tok_id("mec") in fixed.indices
assert tok_id("003") in fixed.indices
assert tok_id("mec") not in before.indices
assert tok_id("003") not in before.indices

print("check_migrate_sparse: OK (8 assertions)")
