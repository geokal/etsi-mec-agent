"""Model-free check of the sparse-vector text the migration rebuilds.

    uv run python scripts/check_migrate_sparse.py

No ONNX model loads and no Qdrant connection is made: the sparse tokenizer imports without
fastembed now that embeddings.py loads it lazily. The migration copies stored points without
re-embedding, so the sparse vector it rebuilds from the payload has to equal the one ingest
produced: ingest tokenises `spec_id edition clause heading\\nbody` (ingest.py:243-244), while
the payload stores only the body under "text". Reading the body alone — what this did before
the fix — drops the cover-derived spec number from the sparse vector.

It also drives the migration's main() end to end against a fake client: the copy must inherit the
SOURCE's embedding identity (it re-embeds nothing), and a source that records no identity must
abort instead of being stamped with whatever provider happens to be configured.
"""
import sys
import types
import zlib
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root / "scripts"))

from etsi_mec_agent.config import settings
from etsi_mec_agent.embeddings import _token_sparse
import migrate_add_sparse
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

# --- the copy inherits the SOURCE's identity claim, and refuses an unlabelled source ----------
# A migration copies vectors it did not make, so stamping the current provider's identity would
# label them with whatever EMBEDDING_MODE happens to be set — the exact lie the guard catches.
class _FakeCollection:
    """The identity a collection claims, plus the point count the migration verifies against."""

    def __init__(self, metadata):
        """metadata is the claim; zero points means the copy loop has nothing to do."""
        self.config = types.SimpleNamespace(metadata=metadata)
        self.points_count = 0


class _FakeMigrateClient:
    """Stands in for Qdrant: the aliased collection itself, empty, so nothing is scrolled."""

    def __init__(self, metadata):
        """One empty source behind the alias, plus a record of how the switch happened."""
        self.source = _FakeCollection(metadata)
        self.aliases_moved = []
        self.deleted = []

    def get_aliases(self):
        """No aliases, so the name resolves to itself and source == alias."""
        return types.SimpleNamespace(aliases=[])

    def collection_exists(self, name):
        """Always False: --target is accepted as a fresh collection."""
        return False

    def get_collection(self, name):
        """Hand back the fake source whatever name is asked for."""
        return self.source

    def scroll(self, **kwargs):
        """No points and no offset, so the copy loop ends on its first pass."""
        return [], None

    def update_collection_aliases(self, change_aliases_operations):
        """Record the alias move that finishes the migration."""
        self.aliases_moved.append(change_aliases_operations)

    def delete_collection(self, name):
        """Record the source collection being dropped."""
        self.deleted.append(name)


def _migrate(metadata, *argv):
    """Run the migration's main() against a fake source; returns (client, created, SystemExit)."""
    client = _FakeMigrateClient(metadata)
    created = []
    held = (migrate_add_sparse.get_qdrant_client, migrate_add_sparse.ensure_collection, sys.argv)
    migrate_add_sparse.get_qdrant_client = lambda: client
    migrate_add_sparse.ensure_collection = lambda **kw: created.append(kw)
    sys.argv = ["migrate_add_sparse.py", *argv]
    try:
        migrate_add_sparse.main()
        return client, created, None
    except SystemExit as exc:
        return client, created, exc
    finally:
        (migrate_add_sparse.get_qdrant_client,
         migrate_add_sparse.ensure_collection, sys.argv) = held


made_by = {"mode": "cloud", "dense_model": "intfloat/multilingual-e5-small", "dense_dim": 384}
client, created, err = _migrate(made_by, "--target", "fresh_copy")
assert err is None, err
assert created == [{"collection_name": "fresh_copy", "metadata": made_by}], created
assert client.deleted == [settings.qdrant_index] and client.aliases_moved, client.deleted

client, created, err = _migrate({}, "--target", "fresh_copy")
assert err is not None and "label_collection_identity" in str(err), err
assert created == [], "an unlabelled source must not be copied at all"

print("check_migrate_sparse: OK (13 assertions)")
