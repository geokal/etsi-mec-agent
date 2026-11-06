"""Checks for the embedding provider seam and the collection identity guard.

    uv run python scripts/check_embedder_swap.py

No models, no Qdrant, no .env bending: `EMBEDDING_MODE` is pinned before `config` is imported and
`fastembed` is blocked in `sys.modules`, so a provider that touched ONNX here would fail instead of
passing. What it pins: the mode switch picks the ONNX provider and names its own limits, the
identity signature is what the guard compares, sparse stays client-side crc32 in every mode, and
`ensure_collection` stamps that signature at birth.
"""
import dataclasses
import os
import sys
import types
import zlib
from pathlib import Path

os.environ["EMBEDDING_MODE"] = "local"   # pinned first: load_dotenv() does not override the environ

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

sys.modules["fastembed"] = None   # any `from fastembed import ...` raises ImportError

from qdrant_client import models

from etsi_mec_agent import embeddings, store
from etsi_mec_agent.config import settings
from etsi_mec_agent.embeddings import OnnxProvider, get_provider
from etsi_mec_agent.store import _identity_problem, ensure_collection, verify_collection_identity


class _Vec(list):
    """Stands in for a numpy vector: the seam calls .tolist() on the model's output."""

    def tolist(self):
        return list(self)


class _FakeDense:
    def embed(self, texts):
        return [_Vec([0.1] * settings.dense_dim) for _ in texts]


class _FakeColBERT:
    def __init__(self, tokens=4):
        self.tokens = tokens

    def query_embed(self, text):
        return [_Vec([_Vec([0.0] * settings.colbert_dim) for _ in range(self.tokens)])]

    def passage_embed(self, texts):
        return [_Vec([_Vec([0.0] * settings.colbert_dim) for _ in range(self.tokens)]) for _ in texts]


class _FakeClient:
    """Records what the seam asked of Qdrant; never opens a connection."""

    def __init__(self, metadata=None, exists=False):
        self.metadata = metadata
        self.exists = exists
        self.read_names = []
        self.created = []
        self.updates = []

    def get_collection(self, name):
        self.read_names.append(name)
        return types.SimpleNamespace(
            config=types.SimpleNamespace(metadata=self.metadata),
            points_count=0,
        )

    def collection_exists(self, name):
        return self.exists

    def create_collection(self, collection_name, **kwargs):
        self.created.append((collection_name, kwargs))

    def create_payload_index(self, **kwargs):
        pass

    def update_collection(self, collection_name, metadata=None, **kwargs):
        self.updates.append((collection_name, metadata))
        self.metadata = metadata


# --- the mode switch -------------------------------------------------------------------------

provider = get_provider()
assert isinstance(provider, OnnxProvider), provider
assert get_provider() is provider, "provider must be one instance per process (Streamlit reruns)"

identity = provider.identity()
assert identity == {
    "mode": "local",
    "dense_model": settings.dense_model,
    "dense_dim": settings.dense_dim,
    "colbert_model": settings.colbert_model,
    "colbert_dim": settings.colbert_dim,
    "sparse": "crc32-term-frequency",
}, identity

# An unshipped runtime has to fail loudly, not fall back to ONNX against a cloud-made collection.
_real_settings = embeddings.settings
embeddings.settings = dataclasses.replace(_real_settings, embedding_mode="cloud")
get_provider.cache_clear()
try:
    get_provider()
    raise AssertionError("EMBEDDING_MODE=cloud must raise, not silently embed locally")
except ValueError as exc:
    assert "cloud" in str(exc), exc
finally:
    embeddings.settings = _real_settings
    get_provider.cache_clear()
assert isinstance(get_provider(), OnnxProvider)

# --- query vectors, with the models faked out ------------------------------------------------

embeddings.get_embedders = lambda: (_FakeDense(), _FakeColBERT())
qv = provider.query_vectors("Mp1 reference point")
assert set(qv) == {"dense", "colbert", "sparse"}, qv.keys()
assert len(qv["dense"]) == settings.dense_dim, len(qv["dense"])
assert len(qv["colbert"]) == 4 and all(len(t) == settings.colbert_dim for t in qv["colbert"]), qv["colbert"]

# Sparse is crc32 term frequency computed here, never by the embedder: a provider swap changes the
# dense/colbert space but must not change the token-id space the stored vectors live in.
want_ids = sorted({zlib.crc32(t.encode()) & 0x7FFFFFFF for t in ["mp1", "reference", "point"]})
assert isinstance(qv["sparse"], models.SparseVector), qv["sparse"]
assert sorted(qv["sparse"].indices) == want_ids, qv["sparse"].indices
assert len(qv["sparse"].values) == len(want_ids), qv["sparse"].values

batch = provider.embed_dense_batch(["a", "b"])
assert len(batch) == 2 and len(batch[0]) == settings.dense_dim, batch
cbatch = provider.embed_colbert_batch(["a", "b"])
assert len(cbatch) == 2 and all(len(t) == settings.colbert_dim for t in cbatch[0]), cbatch

# --- the identity guard ---------------------------------------------------------------------

assert _identity_problem(None, identity) is not None
assert "label_collection_identity.py" in _identity_problem(None, identity), _identity_problem(None, identity)
assert _identity_problem(dict(identity), identity) is None

swapped = dict(identity, dense_model="Qdrant/bm25")
problem = _identity_problem(swapped, identity)
assert problem and "dense_model" in problem, problem
assert "Qdrant/bm25" in problem and settings.dense_model in problem, problem

ok = _FakeClient(metadata=dict(identity))
verify_collection_identity(client=ok)                       # matching: silent
assert ok.read_names == [settings.qdrant_index], ok.read_names   # one read, of the configured name

for bad, needle in ((None, "label_collection_identity.py"),
                    ({}, "label_collection_identity.py"),   # an unlabeled collection reads back {}
                    (swapped, "dense_model")):
    try:
        verify_collection_identity(client=_FakeClient(metadata=bad))
        raise AssertionError(f"guard accepted metadata={bad!r}")
    except RuntimeError as exc:
        assert needle in str(exc), exc

# --- collections are self-describing at birth ------------------------------------------------

fresh = _FakeClient(exists=False)
store.get_qdrant_client = lambda: fresh
ensure_collection(collection_name="probe")
assert len(fresh.created) == 1, fresh.created
name, kwargs = fresh.created[0]
assert name == "probe", name
assert kwargs["metadata"] == identity, kwargs.get("metadata")
assert set(kwargs["vectors_config"]) == {"dense", "colbert"}, kwargs["vectors_config"]
assert set(kwargs["sparse_vectors_config"]) == {"sparse"}, kwargs["sparse_vectors_config"]

existing = _FakeClient(exists=True)
store.get_qdrant_client = lambda: existing
ensure_collection(collection_name="probe")
assert existing.created == [], "an existing collection must not be recreated or restamped"

# --- the label script will not overwrite a different claim -----------------------------------

sys.path.insert(0, str(root / "scripts"))
import label_collection_identity as lci

lci.resolve_collection = lambda client, name: name
lci.get_provider = lambda: types.SimpleNamespace(identity=lambda: dict(identity))


def _label(metadata, *argv):
    """Run the label script's main() against a fake collection; returns (client, SystemExit|None)."""
    client = _FakeClient(metadata=metadata)
    lci.get_qdrant_client = lambda: client
    held = sys.argv
    sys.argv = ["label_collection_identity.py", *argv]
    try:
        lci.main()
        return client, None
    except SystemExit as exc:
        return client, exc
    finally:
        sys.argv = held


client, err = _label({}, "--apply")                     # unlabeled: stamping is the whole point
assert err is None and client.updates == [("etsi_mec_specs", identity)], (err, client.updates)

client, err = _label(swapped, "--apply")                # another provider's claim: refuse
assert err is not None and "--force" in str(err), err
assert client.updates == [], client.updates

client, err = _label(swapped, "--apply", "--force")     # ...unless the operator vouches for it
assert err is None and client.updates == [("etsi_mec_specs", identity)], (err, client.updates)

client, err = _label(swapped)                           # dry run reports and writes nothing
assert err is None and client.updates == [], (err, client.updates)

print("[check] provider seam: EMBEDDING_MODE=local -> OnnxProvider, cached per process,")
print("[check]   unsupported mode raises ValueError, identity signature complete,")
print("[check]   query/batch shapes, sparse stays client-side crc32,")
print("[check]   guard silent on match + names the swapped key and the label script otherwise,")
print("[check]   ensure_collection stamps metadata at birth and never restamps,")
print("[check]   label script stamps an unlabeled collection and refuses to overwrite")
print("[check]   a different claim without --force — OK")
