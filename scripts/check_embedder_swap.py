"""Checks for the embedding provider seam and the collection identity guard.

    uv run python scripts/check_embedder_swap.py

No models, no Qdrant, no corpus, no .env bending: `EMBEDDING_MODE` is pinned before `config` is
imported, `fastembed` is blocked in `sys.modules`, so a provider that touched ONNX here would fail
instead of passing, and the ingest checks build their own blank PDF because `data/specs/` is
gitignored and absent from a clean checkout. What it pins: the mode switch picks the ONNX provider
and names its own limits, cloud mode sends `models.Document` (and needs `QDRANT_URL` plus the
client's `cloud_inference` flag), the identity signature is what the guard compares, sparse stays
client-side crc32 in every mode, and `ensure_collection` stamps that signature at birth.
"""
import dataclasses
import os
import sys
import tempfile
import types
import zlib
from pathlib import Path

os.environ["EMBEDDING_MODE"] = "local"   # pinned first: load_dotenv() does not override the environ
os.environ["QDRANT_URL"] = ""            # a developer's cloud cluster must not make cloud mode
os.environ["QDRANT_API_KEY"] = ""        # constructible here — these tests set it explicitly

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

sys.modules["fastembed"] = None   # any `from fastembed import ...` raises ImportError

from qdrant_client import models

from etsi_mec_agent import embeddings, store
from etsi_mec_agent.config import settings
from etsi_mec_agent.embeddings import CloudInferenceProvider, OnnxProvider, get_provider
from etsi_mec_agent.store import _identity_problem, ensure_collection, verify_collection_identity


class _Vec(list):
    """Stands in for a numpy vector: the seam calls .tolist() on the model's output."""

    def tolist(self):
        """Recursive like numpy's: a ColBERT passage is a 2-D array, and ingest stores plain lists."""
        return [v.tolist() if isinstance(v, _Vec) else v for v in self]


class _FakeDense:
    """Stands in for fastembed's TextEmbedding at the configured dense width."""

    def embed(self, texts):
        """One dense vector per text, the shape the real embed() yields."""
        return [_Vec([0.1] * settings.dense_dim) for _ in texts]


class _FakeColBERT:
    """Stands in for LateInteractionTextEmbedding at the configured ColBERT width."""

    def __init__(self, tokens=4):
        """tokens is the multivector length every call below returns."""
        self.tokens = tokens

    def query_embed(self, text):
        """One multivector for the query, nested the way the real query_embed nests it."""
        return [_Vec([_Vec([0.0] * settings.colbert_dim) for _ in range(self.tokens)])]

    def passage_embed(self, texts):
        """One multivector per passage."""
        return [_Vec([_Vec([0.0] * settings.colbert_dim) for _ in range(self.tokens)]) for _ in texts]


class _FakeClient:
    """Records what the seam asked of Qdrant; never opens a connection."""

    def __init__(self, metadata=None, exists=False):
        """metadata is the identity the collection claims; exists is whether it is there at all."""
        self.metadata = metadata
        self.exists = exists
        self.read_names = []
        self.created = []
        self.updates = []

    def get_collection(self, name):
        """Record the read and hand back a collection carrying the claimed metadata."""
        self.read_names.append(name)
        return types.SimpleNamespace(
            config=types.SimpleNamespace(metadata=self.metadata),
            points_count=0,
        )

    def collection_exists(self, name):
        """Answer from the constructor flag: this gate never creates what it did not ask for."""
        return self.exists

    def create_collection(self, collection_name, **kwargs):
        """Record the create call — the metadata stamped at birth is the assertion."""
        self.created.append((collection_name, kwargs))

    def create_payload_index(self, **kwargs):
        """Accept and ignore: payload indexes are not what this gate asserts."""
        pass

    def update_collection(self, collection_name, metadata=None, **kwargs):
        """Record the label write and adopt it, so a second read sees the new claim."""
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

_real_settings = embeddings.settings


def _settings(**over):
    """Point both modules at a replaced Settings and drop the cached provider."""
    replaced = dataclasses.replace(_real_settings, **over)
    embeddings.settings = store.settings = replaced
    get_provider.cache_clear()
    return replaced


# An unshipped runtime has to fail loudly, not fall back to ONNX against a collection another
# provider made.
_settings(embedding_mode="openai")
try:
    get_provider()
    raise AssertionError("EMBEDDING_MODE=openai must raise, not silently embed locally")
except ValueError as exc:
    assert "openai" in str(exc), exc
_settings()
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
# Point-ready, not model-ready: ingest.py puts these straight into PointStruct.vector, so the
# provider converts (_Vec subclasses list; .tolist() returns a plain one) and a cloud Document
# can take the same slot.
assert type(batch[0]) is list and type(cbatch[0][0]) is list, (type(batch[0]), type(cbatch[0][0]))

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

# --- cloud mode: text out, vectors made server-side ------------------------------------------

_settings(embedding_mode="cloud")
try:
    get_provider()
    raise AssertionError("EMBEDDING_MODE=cloud without QDRANT_URL must raise")
except ValueError as exc:
    assert "QDRANT_URL" in str(exc), exc

cloud_cfg = _settings(embedding_mode="cloud", qdrant_url="https://cluster.example:6333",
                      qdrant_api_key="not-a-real-key",
                      colbert_model="answerdotai/answerai-colbert-small-v1", colbert_dim=96)
cloud = get_provider()
assert isinstance(cloud, CloudInferenceProvider), cloud

cid = cloud.identity()
assert cid["mode"] == "cloud", cid
assert cid["colbert_model"] == cloud_cfg.colbert_model and cid["colbert_dim"] == 96, cid
assert cid["dense_model"] == cloud_cfg.dense_model and cid["dense_dim"] == cloud_cfg.dense_dim, cid
# The guard is what stops a cloud collection being queried by the local provider and vice versa.
assert _identity_problem(cid, identity) and "mode" in _identity_problem(cid, identity), cid

cq = cloud.query_vectors("Mp1 reference point")
assert isinstance(cq["dense"], models.Document) and cq["dense"].model == cloud_cfg.dense_model, cq["dense"]
assert isinstance(cq["colbert"], models.Document), cq["colbert"]
assert cq["colbert"].model == cloud_cfg.colbert_model, cq["colbert"]
assert cq["dense"].text == cq["colbert"].text == "Mp1 reference point", cq
# Sparse stays client-side crc32 even here: server-side BM25 would orphan every stored sparse vector.
assert sorted(cq["sparse"].indices) == want_ids, cq["sparse"].indices
assert [d.model for d in cloud.embed_dense_batch(["a", "b"])] == [cloud_cfg.dense_model] * 2
assert [d.model for d in cloud.embed_colbert_batch(["a", "b"])] == [cloud_cfg.colbert_model] * 2

# The client flag decides who embeds a Document: with cloud_inference off, qdrant-client embeds it
# locally through fastembed instead of sending the text to the cluster.
_real_QdrantClient = store.QdrantClient
built = []
store.QdrantClient = lambda **kw: built.append(kw)
store.get_qdrant_client()
assert built[-1]["cloud_inference"] is True, built[-1]
assert built[-1]["url"] == cloud_cfg.qdrant_url, built[-1]

_settings(embedding_mode="local", qdrant_url="https://cluster.example:6333", qdrant_api_key="k")
store.get_qdrant_client()
assert built[-1]["cloud_inference"] is False, built[-1]   # a raw-vector copy migration stays raw

store.QdrantClient = _real_QdrantClient
embeddings.settings = store.settings = _real_settings
get_provider.cache_clear()
assert isinstance(get_provider(), OnnxProvider)

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

# A raw-vector copy passes the source's claim through: stamping the current provider's would
# label vectors nobody re-embedded as made by whatever mode happens to be configured.
copied_from = dict(identity, dense_model="elsewhere/bge-small-en-v1.5")
copied = _FakeClient(exists=False)
store.get_qdrant_client = lambda: copied
ensure_collection(collection_name="target", metadata=copied_from)
assert copied.created[0][1]["metadata"] == copied_from, copied.created[0][1]["metadata"]

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
assert err is None and client.updates == [(settings.qdrant_index, identity)], (err, client.updates)

client, err = _label(swapped, "--apply")                # another provider's claim: refuse
assert err is not None and "--force" in str(err), err
assert client.updates == [], client.updates

client, err = _label(swapped, "--apply", "--force")     # ...unless the operator vouches for it
assert err is None and client.updates == [(settings.qdrant_index, identity)], (err, client.updates)

client, err = _label(swapped)                           # dry run reports and writes nothing
assert err is None and client.updates == [], (err, client.updates)

# --- the hybrid query nests RRF under a ColBERT rescore -------------------------------------

from etsi_mec_agent import search as search_mod


class _QueryCapture:
    """Records the hybrid query_points request so its nesting and budget are assertable."""

    def __init__(self):
        """Start with no request recorded."""
        self.kwargs = None

    def query_points(self, **kwargs):
        """Keep the request and return no points: only its shape matters here."""
        self.kwargs = kwargs
        return types.SimpleNamespace(points=[])


colbert_q = [[0.0] * settings.colbert_dim for _ in range(4)]
cap = _QueryCapture()
search_mod._run_hybrid_retrieval(
    cap, query_dense=[0.0] * settings.dense_dim,
    query_sparse=models.SparseVector(indices=[1], values=[1.0]),
    query_colbert=colbert_q, top_k=30, prefetch_limit=100,
)
kw = cap.kwargs
inner = kw["prefetch"]
assert isinstance(inner, models.Prefetch) and len(inner.prefetch) == 2, inner
assert {p.using for p in inner.prefetch} == {"dense", "sparse"}, [p.using for p in inner.prefetch]
assert inner.query == models.FusionQuery(fusion=models.Fusion.RRF), inner.query
assert inner.limit == kw["limit"] == 30, (inner.limit, kw["limit"])
assert kw["using"] == "colbert" and kw["query"] == colbert_q, (kw["using"], kw["query"])
assert kw["collection_name"] == settings.qdrant_index, kw["collection_name"]

# --- the entry points enforce the guard, and --recreate-index fails before it wipes ----------

import etsi_mec_agent.ingest as ingest_mod

# data/specs/ is gitignored, so a gate that reads a real PDF cannot run on a clean checkout. Neither
# ingest check below parses the file — one aborts at the identity guard, one at the warm-up — but a
# blank one-pager keeps that true even if an edit ever lets a run reach ingest_single_pdf: it parses
# to a single empty page, so the run ends at "No text extracted" instead of inside pymupdf.
fixture = tempfile.TemporaryDirectory()
pdf = Path(fixture.name) / "MEC001.pdf"
pdf.write_bytes(b"%PDF-1.4\n"
                b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
                b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
                b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>endobj\n"
                b"trailer<</Root 1 0 R>>\n")


class _Leaky(_FakeClient):
    """An existing collection whose vectors another provider made, and that records traffic."""

    def __init__(self, metadata):
        """metadata is the foreign claim; queried/upserted record any traffic past the guard."""
        super().__init__(metadata=metadata, exists=True)
        self.queried = []
        self.upserted = []

    def query_points(self, **kwargs):
        """Record a query that must never happen past the guard."""
        self.queried.append(kwargs)
        return types.SimpleNamespace(points=[])

    def upsert(self, **kwargs):
        """Record a write that must never happen past the guard."""
        self.upserted.append(kwargs)


def _guard_fires(run, label):
    """`run` must abort with the guard's RuntimeError before it embeds, queries or upserts.

    Nothing else here would stop it: get_embedders is faked above and the client below records
    instead of connecting, so an entry point that dropped or delayed its guard runs to completion
    and scores a collection another provider made. Demanding the guard's own message, then an empty
    traffic record, is what pins the abort to the guard.
    """
    leaky = _Leaky(dict(identity, dense_model="Qdrant/bm25"))
    try:
        run(leaky)
        raise AssertionError(f"{label} accepted a collection another provider made")
    except RuntimeError as exc:
        assert "dense_model" in str(exc), exc
    assert leaky.queried == [] and leaky.upserted == [], (label, leaky.queried, leaky.upserted)


def _search(leaky):
    """search_specs pointed at a collection the current provider did not make."""
    held = search_mod.get_qdrant_client
    search_mod.get_qdrant_client = lambda: leaky
    try:
        search_mod.search_specs("Mp1 reference point", top_k=5, use_bm25=True)
    finally:
        search_mod.get_qdrant_client = held


def _ingest(leaky):
    """ingest_pdfs without --recreate-index, pointed at the same mismatched collection."""
    held = (ingest_mod.get_qdrant_client, ingest_mod.ensure_collection)
    ingest_mod.get_qdrant_client = lambda: leaky
    ingest_mod.ensure_collection = lambda **kw: None
    try:
        ingest_mod.ingest_pdfs([pdf])
    finally:
        ingest_mod.get_qdrant_client, ingest_mod.ensure_collection = held


_guard_fires(_search, "search_specs")
_guard_fires(_ingest, "ingest_pdfs")

# The warm-up must precede the wipe: the ONNX models construct lazily, so a load failure used to
# surface inside the first batch's except with the collection already emptied by --recreate-index.
order = []


class _WillNotLoad:
    """A provider whose model construction fails, after recording that it was asked."""

    def embed_dense_batch(self, texts):
        """Record the warm-up call, then fail the way a missing ONNX model does."""
        order.append("warm")
        raise RuntimeError("the model would not load")


held = (ingest_mod.get_provider, ingest_mod.ensure_collection, ingest_mod.get_qdrant_client)
ingest_mod.get_provider = lambda: _WillNotLoad()
ingest_mod.ensure_collection = lambda **kw: order.append("recreate")
ingest_mod.get_qdrant_client = lambda: _FakeClient(metadata=dict(identity), exists=True)
try:
    ingest_mod.ingest_pdfs([pdf], recreate_index=True)
    raise AssertionError("a model that will not load must stop the ingest")
except RuntimeError as exc:
    assert "would not load" in str(exc), exc
finally:
    ingest_mod.get_provider, ingest_mod.ensure_collection, ingest_mod.get_qdrant_client = held
assert order == ["warm"], f"--recreate-index wiped before the embedders proved they load: {order}"

# The same failure without --recreate-index: there is no wipe to precede, so the warm-up is the only
# thing between a provider that will not load and a batch loop that swallows the error per chunk,
# prints [SKIPPED] for each one, and exits 0 having indexed nothing. It must still come *after* the
# identity guard — _guard_fires(_ingest, ...) above proves that by aborting before any embed.
order.clear()
held = (ingest_mod.get_provider, ingest_mod.ensure_collection, ingest_mod.get_qdrant_client)
ingest_mod.get_provider = lambda: _WillNotLoad()
ingest_mod.ensure_collection = lambda **kw: order.append(("ensure", kw.get("recreate", False)))
ingest_mod.get_qdrant_client = lambda: _FakeClient(metadata=dict(identity), exists=True)
try:
    ingest_mod.ingest_pdfs([pdf])
    raise AssertionError("an incremental ingest must fail when the provider will not load")
except RuntimeError as exc:
    assert "would not load" in str(exc), exc
finally:
    ingest_mod.get_provider, ingest_mod.ensure_collection, ingest_mod.get_qdrant_client = held
assert order == [("ensure", False), "warm"], f"incremental run never warmed the embedders: {order}"

# --- cloud mode must prove inference before --recreate-index wipes anything ---------------------
# In cloud mode embed_dense_batch only builds a Document, so the warm-up above cannot fail: the
# cluster resolves the model on the first upsert, which the wipe has already preceded. A model id
# the cluster does not serve must abort while the real collection is still intact, must exercise
# BOTH models, and must not leave the throwaway probe collection behind.
cloud_order = []


class _CloudDocuments:
    """Batch calls return real Documents without touching the network, as the real provider does."""

    def embed_dense_batch(self, texts):
        """Record the call and hand back the point-ready value the cluster would resolve."""
        cloud_order.append("dense")
        return [models.Document(text=t, model=settings.dense_model) for t in texts]

    def embed_colbert_batch(self, texts):
        """Record the call and hand back the point-ready value the cluster would resolve."""
        cloud_order.append("colbert")
        return [models.Document(text=t, model=settings.colbert_model) for t in texts]


class _NoInference(_FakeClient):
    """A cluster that accepts the probe collection but rejects the point: model not served."""

    def __init__(self):
        """The real collection exists and claims cloud mode, so the identity guard lets the run past."""
        super().__init__(metadata=dict(identity, mode="cloud"), exists=True)
        self.deleted = []

    def upsert(self, **kwargs):
        """Fail the way an unserved model does, after recording which collection was written."""
        cloud_order.append(("upsert", kwargs["collection_name"]))
        raise RuntimeError("model is not available for inference")

    def delete_collection(self, name):
        """Record the cleanup, so a leaked probe collection is visible."""
        self.deleted.append(name)
        cloud_order.append(("delete", name))


probe_name = settings.qdrant_index + "__preflight"
cluster = _NoInference()
held = (ingest_mod.get_provider, ingest_mod.ensure_collection,
        ingest_mod.get_qdrant_client, ingest_mod.settings)
ingest_mod.get_provider = lambda: _CloudDocuments()
ingest_mod.ensure_collection = lambda **kw: cloud_order.append(
    ("ensure", kw.get("collection_name"), kw.get("recreate", False)))
ingest_mod.get_qdrant_client = lambda: cluster
ingest_mod.settings = _settings(embedding_mode="cloud", qdrant_url="https://example.invalid")
try:
    ingest_mod.ingest_pdfs([pdf], recreate_index=True)
    raise AssertionError("an unserved cloud model must stop the ingest before the wipe")
except RuntimeError as exc:
    assert "not available for inference" in str(exc), exc
finally:
    (ingest_mod.get_provider, ingest_mod.ensure_collection,
     ingest_mod.get_qdrant_client, ingest_mod.settings) = held
    _settings()

assert ("ensure", None, True) not in cloud_order, \
    f"--recreate-index wiped the real collection before inference was proved: {cloud_order}"
assert ("upsert", probe_name) in cloud_order, f"the probe never reached the cluster: {cloud_order}"
assert "colbert" in cloud_order, f"the probe exercised only the dense model: {cloud_order}"
assert cluster.deleted == [probe_name], f"the probe collection leaked: {cluster.deleted}"

fixture.cleanup()

print("[check] provider seam: EMBEDDING_MODE=local -> OnnxProvider, cached per process,")
print("[check]   unsupported mode and cloud-without-QDRANT_URL raise ValueError,")
print("[check]   identity signature complete, query/batch shapes point-ready,")
print("[check]   cloud -> Documents carrying the configured ids + cloud_inference=True client,")
print("[check]   sparse stays client-side crc32 in both modes,")
print("[check]   guard silent on match + names the swapped key and the label script otherwise,")
print("[check]   ensure_collection stamps metadata at birth, never restamps, and honours a")
print("[check]   copied source's claim instead of the current provider's,")
print("[check]   label script stamps an unlabeled collection and refuses to overwrite")
print("[check]   a different claim without --force,")
print("[check]   hybrid nests dense+sparse under RRF and rescores the pool with ColBERT,")
print("[check]   search_specs and ingest_pdfs both refuse a mismatched collection before")
print("[check]   embedding, and a recreate and an incremental ingest alike prove the models")
print("[check]   load before a chunk is at risk,")
print("[check]   cloud --recreate-index probes BOTH models through a throwaway collection and")
print("[check]   cleans it up, so an unserved model id aborts before the wipe — OK")
