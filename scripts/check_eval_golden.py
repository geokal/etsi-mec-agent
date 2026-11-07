"""Model-free check of eval_rag's golden questions and pass/fail logic.

    $env:QDRANT_INDEX="etsi_mec_prototype"; uv run python scripts/check_eval_golden.py

Imports the real eval_rag symbols (fastembed is stubbed, so no ONNX loads and nothing is embedded).
Two things must hold: every question's keywords exist in at least one of its expected specs, and
first_hit_rank treats `docs` as a set — a hit from any listed spec counts, a hit from outside does not.
"""
import sys
import types
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root / "scripts"))

stub = types.ModuleType("fastembed")
stub.TextEmbedding = object
stub.LateInteractionTextEmbedding = object
sys.modules.setdefault("fastembed", stub)

from qdrant_client import models

import eval_rag
from etsi_mec_agent.embeddings import get_provider
from etsi_mec_agent.store import get_qdrant_client

# docs is a set: MEC-011 answers q14 as well as MEC-003, and MEC-036 answers neither.
q14 = next(q for q in eval_rag.QUESTIONS if q["id"] == "q14")
assert eval_rag.first_hit_rank([("MEC-011", "service registration enables discovery")], q14, 5) == 1
assert eval_rag.first_hit_rank([("MEC-036", "service registration enables discovery")], q14, 5) is None
assert eval_rag.first_hit_rank([("MEC-999", "Mm6")],
                               next(q for q in eval_rag.QUESTIONS if q["id"] == "q03"), 5) is None
# docs=None keeps its meaning: any document may satisfy the question.
q12 = next(q for q in eval_rag.QUESTIONS if q["id"] == "q12")
assert eval_rag.first_hit_rank([("whatever-spec", "the MEC orchestrator selects the MEC host")], q12, 5) == 1


# The hybrid and aggregate columns must measure what search_specs actually ships: the same
# candidate pool and the same current-editions filter. Both were missing, so the recall they
# reported was for an unfiltered, under-filled retrieval nobody can run from the CLI.
class _Capture:
    """Records what eval_rag asked Qdrant for, so the budget and the filter are assertable."""

    def __init__(self):
        """Start empty; every query_points call appends the kwargs it was given."""
        self.calls = []

    def query_points(self, **kwargs):
        """Stand in for the client: keep the request, return no points."""
        self.calls.append(kwargs)
        return types.SimpleNamespace(points=[])


cap = _Capture()
q_dense = [0.0] * eval_rag.settings.dense_dim
q_sparse = models.SparseVector(indices=[1], values=[1.0])
q_colbert = [[0.0] * eval_rag.settings.colbert_dim for _ in range(4)]

eval_rag.hits_hybrid(cap, q_dense, q_sparse, q_colbert, 5)
h = cap.calls[-1]
# search_specs' display path at --top-k 5: fetch_k*6 = 30 results off a max(25*4, 100) pool.
assert h["limit"] == h["prefetch"].limit == 30, (h["limit"], h["prefetch"].limit)
assert all(p.filter == eval_rag.build_filter() for p in h["prefetch"].prefetch), h["prefetch"].prefetch
assert h["query_filter"] == eval_rag.build_filter(), h["query_filter"]

eval_rag.hits_aggregate(cap, q_dense, q_sparse, q_colbert, ctx=15, per_doc=2)
a = cap.calls[-1]
# search_specs' answer path at --answer-context 15: prefetch_limit=max(25,45)=45, hybrid pool
# max(45*4,100)=180, top_k=15*6=90 (search.py:491,514 — move those and move these).
assert a["prefetch"].prefetch[0].limit == 180, a["prefetch"].prefetch[0].limit
assert a["limit"] == a["prefetch"].limit == 90, (a["limit"], a["prefetch"].limit)
assert all(p.filter == eval_rag.build_filter() for p in a["prefetch"].prefetch), a["prefetch"].prefetch
assert a["query_filter"] == eval_rag.build_filter(), a["query_filter"]
print("OK: both hybrid columns request search_specs' own pool size and current-editions filter.")


# Both columns must also stitch before they dedup, as search_specs does (search.py:517→528 for the
# hybrid display, :613→615 for the ColBERT path). Part 2 of a page usually outranks part 1 here, so
# a dedup-first column keeps the part WITHOUT the keyword and reports a miss the product does not
# produce: --top-k 5 prints the merged excerpt.
class _SplitPage:
    """One page as two ranked parts, the golden keyword living only in the lower-ranked one."""

    def query_points(self, **kwargs):
        """Return part 2 then part 1 — the arrival order this corpus actually produces."""
        def point(part, text):
            """One stored point of the split page, shaped the way query_points returns it."""
            return types.SimpleNamespace(
                score=1.0 - part * 0.1,
                payload={"text": text, "doc_id": "MEC003.pdf", "spec_id": "MEC-003",
                         "page": 15, "chunk_part": part, "total_parts": 2},
            )
        return types.SimpleNamespace(points=[
            point(2, "Table 6.2.1-1 continues in the next part"),
            point(1, "the Mp1 reference point connects the application to the platform"),
        ])


split = _SplitPage()
q01 = next(q for q in eval_rag.QUESTIONS if q["id"] == "q01")   # docs ["MEC-003"], kw ["Mp1"]
for label, column in (
    ("hybrid", lambda: eval_rag.hits_hybrid(split, q_dense, q_sparse, q_colbert, 5)),
    ("colbert", lambda: eval_rag.hits_colbert(split, q_dense, q_colbert, 5)),
):
    hits = column()
    assert len(hits) == 1, f"{label} column left the page split: {hits}"
    assert eval_rag.first_hit_rank(hits, q01, 5) == 1, f"{label} column lost the keyword: {hits}"
print("OK: both retrieval columns stitch page parts before dedup, as search_specs does.")

# The guard has to fire at eval_rag's own entry point as well: a direct verify_collection_identity
# check still passes if main() drops or delays its call, and the eval would then score a collection
# another provider made and print the result as recall.
class _ForeignCollection:
    """A collection another provider made, recording every read main() attempts past the guard."""

    def __init__(self, metadata):
        """metadata is the foreign identity claim the guard must reject."""
        self.metadata = metadata
        self.scrolled = []
        self.queried = []

    def get_collection(self, name):
        """Hand back the foreign claim, shaped the way the guard reads it."""
        return types.SimpleNamespace(config=types.SimpleNamespace(metadata=self.metadata))

    def scroll(self, **kwargs):
        """Record the corpus read and return nothing — the guard must fire before this."""
        self.scrolled.append(kwargs)
        return [], None

    def query_points(self, **kwargs):
        """Record a query that must never happen past the guard."""
        self.queried.append(kwargs)
        return types.SimpleNamespace(points=[])


foreign = _ForeignCollection(dict(get_provider().identity(), dense_model="Qdrant/bm25"))
held = (eval_rag.get_qdrant_client, sys.argv)
eval_rag.get_qdrant_client = lambda: foreign
sys.argv = ["eval_rag.py", "--top-k", "5"]
try:
    eval_rag.main()
    raise AssertionError("eval_rag.main() accepted a collection another provider made")
except RuntimeError as exc:
    assert "dense_model" in str(exc), exc
finally:
    eval_rag.get_qdrant_client, sys.argv = held
assert foreign.scrolled == [] and foreign.queried == [], (foreign.scrolled, foreign.queried)
print("OK: eval_rag's entry point refuses a foreign collection before it reads or embeds.")

client = get_qdrant_client()
corpus = eval_rag.load_corpus(client)
print(f"collection {eval_rag.settings.qdrant_index!r}: {len(corpus)} specs, "
      f"{sum(len(t.split()) for t in corpus.values()):,} words of current text")

missing = [q["id"] for q in eval_rag.QUESTIONS if not eval_rag.evidence_ok(q, corpus)]
for q in eval_rag.QUESTIONS:
    specs = q["docs"] or []
    where = [s for s in specs if any(k.lower() in corpus.get(s, "") for k in q["kw"])]
    print(f"  {q['id']}  {'ok' if q not in missing else 'EVIDENCE-MISSING'}  "
          f"docs={q['docs'] or 'any'}  responds_from={where or ['any-spec' if not specs else []]}")

assert not missing, f"golden questions with no evidence in their own specs: {missing}"
assert all(q.get("docs", None) is None or all(d in corpus for d in q["docs"]) for q in eval_rag.QUESTIONS), \
    [q["id"] for q in eval_rag.QUESTIONS if q["docs"] and any(d not in corpus for d in q["docs"])]
print("OK: every question is answerable from a spec that is actually in the collection.")
