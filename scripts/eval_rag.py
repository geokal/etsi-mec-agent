"""Golden-question retrieval eval for the ETSI MEC pipeline.

    uv run python scripts/eval_rag.py [--top-k 5]

Exercises the two production retrieval paths from etsi_mec_agent.search:
  colbert — dense prefetch + ColBERT MaxSim rescore (search_specs path 2)
  hybrid  — dense prefetch + BM25 re-rank + RRF (_run_hybrid_retrieval)

A question passes when a top-k chunk from the expected document contains
one of its expected keywords. Questions whose keywords exist nowhere in
the expected document are flagged EVIDENCE-MISSING (bad golden data, not
a retrieval failure) so the numbers stay honest.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qdrant_client import models

from etsi_mec_agent.config import settings
from etsi_mec_agent.search import _dedup_docs, _run_hybrid_retrieval, get_embedders
from etsi_mec_agent.store import get_qdrant_client

# doc=None means any document may satisfy the question.
QUESTIONS = [
    {"id": "q01", "query": "Which reference point connects the MEC application to the MEC platform?", "doc": "MEC003", "kw": ["Mp1"]},
    {"id": "q02", "query": "Which reference point connects the MEC platform to the MEC orchestrator?", "doc": "MEC003", "kw": ["Mm3"]},
    {"id": "q03", "query": "What is the Mm6 reference point used for?", "doc": "MEC003", "kw": ["Mm6"]},
    {"id": "q04", "query": "Which service exposes radio network information to applications?", "doc": None, "kw": ["RNIS", "Radio Network Information"]},
    {"id": "q05", "query": "What is the Edge Enabler Client (EEC)?", "doc": None, "kw": ["Edge Enabler Client", "EEC"]},
    {"id": "q06", "query": "What traffic influence rules does the TCR expose?", "doc": None, "kw": ["traffic influence", "TCR"]},
    {"id": "q07", "query": "Which service continuity modes are defined for MEC applications?", "doc": None, "kw": ["service continuity"]},
    {"id": "q08", "query": "What is the User app LCM proxy?", "doc": "MEC003", "kw": ["User app LCM proxy", "Ulv"]},
    {"id": "q09", "query": "How does a road tunnel affect TCP congestion control in the MEC use case?", "doc": "MEC002", "kw": ["road tunnel", "TCP"]},
    {"id": "q10", "query": "What are the components of the MEC host level reference architecture?", "doc": "MEC003", "kw": ["MEC host", "Virtualisation"]},
    {"id": "q11", "query": "What application lifecycle states are defined?", "doc": None, "kw": ["lifecycle states", "LIFE_CYCLE"]},
    {"id": "q12", "query": "Which API is used to select a MEC system for application instantiation?", "doc": None, "kw": ["Mm5", "Mm6"]},
    {"id": "q13", "query": "How does DNS resolution steer users to the closest MEC server?", "doc": None, "kw": ["DNS"]},
    {"id": "q14", "query": "What is the role of the MEC platform in service discovery?", "doc": "MEC003", "kw": ["service registration", "discovery"]},
    {"id": "q15", "query": "Which interface carries the mpInfoService between platform and orchestrator?", "doc": None, "kw": ["mpInfoService", "Mm3"]},
    {"id": "q16", "query": "What is a MEC service and how does it relate to a MEC application?", "doc": "MEC003", "kw": ["MEC service"]},
]


def load_corpus(client):
    """doc_id -> lowercased full text, for the evidence pre-check."""
    corpus: dict = {}
    offset = None
    while True:
        pts, offset = client.scroll(
            collection_name=settings.qdrant_index, limit=500,
            with_payload=["doc_id", "text"], with_vectors=False, offset=offset,
        )
        for p in pts:
            doc = p.payload.get("doc_id", "")
            corpus[doc] = corpus.get(doc, "") + "\n" + p.payload.get("text", "").lower()
        if offset is None:
            break
    return corpus


def evidence_ok(q, corpus):
    docs = [q["doc"]] if q["doc"] else list(corpus)
    return any(any(k.lower() in corpus.get(d, "") for k in q["kw"]) for d in docs)


def hits_colbert(client, q_dense, q_colbert, k):
    res = client.query_points(
        collection_name=settings.qdrant_index,
        prefetch=models.Prefetch(query=q_dense, using="dense", limit=25),
        query=q_colbert, using="colbert", limit=k,  # exactly what search_specs requests
    )

    class _D:  # match the shape search_specs wraps hits in so _dedup_docs applies identically
        pass

    docs = []
    for h in res.points:
        d = _D()
        d.content = h.payload.get("text", "")
        d.meta = h.payload
        docs.append(d)
    return [(d.meta.get("doc_id", ""), d.content) for d in _dedup_docs(docs, keep=k)]


def hits_hybrid(client, q_text, q_dense, k):
    raw = _run_hybrid_retrieval(client, q_text, q_dense, top_k=k * 6, prefetch_limit=100)  # defaults from search_specs
    raw = _dedup_docs(raw, keep=k)  # same post-fusion dedup as search_specs
    return [(d.meta.get("doc_id", ""), d.content or "") for d in raw]


def first_hit_rank(hits, q, k):
    for rank, (doc, text) in enumerate(hits[:k], 1):
        if q["doc"] and doc != q["doc"]:
            continue
        if any(kw.lower() in text.lower() for kw in q["kw"]):
            return rank
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top-k", type=int, default=5)
    k = ap.parse_args().top_k

    client = get_qdrant_client()
    corpus = load_corpus(client)
    dense_model, colbert_model = get_embedders()

    modes = {"colbert": [], "hybrid": []}
    print(f"{'id':4} {'evidence':9} {'colbert':8} {'hybrid':8} query")
    for q in QUESTIONS:
        if not evidence_ok(q, corpus):
            print(f"{q['id']:4} {'MISSING':9} {'-':8} {'-':8} {q['query'][:50]}")
            continue
        q_dense = list(dense_model.embed([q["query"]]))[0].tolist()
        q_colbert = list(colbert_model.query_embed(q["query"]))[0].tolist()

        r_c = first_hit_rank(hits_colbert(client, q_dense, q_colbert, k), q, k)
        r_h = first_hit_rank(hits_hybrid(client, q["query"], q_dense, k), q, k)
        modes["colbert"].append(r_c is not None)
        modes["hybrid"].append(r_h is not None)
        print(f"{q['id']:4} {'ok':9} "
              f"{(f'@{r_c}' if r_c else 'MISS'):8} {(f'@{r_h}' if r_h else 'MISS'):8} "
              f"{q['query'][:50]}")

    print(f"\nrecall@{k}: " + "  ".join(
        f"{m}={sum(v)}/{len(v)} ({100 * sum(v) / max(len(v), 1):.0f}%)" for m, v in modes.items()))


if __name__ == "__main__":
    main()
