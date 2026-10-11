"""Golden-question retrieval eval for the ETSI MEC pipeline.

    uv run python scripts/eval_rag.py [--top-k 5] [--answer-context 15] [--per-doc 2]

Exercises three retrieval paths from etsi_mec_agent.search:
  colbert   — dense prefetch + ColBERT MaxSim rescore (search_specs path 2)
  hybrid    — Qdrant server-side dense+sparse prefetch with RRF fusion (_run_hybrid_retrieval)
  aggregate — what --answer reads: the hybrid path with contiguous page parts stitched first,
              measured over the same --answer-context / --per-doc budget the CLI uses
              (defaults 15 / 2). This column is the LLM's evidence, not the --top-k printed.

A question passes when a top-k chunk from one of its expected specs contains one of its
expected keywords. `docs` names content-derived spec identities (MEC-003), not filenames, and
is a **set** because a term is often defined normatively in more than one live spec — scoring
only one of them punishes a correct retrieval. Every path is measured behind build_filter() — the same current-editions-only filter
search_specs ships — because 59 of the 106 doc_ids name a different spec than the text
they hold. Questions whose keywords exist nowhere in the expected spec are flagged
EVIDENCE-MISSING (bad golden data, not a retrieval failure) so the numbers stay honest.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qdrant_client import models

from etsi_mec_agent.config import settings
from etsi_mec_agent.dedup import SimpleDoc, _dedup_docs, _stitch_parts
from etsi_mec_agent.search import _run_hybrid_retrieval, build_filter, get_embedders
from etsi_mec_agent.store import get_qdrant_client


def doc_key(meta):
    """The document a chunk belongs to: its content-derived spec, else its doc_id."""
    return (meta or {}).get("spec_id") or (meta or {}).get("doc_id", "")

QUESTIONS = [
    # docs is the set of specs whose *current* edition defines the thing asked about; None means
    # any document may satisfy it. Sets come from scanning the live collection, not from what the
    # question seems to be about: "Mm6", "life cycle management api" and the RNIS requirement each
    # live in exactly one spec, while "LCM proxy" and "MEC service" are defined in several.
    {"id": "q01", "query": "Which reference point connects the MEC application to the MEC platform?", "docs": ["MEC-003"], "kw": ["Mp1"]},
    {"id": "q02", "query": "Which reference point connects the MEC platform to the MEC orchestrator?", "docs": ["MEC-003"], "kw": ["Mm3"]},
    {"id": "q03", "query": "What is the Mm6 reference point used for?", "docs": ["MEC-003"], "kw": ["Mm6"]},
    {"id": "q04", "query": "Which service exposes radio network information to applications?", "docs": None, "kw": ["RNIS", "Radio Network Information"]},
    {"id": "q05", "query": "What is the Edge Enabler Client (EEC)?", "docs": None, "kw": ["Edge Enabler Client", "EEC"]},
    # "TCR" occurs in no chunk of this corpus; the entity that expresses traffic influence
    # policies toward the 5GC is the CCMF acting as an AF over Nnef_TrafficInfluence. That
    # service is named in MEC-059 (the CCMF spec), MEC-031 and MEC-038 (both propose it).
    {"id": "q06", "query": "Over which 5GC service does the CCMF express application-specific traffic influence policies?", "docs": ["MEC-059", "MEC-031", "MEC-038"], "kw": ["Nnef_TrafficInfluence"]},
    {"id": "q07", "query": "Which service continuity modes are defined for MEC applications?", "docs": None, "kw": ["service continuity"]},
    # No single spec owns "User app LCM proxy": MEC-021 calls it a MEC system level functional
    # entity, MEC-017 and MEC-024 describe its Mm9, MEC-003 places it in the architecture.
    {"id": "q08", "query": "What is the User app LCM proxy?", "docs": ["MEC-003", "MEC-021", "MEC-017", "MEC-024"], "kw": ["LCM proxy"]},
    {"id": "q09", "query": "How does a road tunnel affect TCP congestion control in the MEC use case?", "docs": ["MEC-002"], "kw": ["road tunnel", "TCP"]},
    {"id": "q10", "query": "What are the components of the MEC host level reference architecture?", "docs": ["MEC-003"], "kw": ["MEC host", "Virtualisation"]},
    {"id": "q11", "query": "What is the UU interface used for in the V2X deployment?", "docs": ["MEC-030"], "kw": ["uu interface"]},
    # Was kw ["Mm5", "Mm6"]: "Mm5" occurs in 56 chunks, so any of them counted as evidence and
    # no ranking could single the selection step out. "selects the MEC host" occurs in 2 chunks
    # (MEC-010-2, MEC-040), and docs stays None because those two both answer it correctly — a
    # miss here is a retrieval miss, not a golden-data miss.
    {"id": "q12", "query": "Which entity selects the MEC host for application instantiation?", "docs": None, "kw": ["selects the MEC host"]},
    {"id": "q13", "query": "How does DNS resolution steer users to the closest MEC server?", "docs": None, "kw": ["DNS"]},
    # MEC-003 states the platform's role (registration enabling discovery, Mp1 in 7.2.1); MEC-011
    # is the spec that defines the registration and discovery APIs themselves.
    {"id": "q14", "query": "What is the role of the MEC platform in service discovery?", "docs": ["MEC-003", "MEC-011"], "kw": ["service registration", "discovery"]},
    # Was kw ["mpInfoService", "Mm3"] with docs None: "mpInfoService" occurs in **0** chunks, so
    # the question was scoring bare "Mm3" presence. "life cycle management api" occurs in 3 chunks,
    # all MEC-010-2, which is what actually documents the API over Mm3/Mm3*.
    {"id": "q15", "query": "Over which reference point does the MEC orchestrator's application life cycle management API run?", "docs": ["MEC-010-2"], "kw": ["life cycle management api"]},
    # "A MEC service is provided and consumed" is in MEC-002, the term definition in MEC-001.
    {"id": "q16", "query": "What is a MEC service and how does it relate to a MEC application?", "docs": ["MEC-003", "MEC-002", "MEC-001", "MEC-011"], "kw": ["MEC service"]},
]


def load_corpus(client):
    """doc_id -> lowercased full text, for the evidence pre-check."""
    corpus: dict = {}
    offset = None
    while True:
        pts, offset = client.scroll(
            collection_name=settings.qdrant_index, limit=500,
            with_payload=["doc_id", "spec_id", "is_current", "text"],
            with_vectors=False, offset=offset,
        )
        for p in pts:
            if (p.payload or {}).get("is_current") is False:
                continue      # the eval measures the path search_specs actually ships
            doc = doc_key(p.payload)
            corpus[doc] = corpus.get(doc, "") + "\n" + p.payload.get("text", "").lower()
        if offset is None:
            break
    return corpus


def evidence_ok(q, corpus):
    docs = q["docs"] or list(corpus)
    return any(any(k.lower() in corpus.get(d, "") for k in q["kw"]) for d in docs)


def hits_colbert(client, q_dense, q_colbert, k):
    res = client.query_points(
        collection_name=settings.qdrant_index,
        prefetch=models.Prefetch(query=q_dense, using="dense", limit=25, filter=build_filter()),
        query=q_colbert, using="colbert", limit=k,  # exactly what search_specs requests
        query_filter=build_filter(),
    )

    # Same wrapper search_specs builds for its own hits, so _dedup_docs applies identically.
    docs = [SimpleDoc(h.payload.get("text", ""), h.payload) for h in res.points]
    return [(doc_key(d.meta), d.content) for d in _dedup_docs(docs, keep=k)]


def hits_hybrid(client, q_text, q_dense, k):
    raw = _run_hybrid_retrieval(client, q_text, q_dense, top_k=k * 6, prefetch_limit=100)  # defaults from search_specs
    raw = _dedup_docs(raw, keep=k)  # same post-fusion dedup as search_specs
    return [(doc_key(d.meta), d.content or "") for d in raw]


def hits_aggregate(client, q_text, q_dense, ctx=15, per_doc=2):
    """What --answer sees: contiguous page parts stitched, up to per_doc excerpts per doc."""
    raw = _run_hybrid_retrieval(client, q_text, q_dense, top_k=ctx * 6, prefetch_limit=100)
    docs = _dedup_docs(_stitch_parts(raw), keep=ctx, per_doc=per_doc)
    return [(doc_key(d.meta), d.content) for d in docs]


def first_hit_rank(hits, q, k):
    for rank, (doc, text) in enumerate(hits[:k], 1):
        if q["docs"] and doc not in q["docs"]:
            continue
        if any(kw.lower() in text.lower() for kw in q["kw"]):
            return rank
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--answer-context", type=int, default=15,
                    help="Evidence budget the aggregate column measures (mirrors search --answer-context)")
    ap.add_argument("--per-doc", type=int, default=2,
                    help="Excerpts per document in the aggregate column (mirrors search --per-doc)")
    args = ap.parse_args()
    if args.answer_context < 1 or args.per_doc < 1:
        ap.error("--answer-context and --per-doc must be >= 1")
    k, window, per_doc = args.top_k, args.answer_context, args.per_doc

    client = get_qdrant_client()
    corpus = load_corpus(client)
    dense_model, colbert_model = get_embedders()

    modes = {"colbert": [], "hybrid": []}
    agg = []
    print(f"{'id':4} {'evidence':9} {'colbert':8} {'hybrid':8} {'aggregate':10} query")
    for q in QUESTIONS:
        if not evidence_ok(q, corpus):
            print(f"{q['id']:4} {'MISSING':9} {'-':8} {'-':8} {'-':10} {q['query'][:50]}")
            continue
        q_dense = list(dense_model.embed([q["query"]]))[0].tolist()
        q_colbert = list(colbert_model.query_embed(q["query"]))[0].tolist()

        r_c = first_hit_rank(hits_colbert(client, q_dense, q_colbert, k), q, k)
        r_h = first_hit_rank(hits_hybrid(client, q["query"], q_dense, k), q, k)
        r_a = first_hit_rank(hits_aggregate(client, q["query"], q_dense, ctx=window, per_doc=per_doc), q, window)
        modes["colbert"].append(r_c is not None)
        modes["hybrid"].append(r_h is not None)
        agg.append(r_a is not None)
        print(f"{q['id']:4} {'ok':9} "
              f"{(f'@{r_c}' if r_c else 'MISS'):8} {(f'@{r_h}' if r_h else 'MISS'):8} "
              f"{(f'@{r_a}' if r_a else 'MISS'):10} "
              f"{q['query'][:50]}")

    print(f"\nrecall@{k}: " + "  ".join(
        f"{m}={sum(v)}/{len(v)} ({100 * sum(v) / max(len(v), 1):.0f}%)" for m, v in modes.items())
        + f"   aggregate@{window}: {sum(agg)}/{len(agg)} "
          f"({100 * sum(agg) / max(len(agg), 1):.0f}%)")


if __name__ == "__main__":
    main()
