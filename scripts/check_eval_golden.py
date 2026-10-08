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

import eval_rag
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
