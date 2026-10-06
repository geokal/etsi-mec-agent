# Query-Time Aggregation (Design A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `search --answer` read the whole continuation of every retrieved passage and up to N excerpts per document, instead of five truncated chunks.

**Architecture:** Retrieval keeps its current Qdrant queries. A new dependency-free module owns rank shaping: stitch contiguous `chunk_part`s back into whole passages, then dedup with a per-document allowance. `search_specs` splits the old single result list into a *display* list (`--top-k`, unchanged) and an *evidence* list (`--answer-context`, default 15, `--per-doc` 2) that is what reaches the LLM.

**Tech Stack:** Python 3.12, `uv`, Qdrant client 1.19, pytest is **not** used in this repo — verification is assert scripts under `scripts/check_*.py` run with `uv run python` (see `AGENTS.md` "Do not").

Spec: `docs/superpowers/specs/2026-10-06-corpus-wide-answering-design.md` §3 (A1–A5).

---

## File structure

| File | Responsibility |
|---|---|
| Create `src/etsi_mec_agent/dedup.py` | Pure rank/text shaping: `SimpleDoc`, `_stitch`, `_stitch_parts`, `_dedup_docs`. Stdlib only — importing it must never load fastembed/ONNX, so checks run in milliseconds. |
| Create `scripts/check_dedup_stitch.py` | Assert-based checks for the three functions. The repo's test pattern. |
| Modify `src/etsi_mec_agent/search.py` | Delete the two helpers now in `dedup.py` (`:93-130` `_dedup_docs`, `:137-143` `_make_simple_doc`), import them, add `answer_context`/`per_doc`, split display vs evidence, add two CLI flags. |
| Modify `scripts/eval_rag.py` | Import `_dedup_docs` from `dedup`, add the `aggregate` column. |
| Modify `README.md`, `AGENTS.md` | Document the new flags and the evidence-vs-display split. |

Call sites that must be updated when the helpers move: `search.py:312`, `search.py:377`, `search.py:410`, `eval_rag.py:23`, `eval_rag.py:85`, `eval_rag.py:90`.

---

## Task 1: `dedup.py` with `SimpleDoc` and overlap-aware `_stitch`

**Files:**
- Create: `src/etsi_mec_agent/dedup.py`
- Create: `scripts/check_dedup_stitch.py`

- [ ] **Step 1: Write the failing check**

Create `scripts/check_dedup_stitch.py`:

```python
"""Checks for the rank-shaping helpers in etsi_mec_agent.dedup.

    uv run python scripts/check_dedup_stitch.py

No models, no Qdrant: dedup.py is stdlib-only on purpose so this runs in ms.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from etsi_mec_agent.dedup import SimpleDoc, _stitch

# The 40-word sliding window repeats part of a's tail at the head of b; stitch drops it.
a = " ".join(f"alpha{i}" for i in range(40)) + " TAIL0 TAIL1 TAIL2"
b = "TAIL0 TAIL1 TAIL2 beta0 beta1 beta2"
out = _stitch(a, b)
assert out.count("TAIL1") == 1, f"overlap not trimmed: {out}"
assert out.startswith("alpha0") and out.endswith("beta2"), out

# No overlap: plain concatenation, nothing invented.
assert _stitch("one two", "three four") == "one two three four"

# b entirely contained in a's tail (degenerate window) -> no duplicate text.
assert _stitch("x y z", "y z") == "x y z"

print("[check] _stitch: overlap trim, concat, degenerate tail — OK")
```

- [ ] **Step 2: Run it and watch it fail**

Run: `uv run python scripts/check_dedup_stitch.py`
Expected: `ModuleNotFoundError: No module named 'etsi_mec_agent.dedup'`

- [ ] **Step 3: Write the minimal implementation**

Create `src/etsi_mec_agent/dedup.py`:

```python
"""Rank shaping for retrieval results: stitching passage parts and dedup.

Stdlib-only by design. `search.py` imports fastembed at module level, so anything
that lives here would otherwise drag ONNX runtime into every check script.
"""
import re

_IMG_RE = re.compile(r"!\[.*?\]\(.*?\)")   # markdown image refs vary between copies of the same text


class SimpleDoc:
    """A retrieved excerpt: its text plus the Qdrant payload it came with."""

    def __init__(self, content: str, meta: dict):
        self.content = content
        self.meta = meta


def _stitch(a: str, b: str, max_overlap: int = 60) -> str:
    """Append b to a, dropping the tokens b repeats from the end of a.

    `chunk_page_text` emits chunks with a 40-word overlap and joins words with single
    spaces, so token-level comparison is lossless and the trim can't corrupt markdown.
    """
    aw, bw = a.split(), b.split()
    n = min(len(aw), len(bw), max_overlap)
    for k in range(n, 0, -1):
        if aw[-k:] == bw[:k]:
            return " ".join(aw + bw[k:])
    return " ".join(aw + bw)
```

- [ ] **Step 4: Run the check again**

Run: `uv run python scripts/check_dedup_stitch.py`
Expected: `[check] _stitch: overlap trim, concat, degenerate tail — OK`

- [ ] **Step 5: Commit**

```bash
git add src/etsi_mec_agent/dedup.py scripts/check_dedup_stitch.py
git commit -m "Add dependency-free dedup module with overlap-aware stitch"
```

---

## Task 2: Move `_dedup_docs` into `dedup.py` and add `per_doc`

**Files:**
- Modify: `src/etsi_mec_agent/dedup.py` (append)
- Modify: `src/etsi_mec_agent/search.py:90-130` (delete `_dedup_docs` and `_IMG_RE`)
- Modify: `scripts/check_dedup_stitch.py` (append)

- [ ] **Step 1: Write the failing checks**

Append to `scripts/check_dedup_stitch.py`:

```python
from etsi_mec_agent.dedup import _dedup_docs


def _docs(specs):
    """specs: list of (doc_id, text) -> SimpleDoc list"""
    return [SimpleDoc(t, {"doc_id": d}) for d, t in specs]


# Baseline behaviour must not change: one excerpt per doc_id, capped at keep.
one = _dedup_docs(_docs([("MEC003", "aa bb cc"), ("MEC003", "dd ee ff"), ("MEC030", "gg")]), keep=5)
assert [d.meta["doc_id"] for d in one] == ["MEC003", "MEC030"], one

# per_doc=2 keeps the continuation of the same document.
two = _dedup_docs(_docs([("MEC003", "aa bb"), ("MEC003", "dd ee"), ("MEC003", "ff gg"),
                         ("MEC030", "hh ii")]), keep=5, per_doc=2)
assert [d.meta["doc_id"] for d in two] == ["MEC003", "MEC003", "MEC030"], two

# keep still wins over per_doc.
capped = _dedup_docs(_docs([("MEC003", "aa"), ("MEC003", "bb"), ("MEC003", "cc")]), keep=2, per_doc=2)
assert len(capped) == 2, capped

# Identical text under two doc_ids collapses (this is what hides the mislabelled copies).
same = _dedup_docs(_docs([("MEC003", "xx yy zz"), ("MEC070", "xx yy zz")]), keep=5)
assert len(same) == 1, same

print("[check] _dedup_docs: default, per_doc, keep cap, cross-id text collapse — OK")
```

- [ ] **Step 2: Run it and watch it fail**

Run: `uv run python scripts/check_dedup_stitch.py`
Expected: `ImportError: cannot import name '_dedup_docs' from 'etsi_mec_agent.dedup'`

- [ ] **Step 3: Move the function, keeping semantics identical**

Append to `src/etsi_mec_agent/dedup.py`. This is `search.py:93-130` with `per_doc` added to pass 2 and nothing else changed:

```python
def _dedup_docs(docs: list, keep: int, per_doc: int = 1) -> list:
    """
    Two-pass deduplication of retrieved hits.

    Pass 1 — text fingerprint (first 400 chars, images stripped): catches the same
    chunk stored under different doc_ids, which is common in this corpus because each
    PDF extraction embeds a different image path in the markdown.

    Pass 2 — up to `per_doc` excerpts per doc_id, `keep` results overall. per_doc=1
    reproduces the historical one-chunk-per-document rule; larger values let a
    document contribute more than its single best-ranked page.
    """
    seen_text: set = set()
    after_text: list = []
    for doc in docs:
        raw = doc.content if hasattr(doc, "content") else (doc.meta or {}).get("text", "")
        fp = _IMG_RE.sub("", raw or "")[:400].strip()
        if fp not in seen_text:
            seen_text.add(fp)
            after_text.append(doc)

    counts: dict = {}
    unique: list = []
    for doc in after_text:
        meta = doc.meta if hasattr(doc, "meta") else {}
        doc_id = (meta or {}).get("doc_id", "") or (doc.content or "")[:40]
        if counts.get(doc_id, 0) >= per_doc:
            continue
        counts[doc_id] = counts.get(doc_id, 0) + 1
        unique.append(doc)
        if len(unique) >= keep:
            break

    return unique
```

In `src/etsi_mec_agent/search.py`, delete lines 90-130 (`_IMG_RE` plus the whole `_dedup_docs`) and add to the imports at the top:

```python
from etsi_mec_agent.dedup import SimpleDoc, _dedup_docs
```

- [ ] **Step 4: Run the check and the import smoke test**

Run: `uv run python scripts/check_dedup_stitch.py`
Expected: both `[check]` lines, exit 0.

Run: `uv run python -c "import sys; sys.path.insert(0,'src'); import etsi_mec_agent.search as s; print(s._dedup_docs.__module__)"`
Expected: `etsi_mec_agent.dedup` — proves the moved name resolves for existing callers.

- [ ] **Step 5: Commit**

```bash
git add src/etsi_mec_agent/dedup.py src/etsi_mec_agent/search.py scripts/check_dedup_stitch.py
git commit -m "Move dedup into stdlib-only module with per-document allowance"
```

---

## Task 3: `_stitch_parts` — reassemble contiguous page parts

**Files:**
- Modify: `src/etsi_mec_agent/dedup.py` (append)
- Modify: `scripts/check_dedup_stitch.py` (append)

- [ ] **Step 1: Write the failing checks**

Append to `scripts/check_dedup_stitch.py`:

```python
from etsi_mec_agent.dedup import _stitch_parts


def _parts(doc_id, page, part, total, text):
    return SimpleDoc(text, {"doc_id": doc_id, "page": page,
                            "chunk_part": part, "total_parts": total})


# 1+2 of the same page merge; the row that only existed in part 2 becomes visible.
p1 = _parts("MEC003", 16, 1, 2, "|Mm7:|The Mm7 reference point between the VIM and")
p2 = _parts("MEC003", 16, 2, 2, "point between the VIM and the VI is used to manage the VI")
st = _stitch_parts([p1, p2])
assert len(st) == 1, st
assert "used to manage" in st[0].content, st[0].content
assert st[0].meta["stitched_parts"] == 2 and st[0].meta["chunk_part"] == 1, st[0].meta

# 1+3 is a gap: joining it would fabricate a page that was never retrieved.
gap = _stitch_parts([_parts("MEC003", 16, 1, 3, "aaa bbb"), _parts("MEC003", 16, 3, 3, "eee fff")])
assert len(gap) == 2, gap

# Different pages of the same doc never merge.
pages = _stitch_parts([_parts("MEC003", 16, 1, 2, "one"), _parts("MEC003", 17, 1, 2, "two")])
assert len(pages) == 2, pages

# Whole-page docs (total_parts=1) and points with no part metadata pass through untouched.
solo = _parts("MEC030", 8, 1, 1, "solo text")
legacy = SimpleDoc("legacy text", {"doc_id": "MEC030"})
assert _stitch_parts([solo, legacy]) == [solo, legacy]

# Out-of-order parts are not falsely merged (2 arriving before 1 is not contiguous).
dis = _stitch_parts([_parts("MEC003", 16, 2, 3, "bbb"), _parts("MEC003", 16, 1, 3, "aaa")])
assert len(dis) == 2, dis

print("[check] _stitch_parts: merge, gap, pages, passthrough, order — OK")
```

- [ ] **Step 2: Run it and watch it fail**

Run: `uv run python scripts/check_dedup_stitch.py`
Expected: `ImportError: cannot import name '_stitch_parts'`

- [ ] **Step 3: Implement**

Append to `src/etsi_mec_agent/dedup.py`:

```python
def _stitch_parts(docs: list) -> list:
    """
    Merge runs of contiguous `chunk_part`s sharing a doc_id+page into one excerpt.

    Retrieval returns parts in score order, so a page split into 1 of 2 / 2 of 2 only
    merges when the two halves both arrived and sit next to each other in that order.
    Non-contiguous or reversed parts stay separate rather than fabricating a page.
    Points without usable part metadata pass through unchanged.
    """
    groups: list = []
    open_group: dict = {}          # (doc_id, page) -> index of the run still accepting parts

    for doc in docs:
        meta = doc.meta or {}
        part, total = meta.get("chunk_part"), meta.get("total_parts")
        if not isinstance(part, int) or not isinstance(total, int) or total <= 1:
            groups.append([doc])
            continue
        key = (meta.get("doc_id", ""), meta.get("page"))
        idx = open_group.get(key)
        if idx is not None and groups[idx][-1].meta["chunk_part"] + 1 == part:
            groups[idx].append(doc)
        else:
            open_group[key] = len(groups)
            groups.append([doc])

    out: list = []
    for group in groups:
        if len(group) == 1:
            out.append(group[0])
            continue
        text = group[0].content
        for nxt in group[1:]:
            text = _stitch(text, nxt.content)
        meta = dict(group[0].meta)
        meta["stitched_parts"] = len(group)
        out.append(SimpleDoc(text, meta))
    return out
```

- [ ] **Step 4: Run the check**

Run: `uv run python scripts/check_dedup_stitch.py`
Expected: three `[check]` lines, exit 0.

- [ ] **Step 5: Commit**

```bash
git add src/etsi_mec_agent/dedup.py scripts/check_dedup_stitch.py
git commit -m "Reassemble contiguous chunk parts at query time"
```

---

## Task 4: Point `search.py` at the moved helpers, delete the duplicates

**Files:**
- Modify: `src/etsi_mec_agent/search.py:77-82` (`_run_hybrid_retrieval` inner `_Doc`)
- Modify: `src/etsi_mec_agent/search.py:137-143` (`_make_simple_doc`)
- Modify: `src/etsi_mec_agent/search.py:312`, `:369-378`, `:410`

- [ ] **Step 1: Replace the duplicated doc wrapper**

In `_run_hybrid_retrieval`, delete the local class and use the shared one — replace lines 77-82:

```python
    class _Doc:
        def __init__(self, text, meta):
            self.content = text
            self.meta = meta

    return [_Doc(h.payload.get("text", ""), h.payload) for h in results.points]
```

with:

```python
    return [SimpleDoc(h.payload.get("text", ""), h.payload) for h in results.points]
```

- [ ] **Step 2: Delete `_make_simple_doc` and its call site**

Remove the function at `search.py:137-143`. Then in the dense+ColBERT path replace line 410:

```python
        docs = [_make_simple_doc(hit.payload.get("text", ""), hit.payload) for hit in unique_points]
```

with:

```python
        docs = [SimpleDoc(hit.payload.get("text", ""), hit.payload) for hit in unique_points]
```

- [ ] **Step 3: Delete the now-dead `_HitDoc` comment reference**

At `search.py:369`, the comment names `_dedup_docs`, which is still accurate. Leave `class _HitDoc` (it carries `._hit` for the print loop) but change its `meta` assignment to nothing else — no edit needed unless the module no longer defines `_dedup_docs`. Confirm with:

Run: `uv run python -c "import ast,sys; src=open('src/etsi_mec_agent/search.py').read(); ast.parse(src); print('parses'); print('_dedup_docs' in src, src.count('_make_simple_doc'))"`
Expected: `parses`, then `True 0` (callers still reference `_dedup_docs` imported from `dedup`; zero remaining `_make_simple_doc`).

- [ ] **Step 4: Verify nothing else referenced the deleted names**

Run: `grep -rn "_make_simple_doc\|_IMG_RE" --include=*.py src scripts | grep -v "\.venv"`
Expected: only `src/etsi_mec_agent/dedup.py` hits for `_IMG_RE`; no `_make_simple_doc` anywhere.

- [ ] **Step 5: Commit**

```bash
git add src/etsi_mec_agent/search.py
git commit -m "Reuse shared SimpleDoc and drop search.py's duplicate wrappers"
```

---

## Task 5: Split evidence budget from display in `search_specs`

**Files:**
- Modify: `src/etsi_mec_agent/search.py:266-275` (signature), `:301-339` (hybrid), `:346-358` (colbert limit), `:377`, `:409-411`, `:426-459` (CLI)

- [ ] **Step 1: Add the two parameters**

In `search_specs`' signature (`:266-275`) add after `use_bm25: bool = False,`:

```python
    answer_context: int = 15,
    per_doc: int = 2,
```

and change `prefetch_limit: int = 25,` to stay as-is. Compute the fetch width once, right after `client = get_qdrant_client()`:

```python
    # The LLM reads more than the terminal prints: retrieval must fetch enough
    # candidates to fill the evidence budget after stitching and dedup.
    fetch_k = max(top_k, answer_context) if generate else top_k
```

- [ ] **Step 2: Hybrid path — two dedup passes over the same raw hits**

Replace `search.py:303-312`:

```python
        hybrid_docs = _run_hybrid_retrieval(
            client=client,
            query_text=query_text,
            query_dense=query_dense,
            top_k=top_k * 6,          # fetch 6× more so dedup still yields top_k unique docs
            prefetch_limit=max(prefetch_limit * 4, 100),
            query_filter=query_filter,
        )

        unique_docs = _dedup_docs(hybrid_docs, keep=top_k)
```

with:

```python
        hybrid_docs = _stitch_parts(_run_hybrid_retrieval(
            client=client,
            query_text=query_text,
            query_dense=query_dense,
            top_k=fetch_k * 6,        # fetch 6x more so dedup still fills the budget
            prefetch_limit=max(prefetch_limit * 4, 100),
            query_filter=query_filter,
        ))

        unique_docs = _dedup_docs(hybrid_docs, keep=top_k)
        answer_docs = _dedup_docs(hybrid_docs, keep=answer_context, per_doc=per_doc) if generate else []
```

and change the generate call at `:338` to use it:

```python
        if generate:
            print(f"[ANSWER CONTEXT] {len(answer_docs)} excerpts, up to {per_doc} per document "
                  f"(~{sum(len(d.content.split()) for d in answer_docs)} words).")
            generate_answer(query_text, answer_docs, stream=stream, show_reasoning=show_reasoning)
        return []
```

- [ ] **Step 3: ColBERT path — same treatment**

`search.py:346-358`: change `limit=top_k` (the outer one, on `query_points`) to `limit=fetch_k`. After `hit_docs` is built at `:376`, stitch before dedup:

```python
    hit_docs  = _stitch_parts([_HitDoc(h) for h in results.points])
    deduped   = _dedup_docs(hit_docs, keep=top_k)
    answer_docs = _dedup_docs(hit_docs, keep=answer_context, per_doc=per_doc) if generate else []
```

Note `_HitDoc` exposes `._hit`, and `_stitch_parts` returns `SimpleDoc` for merged runs, so a merged excerpt no longer has `._hit`. The print loop consumes `unique_points` built from `deduped` — keep it working by rebuilding from the deduped docs instead of hits: replace `search.py:378` (`unique_points = [d._hit for d in deduped]`) and the loop's `payload = hit.payload` / `hit.score` reads so the loop iterates `deduped` and uses `doc.meta`, printing `doc.meta.get("score", "")`. Scores are not returned by the stitched path, so print the doc_id/page/heading header only:

```python
    for i, doc in enumerate(deduped, 1):
        payload = doc.meta or {}
        doc_id        = payload.get("doc_id", "Unknown")
        filename      = payload.get("filename", "")
        page          = payload.get("page", "N/A")
        heading       = payload.get("heading", "")
        text          = (doc.content or "").strip()
        has_diagram   = payload.get("has_diagram", False)
        diagram_paths = payload.get("diagram_paths", [])

        print(f"=== [Result {i}] Page: {page} | Doc: {filename or doc_id} ===")
```

and change the `--answer` call at `:410-411` to pass `answer_docs` instead of rebuilding `docs` from `unique_points`.

- [ ] **Step 4: Expose both knobs on the CLI**

In `main()` add after the `--prefetch` argument (`search.py:427`):

```python
    parser.add_argument(
        "--answer-context", type=int, default=15,
        help="Excerpts handed to the LLM when --answer is used (default 15; terminal still shows --top-k)",
    )
    parser.add_argument(
        "--per-doc", type=int, default=2,
        help="Excerpts allowed per document inside --answer-context (default 2)",
    )
```

and pass them in the `search_specs(...)` call:

```python
        answer_context=args.answer_context,
        per_doc=args.per_doc,
```

- [ ] **Step 5: Import check, then the live behaviour check**

Run: `uv run python -c "import ast; ast.parse(open('src/etsi_mec_agent/search.py').read()); print('parses')"`
Expected: `parses`

Run (no model load, Qdrant only — uses a zero-ish dense vector):

```bash
uv run python -c "
import sys; sys.path.insert(0,'src')
from etsi_mec_agent.search import search_specs
search_specs('Which reference point connects the MEC platform to the MEC orchestrator?',
             top_k=3, use_bm25=True, generate=False)
print('display-only run OK')"
```
Expected: three `[Hybrid i]` blocks and `display-only run OK`.

- [ ] **Step 6: Commit**

```bash
git add src/etsi_mec_agent/search.py
git commit -m "Give the LLM its own evidence budget separate from terminal output"
```

---

## Task 6: Measure it — `aggregate` column in the eval

**Files:**
- Modify: `scripts/eval_rag.py:23` (import), `:85-90` (helpers), table print, summary lines

- [ ] **Step 1: Import from the new home**

Change `eval_rag.py:23` to:

```python
from etsi_mec_agent.dedup import _dedup_docs, _stitch_parts
from etsi_mec_agent.search import _run_hybrid_retrieval, get_embedders
```

- [ ] **Step 2: Add the aggregate retriever**

After `hits_hybrid` (`:88-90`) add — `k` here is the *context* size, deliberately wider than the recall@5 columns, and this is the one function that models what `--answer` now reads:

```python
def hits_aggregate(client, q_text, q_dense, ctx=15, per_doc=2):
    """What --answer sees: contiguous parts stitched, up to per_doc excerpts per doc."""
    raw = _run_hybrid_retrieval(client, q_text, q_dense, top_k=ctx * 6, prefetch_limit=100)
    docs = _dedup_docs(_stitch_parts(raw), keep=ctx, per_doc=per_doc)
    return [(d.meta.get("doc_id", ""), d.content) for d in docs]
```

- [ ] **Step 3: Print the third column**

In the per-question loop add, next to the existing `r_h = first_hit_rank(...)`:

```python
        r_a = first_hit_rank(hits_aggregate(client, q["query"], q_dense, ctx=k * 3), q, k * 3)
```

and add an `agg@{k*3}` cell to the header and each row so the three columns print side by side. Accumulate a `hits_a` counter and print a third recall line:

```python
    print(f"\nrecall@{k}: colbert={hits_c}/{valid} ({pct(hits_c)})  "
          f"hybrid={hits_h}/{valid} ({pct(hits_h)})  "
          f"aggregate@{k*3}: {hits_a}/{valid} ({pct(hits_a)})")
```

If `eval_rag.py` currently has no `pct` helper, use `round(100*hits_a/valid)` inline instead — do not add a helper for one call site.

- [ ] **Step 4: Run the eval** (loads ONNX — user runs this in their own terminal)

Run: `uv run python scripts/eval_rag.py --top-k 5`
Expected: unchanged `colbert=10/16` and `hybrid=10/16`, and an aggregate column of at least 12/16. Per spec §5, **at least two of q06/q08/q12/q14 must move from MISS to a rank** — that is the acceptance bar for A.

- [ ] **Step 5: Commit**

```bash
git add scripts/eval_rag.py
git commit -m "Measure query-time aggregation against the golden question set"
```

---

## Task 7: Document the split

**Files:**
- Modify: `README.md` (usage sections, "How an answer is assembled")
- Modify: `AGENTS.md` (search flags table)

- [ ] **Step 1: Flags table** — in `AGENTS.md`, after the `--use-bm25` row add:

```markdown
| `--answer-context N` | Excerpts handed to the LLM with `--answer` (default 15) — terminal still shows `--top-k` |
| `--per-doc N` | Excerpts per document inside the answer context (default 2) |
```

- [ ] **Step 2: Rewrite the now-false limit** in `README.md` "How an answer is assembled": the first bullet ("**One chunk per document.**") must be replaced with the stitched/per-document behaviour and the two flags, keeping the remaining bullets (paths-not-pixels, one query one pass, labels approximate) intact.

- [ ] **Step 3: Commit**

```bash
git add README.md AGENTS.md
git commit -m "Document the answer-context and per-doc knobs"
```

---

## Task 8: End-to-end proof with the LLM

- [ ] **Step 1: The Mm3 question that motivated this** (user terminal; loads ONNX + calls OpenRouter):

```bash
uv run python -m etsi_mec_agent.search "Which reference point connects the MEC platform to the MEC orchestrator?" --use-bm25 --answer --answer-context 15 --per-doc 2
```

Expected: an `[ANSWER CONTEXT] 15 excerpts, up to 2 per document (~N words)` line, then an answer whose quoted table is not cut mid-row — compare against the pre-change run that ended `|Mm9:|The Mm9 reference`.

- [ ] **Step 2: A both-miss question from the baseline** — `uv run python -m etsi_mec_agent.search "What traffic influence rules does the TCR expose?" --use-bm25 --answer`. Expected: it appears in the excerpt list now, or the eval shows why it still misses.

- [ ] **Step 3: Record the numbers** — save the eval output to `backups/eval_after_stitch.txt` and note in the spec's §5 whether A passed its bar. Design B is not started until this is written down.

---

## Self-review notes

- Spec §3 A1–A5 map to tasks 1-3 (stitch), 2 (per_doc), 5 (answer-context), 6 (eval), 3+1 (check script) — no orphan requirements. A1's "longest-suffix trim" is Task 1 `_stitch`; A2's default-preserving rule is asserted in Task 2.
- No placeholders: every code step shows the code. Commands show expected output.
- Names are consistent across tasks: `SimpleDoc`, `_stitch`, `_stitch_parts`, `_dedup_docs(docs, keep, per_doc=1)`.
- Task 5 Step 3 is the only step that deletes user-visible output (the per-result `Score:` prefix), because scores are meaningless once excerpts are merged; it is called out explicitly rather than smuggled in.
