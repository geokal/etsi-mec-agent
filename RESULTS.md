# Measured results

Every number below has a date and a command. Where a number cannot be reproduced from this
repository, it says so. Single author, 26 Sep – 7 Nov 2026.

## 1. Retrieval quality — 16 golden engineering questions

    uv run python scripts/eval_rag.py --top-k 5

| Measured | colbert@5 | hybrid@5 | aggregate@15 | union | What changed |
|---|---|---|---|---|---|
| 2026-09-26 (start) | 10 | 10 | — | 12 | word-window chunks, filename-based identity |
| 2026-10-06 (§5b) | 10 | 10 | 11 | — | rank shaping: part-stitching, per-document dedup, answer budget |
| 2026-10-06 (§5d) | 11 | 11 | 12 | — | cover-derived identity, edition filtering, `content_md5` quota |
| 2026-10-08 (§5e) | 11 | 12 | **14** | 15 | row-/clause-aware chunking with the clause floor |
| 2026-10-11 (§5i) | 11 | 12 | **14** | 15 | review-round fixes — identical, i.e. rank-neutral |
| 2026-11-07 (c183eb2) | 11 | **14** | **14** | 14 | RRF pool rescored by ColBERT MaxSim (nested prefetch-fuse-rerank) |
| 2026-11-07 (9d7bae7) | 11 | **12** | **12** | 12 | corrected harness — **not comparable to any row above it** |

`union` is the two top-5 paths (colbert@5 ∪ hybrid@5), not all three columns.

**Only the last row is reproducible with today's `eval_rag.py`.** Until 2026-11-07 the hybrid and
aggregate columns queried *without* `build_filter()`, so superseded editions were scoreable while the
ColBERT column was not; the aggregate column fetched a flat 100 candidates per leg instead of the
`--answer-context`-derived 180; and neither top-5 column stitched page parts before dedup the way
`search_specs` does. Every row above the last was printed by that harness, so those numbers are a
faithful record of what the runs printed and **not** measurements of the shipped path.

What the correction cost, per question, against the c183eb2 run:

- **The ColBERT column is unchanged, rank for rank (11/16).** Its only change was the stitch, so on
  this corpus stitching is rank-neutral: it merges page parts into richer excerpts without moving any
  document into or out of the top 5.
- **Hybrid lost exactly two questions, q02 and q14 (14 → 12), and the filter is the only candidate.**
  That column's request shape was otherwise identical and the stitch is neutral, so nothing else could
  remove a hit. The corpus shows the mechanism: `Mm3` (q02) occurs in **5 superseded** MEC-003 chunks
  against 3 current ones, and q14's keywords in **53 superseded** chunks against 23 current. The old
  column was crediting retrievals from editions `search_specs` refuses to return — the drop is the
  measurement becoming honest, not the retriever getting worse.
- **Aggregate moved 14 → 12, and that delta is *not* attributable.** Two changes hit this column at
  once (the filter, and the pool going 100 → 180 per leg); it gained q05 and lost q08, q09 and q13.
  Separating them needs an ablation nobody has run.
- **The three aggregate losses are the finding worth keeping.** q08, q09 and q13 sit at **@1 in both
  top-5 columns** and still miss the 15-excerpt answer budget: the shipped `--answer` path loses
  evidence the terminal display holds at rank 1. A deeper candidate pool reshuffles which documents
  fill the 15 slots, and `--per-doc 2` spends them across more specs. This was invisible while the
  column under-filled its pool.

The corrected hybrid column is a superset of the ColBERT column, with q07 the only hybrid-exclusive
hit, so union is 12/16. The 26 Sep start row (union 12/16) was printed by the old harness, so **no
delta in the table above is a like-for-like comparison** — there is no corrected-harness baseline for
the word-window collection. Creating one means re-measuring `etsi_mec_specs_v2` under the corrected
harness, which needs `scripts/label_collection_identity.py --apply` on that collection first because it
predates the identity guard.

`aggregate@15` counts the questions whose evidence survives the 15-excerpt budget that is actually
handed to the LLM — the metric that decides answer quality, unlike recall@5.

Two claims previously recorded here are retracted by the corrected run: net movement
**11/16 → 14/16 (+27%)** with union 12/16 → 15/16, and the rescore's own effect (hybrid@5 12→14,
union 15→14, q12 leaving the hybrid top-5 while still reaching the LLM at @4). Both were old-harness
numbers. Under the corrected harness hybrid@5 is 12, union is 12, and q12 reaches the answer budget
at @3.

That also removes the nested ColBERT rescore's only recorded justification: the **+2 hybrid@5 it was
credited with in c183eb2 was the filter's superseded editions, not the rescore**. As of this row the
rescore has no measured recall benefit — establishing one needs an A/B against the un-rescored shape
under the corrected harness, which has not been run. What survives from those paragraphs is the
reading rule, not the values: n = 16, so ±1 or ±2 is a one-question swing, not a trend.

The limits of that number, stated before someone finds them: **n = 16**, so one question is 6
points. Three goldens were rewritten mid-project because they were unanswerable as written (q12's
keyword `Mm5` occurred in 56 chunks; `mpInfoService` in 0), and the old collection was re-measured
under the new yardstick so the comparison would mean something:

| Collection | colbert | hybrid | aggregate@15 | union |
|---|---|---|---|---|
| word-window chunks, 4,669 | 12 | 12 | 14 | 16/16 |
| row/clause chunks, 5,254 | 11 | 12 | 14 | 15/16 |

Both rows are old-harness (2026-10-08): unfiltered hybrid/aggregate, a 100-per-leg answer pool, no
stitch in the top-5 columns. They still answer the question they were run for — the two collections
under one yardstick on one day — but neither number is comparable to the corrected row above.

Aggregate is tied. The row/clause chunker was chosen anyway because a table row split across two
chunks produces an answer with a cut cell, which recall cannot see — that is a correctness property
the metric does not score.

## 2. Index and corpus

    curl -s http://localhost:6333/collections/etsi_mec_specs | jq .result.points_count
    uv run python scripts/check_eval_golden.py

54 PDFs · **42 `spec_id`s with exactly one current edition each** · 5,254 chunks (2,309 table,
2,955 prose) · 689,725 words of current text · 3,068 extracted diagrams (370 MB) · zero duplicate
`(spec_id, edition, page, chunk_part)` keys.

Corpus defects found and fixed, which is most of what the work was:

- **106 files were 54 documents.** 52 byte-identical PDFs removed (51 MB) by
  `scripts/dedupe_spec_pdfs.py`, keeping whichever copy the live index's payload pointed at.
- **59 of 106 filenames named a spec whose text they did not hold** (`MEC041.pdf` is GS MEC 040).
  Identity is now read off the PDF cover (`identity.py`) — `spec_id`, `edition`, `pub_date`,
  `content_md5` — and the filename never enters the prompt.
- **1,188 superseded-edition chunks** became unreachable by default (`--all-editions` overrides).
- **10 chunks were pure ETSI page furniture** (running title, `_ETSI_` footer, bare page number) and
  one of them ranked **second** for a real query. Refused at chunk time and deleted: 5,264 → 5,254.
- The clause floor took the index from **9,854 to 5,254 chunks** with **p90 parts/page 46 → 2** —
  fewer, whole chunks instead of one-chunk-per-row confetti.

## 3. Retrieval latency

25 hybrid queries against the live collection (`is_current` filter, payload returned), random
384-d dense vector + 24-term sparse (+ a random 4×128 ColBERT multivector once the rescore existed),
timed client-side:

| Measured | shape | p50 | p95 | min | max |
|---|---|---|---|---|---|
| 2026-10-11 | two prefetches + RRF, per-leg 50, 15 points | 15.3 ms | 31.3 ms | 9.2 ms | 290.6 ms (cold) |
| 2026-11-07 | same legs + ColBERT MaxSim rescore, 15 points | 27.1 ms | 70.8 ms | 12.5 ms | 174.3 ms |
| 2026-11-07 | **non-default** shape: per-leg 200, pool 30, + rescore | 31.0 ms | 38.6 ms | 11.0 ms | 50.1 ms |

The rescore pass is the delta between the first two rows — roughly +12 ms p50 at identical leg
limits.

The third row is **not** what the shipped default sends. `--use-bm25 --top-k 5` without `--answer`
asks for 100 candidates per leg (`max(prefetch_limit * 4, 100)`, `search.py:514`) into a 30-point
pool (`fetch_k * 6`, `search.py:513`); the probe ran 200 per leg, so that p50 belongs to a larger
request than the default. The default shape is unmeasured, and no probe script is committed — all
three rows are records of runs that happened, not reproducible benchmarks.

This is Qdrant-side only. It excludes the ONNX embedding of the query text and the LLM round-trip,
so it is not an end-to-end latency claim — the honest statement is "retrieval is tens of
milliseconds", not "sub-second".

## 4. Answer integrity

- Excerpts are labelled `MEC-003 V4.1.1 p.15` from the stamped identity; the stored filename is
  never shown to the model, because in this corpus it names the wrong spec.
- **Every page citation in the finished answer is verified** against the retrieved set: `p.15`,
  `pp. 15`, prose `page 6`, attributed to the nearest spec code on either side, and separated into
  "that spec was never retrieved" versus "no excerpt of that spec sits on that page".
- Answer text is flattened from typographic Unicode (U+2011 non-breaking hyphen, U+202F narrow
  no-break space) before matching. The first live run produced `ETSI GS MEC 003` and `MEC‑059` in
  those codepoints, and an ASCII-only matcher reported two **correct** citations as fabricated —
  the audit was the bug, not the answer.
- Replayed against the 2026-10-08 run: **7 page citations, all grounded**.
- A no-citation answer under 40 words is reported as a canned router response rather than printed
  as an answer; `openrouter/free` returned the literal string `User Safety: safe` for a real query
  on 2026-10-08. The serving model ID is printed because the free router is not deterministic.

## 5. Verification

No test framework and no fixtures — 9 runnable checks, **296 assertions**, that import the
**production** symbols with `fastembed` stubbed, so they run in seconds on a 16 GB CPU host
without loading a model:

| Check | Assertions | Guards |
|---|---|---|
| `check_embedder_swap.py` | 62 | the provider seam, the identity guard at all three entry points, the warm-up that fails a recreate *or* an incremental ingest before the batch loop can swallow it, the migration's inherited identity |
| `check_answer_grounding.py` | 55 | excerpt labels, citation audit, `generate_answer` wiring with `openai` faked |
| `check_dedup_stitch.py` | 53 | stitching, per-doc quota, whole-text fingerprint |
| `check_app_ui.py` | 42 | the Streamlit screens, password gate and citation rendering |
| `check_chunking.py` | 23 | rows never cut, headers repeated, clause floor, empty-window and furniture filters |
| `check_download_naming.py` | 20 | cover naming, edition in name, duplicate refusal, dedupe keepers, unreadable body, URL-vs-cover disagreement |
| `check_eval_golden.py` | 17 | every golden answerable from the live collection; the eval's request shape, stitch order and guard |
| `check_migrate_sparse.py` | 13 | migration rebuilds each sparse vector with the cover identity ingest prefixes |
| `check_diagram_filter.py` | 11 | blank/tiny/duplicate images, per-document dedup |

    for f in scripts/check_*.py; do uv run python "$f"; done

Assertion counts are static `assert` statements in each file (some run more than once, some inside
loops), counted the same way on 2026-11-07 as in the "169 assertions" figure this replaced. That
figure was already wrong before this branch: it listed 7 of the 9 checks, omitting `check_app_ui.py`
(added in PR #4) and counting `check_answer_grounding.py` at its pre-2026-10-11 size.
`check_app_ui.py` and `check_eval_golden.py` read the live collection and need Qdrant running. No
check reads `data/specs/`, which is gitignored and therefore absent from a clean checkout: the ingest
checks in `check_embedder_swap.py` write a blank one-page PDF into a temp directory, because both
abort before anything is parsed.

## 6. Process

52 of the repository's then-56 commits landed on the build branch over 16 days (26 Sep – 11 Oct),
+4,434/−416 across 31 files, 4,726 lines of Python at that date. Measured at `90fd0b5` the repository
has grown to **86 commits and 6,507 lines of Python**, and the 10 commits on
`feature/embedding-providers` are +1,218/−155 across 16 files, 12 of them Python (+1,110/−138). A
405-line design record (`docs/superpowers/specs/`) with a measured before/after for each change,
including the ones that did not help. [PR #1](https://github.com/geokal/etsi-mec-agent/pull/1) took 7
findings from an
automated review: 6 accepted as found, 1 accepted-and-extended with the reason the suggested fix was
insufficient — and **two of the seven were defects this project had introduced itself** (the 400-char
dedup fingerprint, the run-wide image-dedup set), which is the more useful signal about how the code
got to this state. Two follow-up review passes brought the total to ten: the cover parsers sat
outside the download guard in both fetchers, so one unparseable PDF ended the monitor for every
remaining spec; the migration re-pointed the alias only after deleting the collection that held it,
leaving the searched name pointing at nothing across a crash; and nothing checked the served
document's cover against the number its URL claimed — the failure that produced `MEC041.pdf` holding
GS MEC 040. Each was verified against the code, the live corpus or the server before being changed;
one had already been fixed by the pass before it, and one was applied differently than asked because
the literal instruction would have broken new-spec discovery. A third reviewer (CodeRabbit) then
found an eleventh, in the same migration script: it rebuilt each point's sparse vector from the
stored body alone, dropping the `spec_id edition clause heading` prefix that ingest tokenises into
it, so a migrated chunk would not match a bare clause-number query the way a freshly ingested one
does. The rebuild now reassembles that prefix from the payload's identity fields, pinned by
`check_migrate_sparse.py`; the migration is historical (`sparse` is already declared on the live
collection), so this is verified by the model-free check and by reading, not by a run against the
storage volume.

[PR #5](https://github.com/geokal/etsi-mec-agent/pull/5) — the embedding-provider seam — brought
findings twelve through twenty: **nine findings, one nitpick and one failed pre-merge check**
(docstring coverage, 83/83 documented after the round). All nine were verified valid against the live
code before anything changed and all are fixed in `9d7bae7`. Two
were defects this PR introduced itself, and both sit in the identity work: `ensure_collection`
resolved the identity it was about to write *after* it deleted the collection it was recreating, and
`migrate_add_sparse.py` called it with no way to carry the source's claim, so a copy of a
local-embedded collection made under `EMBEDDING_MODE=cloud` would have been labelled as something
its vectors are not. Neither could occur on master, where `ensure_collection` takes no `metadata`
and records no identity at all. Three more were defects the eval harness has carried since before
this branch — `hits_hybrid` and `hits_aggregate` never sent `build_filter()`, the aggregate column
filled its pool with a flat 100 candidates per leg instead of `search_specs`' 180, and neither top-5
column stitched page parts the way `search.py:517→528` does — confirmed by reading
`git show master:scripts/eval_rag.py`, so the correction is not a regression this PR introduced. It
is the reason every recall row in §1 above the 2026-11-07 corrected one is labelled: those numbers
measured a retriever that saw superseded editions and unstitched pages, which the shipped path does
not. Two separate review bot instances covered this repository and **disagreed** — only one flagged
the missing stitch, only one named `eval_rag` as a third guard entry point, only one raised the
stale RESULTS.md row — so a clean report from one is not a clean diff.

The re-review of `9d7bae7` found two more (findings twenty-one and twenty-two, both fixed in
`90fd0b5`), and one of them was
that round's own fix applied only halfway: the warm-up that keeps a model-load failure from being
swallowed by the per-chunk fallback ran **only** under `--recreate-index`, so an incremental ingest
against a provider that would not construct still printed one `[SKIPPED]` per chunk and exited 0
having indexed nothing. Deleting the warm-up reproduces exactly that — the gate now prints
`[COMPLETED] Indexed 0 chunks` and fails on its own assertion. The other was that
`check_embedder_swap.py` required `data/specs/MEC001.pdf`, which is gitignored, so the gate could not
run on a clean checkout at all; it now writes its own blank one-page PDF and passes in a directory
with no `data/`.

## 7. What this project does not have

No CI, no containerised app, no packaging. Licensed MIT (2026-10-11) over the code in this
repository — the ETSI documents themselves are copyrighted by ETSI, are not committed here
(`data/specs/` is gitignored), and are fetched by `scripts`/`tools` from ETSI's own server. One
author, no users, no traffic, no deployment. **Answer quality is not benchmarked — retrieval is**,
and the LLM is a free-tier router. The `batch_size` 4 → 16 ingest speedup is a code comment, not a
measurement. As of the corrected-harness run of 2026-11-07, four questions are missed by both top-5
paths — q02, q12, q14 and q15 — and three of those still reach the 15-excerpt answer budget (q12 at
@3, q02 at @6, q14 at @7), so the LLM sees evidence the display never shows. q15 reaches nothing on
any path. The mirror image is worth as much: **q08, q09 and q13 rank @1 in both top-5 columns and
MISS the answer budget**, which is a `--per-doc 2` / 15-excerpt quota effect, not a ranking failure.
The earlier causes are recorded in §5b.
