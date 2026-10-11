# Measured results

Every number below has a date and a command. Where a number cannot be reproduced from this
repository, it says so. Single author, 26 Sep – 11 Oct 2026.

## 1. Retrieval quality — 16 golden engineering questions

    uv run python scripts/eval_rag.py --top-k 5

| Measured | colbert@5 | hybrid@5 | aggregate@15 | union | What changed |
|---|---|---|---|---|---|
| 2026-09-26 (start) | 10 | 10 | — | 12 | word-window chunks, filename-based identity |
| 2026-10-06 (§5b) | 10 | 10 | 11 | — | rank shaping: part-stitching, per-document dedup, answer budget |
| 2026-10-06 (§5d) | 11 | 11 | 12 | — | cover-derived identity, edition filtering, `content_md5` quota |
| 2026-10-08 (§5e) | 11 | 12 | **14** | 15 | row-/clause-aware chunking with the clause floor |
| 2026-10-11 (§5i) | 11 | 12 | **14** | 15 | review-round fixes — identical, i.e. rank-neutral |

`aggregate@15` counts the questions whose evidence survives the 15-excerpt budget that is actually
handed to the LLM — the metric that decides answer quality, unlike recall@5. Net movement:
**11/16 → 14/16 (+27%)**, union 12/16 → 15/16.

The limits of that number, stated before someone finds them: **n = 16**, so one question is 6
points. Three goldens were rewritten mid-project because they were unanswerable as written (q12's
keyword `Mm5` occurred in 56 chunks; `mpInfoService` in 0), and the old collection was re-measured
under the new yardstick so the comparison would mean something:

| Collection | colbert | hybrid | aggregate@15 | union |
|---|---|---|---|---|
| word-window chunks, 4,669 | 12 | 12 | 14 | 16/16 |
| row/clause chunks, 5,254 | 11 | 12 | 14 | 15/16 |

Aggregate is tied. The row/clause chunker was chosen anyway because a table row split across two
chunks produces an answer with a cut cell, which recall cannot see — that is a correctness property
the metric does not score.

## 2. Index and corpus

    curl -s http://localhost:6333/collections/etsi_mec_specs | jq .result.points_count
    uv run python scripts/check_eval_golden.py

54 PDFs · **42 `spec_id`s with exactly one current edition each** · 5,254 chunks (2,309 table,
2,955 prose) · 689,731 words of current text · 3,068 extracted diagrams (370 MB) · zero duplicate
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

25 hybrid queries against the live collection (two prefetches + RRF fusion, `is_current` filter,
15 points with payload returned), random 384-d dense vector + 24-term sparse, timed client-side:

**p50 15.3 ms · p95 31.3 ms · min 9.2 ms · max 290.6 ms** (the max is the first query, cold).

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

No test framework and no fixtures — 6 runnable checks, **156 assertions**, that import the
**production** symbols with `fastembed` stubbed, so they run in seconds on a 16 GB CPU host
without loading a model:

| Check | Assertions | Guards |
|---|---|---|
| `check_dedup_stitch.py` | 53 | stitching, per-doc quota, whole-text fingerprint |
| `check_answer_grounding.py` | 48 | excerpt labels, citation audit, `generate_answer` wiring with `openai` faked |
| `check_chunking.py` | 23 | rows never cut, headers repeated, clause floor, empty-window and furniture filters |
| `check_download_naming.py` | 15 | cover naming, edition in name, duplicate refusal, dedupe keepers |
| `check_diagram_filter.py` | 11 | blank/tiny/duplicate images, per-document dedup |
| `check_eval_golden.py` | 6 | every golden answerable from the live collection |

    for s in chunking dedup_stitch diagram_filter download_naming answer_grounding eval_golden; do uv run python scripts/check_$s.py; done

## 6. Process

52 of the repository's 56 commits landed on this branch over 16 days (26 Sep – 11 Oct), +4,434/−416
across 31 files, 4,726 lines of Python in total. A 405-line design
record (`docs/superpowers/specs/`) with a measured before/after for each change, including the ones
that did not help. [PR #1](https://github.com/geokal/etsi-mec-agent/pull/1) took 7 findings from an
automated review: 6 accepted as found, 1 accepted-and-extended with the reason the suggested fix was
insufficient — and **two of the seven were defects this project had introduced itself** (the 400-char
dedup fingerprint, the run-wide image-dedup set), which is the more useful signal about how the code
got to this state.

## 7. What this project does not have

No CI, no containerised app, no packaging. Licensed MIT (2026-10-11) over the code in this
repository — the ETSI documents themselves are copyrighted by ETSI, are not committed here
(`data/specs/` is gitignored), and are fetched by `scripts`/`tools` from ETSI's own server. One
author, no users, no traffic, no deployment. **Answer quality is not benchmarked — retrieval is**,
and the LLM is a free-tier router. The `batch_size` 4 → 16 ingest speedup is a code comment, not a
measurement. q11 and q12 are still missed by at least one path each, with the causes recorded in §5b.
