# Corpus-wide answering for the ETSI MEC agent

Date: 2026-10-06 · Branch: `fix/diagram-capture` · Status: approved, pre-implementation
Decisions taken: current edition wins while aggregating across documents; duplicates skipped at
ingest with source files left untouched; repair the downloader; build order A then B;
`per_doc=2`, `--answer-context 15`.

## 1. Goal

`search --answer` should read everything in the corpus that bears on a question and say so with
citations that name the right document and edition — instead of reading five excerpts and calling
that comprehensive.

## 2. Evidence this is a real problem, measured 2026-10-06

| Measurement | Value |
|---|---|
| PDFs in `data/specs` | 106 files, **54 unique by md5** (17 duplicate groups; 16 copies of GS MEC 009, 10 of GR MEC 059) |
| Distinct spec numbers on disk | **33**, range 001–062, with **29 numbers never fetched** by content, whatever the filenames suggest (MEC004, 006, 007, 008, 012, 014, 019, 020, 022, 023 …) |
| Manifest keys whose URL names that key | 31 of 96 — the rest were filed under a number their URL does not declare |
| Points in the live collection | 4,669, of which 957 (~20 %) sit under a `doc_id` belonging to another document |
| Chunk shape | median 243 words, cap 300, **97 % end with no sentence-final punctuation** |
| Golden-question recall@5 | colbert 10/16, hybrid 10/16, union 12/16; q06, q08, q12, q14 missed by both |
| Encoder ceiling | `max_position_embeddings: 512` in both cached models' `config.json` |

Mechanical causes, each traced to code:

1. `search.py::_dedup_docs` keeps one chunk per `doc_id`; the continuation of a page retrieved as
   `chunk_part 1 of 2` is discarded, so retrieved tables end mid-cell (`|Mm9:|The Mm9 reference`).
2. `ingest.py::chunk_page_text` slices `text.split()` by word count with no notion of line, row or
   clause, so boundaries land inside cells.
3. `tools/monitor_deliver.py` saved whatever PDF an Exa lookup returned under the requested spec
   number. Repaired in `85d7aa0`: files are now named from the identity their URL declares, and the
   response must start with `%PDF`.
4. Several editions of one spec are indexed together (GS MEC 003 V2.2.1, V3.1.1, V4.1.1) and the
   newest is filed as `MEC023.pdf`/`MEC070.pdf`, so answers blend superseded text.

## 3. Design A — aggregation at query time (no re-ingest)

Data flow: query → retrieve (hybrid or colbert path) → **stitch** → **cap** → terminal shows
`--top-k`, LLM reads `--answer-context`.

**A1 `_stitch_parts(docs)`** — merges only *contiguous* `chunk_part`s sharing `doc_id`+`page`
(1+2, 2+3). Because `chunk_page_text` uses a 40-word overlap, part 2 repeats the tail of part 1: the
merge trims the duplicated leading tokens by longest-suffix/longest-prefix match. Non-contiguous
parts stay separate — joining 1+3 would fabricate a page that was never retrieved. Points with no
`chunk_part` (or a malformed one) are treated as a single part and pass through unchanged.

**A2 `_dedup_docs(docs, keep, per_doc=1)`** — Pass 1 (400-char text fingerprint) unchanged; it is
what currently collapses mislabelled copies. Pass 2 keeps up to `per_doc` stitched excerpts per
`doc_id`, total capped at `keep`. Default `per_doc=1` preserves every existing caller's behaviour,
including `eval_rag.py`'s two baseline columns.

**A3 `--answer-context N` (default 15)** — decouples evidence budget from terminal output. Retrieval
fetches `N*6` candidates (90 at N=15) with `prefetch_limit` kept at >=100, the pattern the hybrid
path already uses; both paths call the same stitch→cap function.
Printed results remain `--top-k`.

**A4 Settings** — `per_doc=2`, `--answer-context 15` as the `--answer` default (≈3.5k words).
Exposed as `--per-doc N` so the numbers can be re-measured rather than trusted.

**A5 Verification**
- `scripts/check_dedup_stitch.py` (model-free, runnable here): contiguous parts merge with the
  overlap removed; 1+3 does not merge; `per_doc` and `keep` caps hold; parts missing `chunk_part`
  pass through; a stitched excerpt contains a row that appeared only in part 2.
- `scripts/eval_rag.py` gains an `aggregate` column (`per_doc=2, keep=15`) beside the existing two.

## 4. Design B — structure-aware ingest

Second stage: A is built, measured and accepted on its own, and B's criteria do not depend on any
change A makes beyond the shared stitch→cap function.

**B1 Content-derived identity.** Read the title page (`ETSI GS MEC 010-02 V4.1.1 (2025-05)`) into
payload fields `spec_id`, `series` (GS/GR/FGS), `spec_version`, `edition_date`, plus
`source_filename`; `doc_id` becomes `spec_id`. Per-run md5 set skips byte-identical files with a log
line. Unidentified titles (two slide decks, an app-dev document) keep the filename as `spec_id` and
get a null `edition_date` rather than a guessed one.

**B2 Block-aware chunking.** Page markdown is parsed into blocks: headings, prose, tables (runs of
`|`-lines), image refs. Prose keeps 300 words/40 overlap but never breaks a table row. Tables emit
**row groups** each carrying the clause reference, the table caption and the header row, sized by a
token estimate rather than a word count: markdown pipes and `<br>` runs tokenize as their own
tokens, so 380 words of table text can blow the 512-token ceiling that prose fits. The prototype run
reports the largest real chunk so the bound is measured once instead of guessed. Payload gains `clause` and `block_kind` ∈ {prose, table}.

**B3 Document-side contextualization.** The embedded text becomes `spec_id + clause + heading + body`;
query text stays raw. Asymmetric on purpose — the document side gets the context a clause number
needs, the query side has nothing to add.

**B4 Current-edition marking.** `scripts/mark_current_edition.py` scrolls the collection, groups by
`spec_id`, sets `is_current=true` only on the newest `edition_date` (ties → newest
`spec_version`). Done as a post-ingest pass because the newest edition is only knowable once
everything is ingested. `search_specs` filters `is_current != false` by default — negated so undated
documents survive — and `--all-editions` drops the filter. A `create_payload_index` is added for
`is_current` (BOOL) and `spec_id` (KEYWORD).

**B5 Verification and rollout.** `scripts/check_chunking.py` (model-free): no chunk ends inside a
table cell; every table chunk repeats its header row; a 30-row synthetic table reconstructs
completely from its chunks; prose chunks ≤380 words. Prototype run over 3 specs
(`MEC003`/`MEC030`/`MEC002`) into `QDRANT_INDEX=etsi_mec_prototype` — env var already supported —
with a new `--show-chunks N` that prints candidate chunks. `ingest.py:13` imports fastembed at
module level, so every ingest run including the prototype is executed by the user in their own
terminal.

## 5. Acceptance criteria

A: `check_dedup_stitch.py` passes; baseline `recall@5` columns unchanged at 10/16; the `aggregate`
column converts **at least two** of q06/q08/q12/q14 from MISS to a rank; a `--answer` run on
"Mm3/Mm5" quotes a complete reference-point table rather than ending mid-row.

B: prototype chunks contain no cut cells; after the full re-ingest, distinct indexed `spec_id`
count equals unique-document count (no alias documents), `is_current` true for exactly one edition
per `spec_id`, and eval `aggregate` recall ≥ 12/16 with every cited `spec_id` verifiably containing
the quoted text.

## 5b. Result of A, measured 2026-10-06

`uv run python scripts/eval_rag.py --top-k 5` on the live collection:
`recall@5: colbert=10/16 (62%)  hybrid=10/16 (62%)   aggregate@15: 11/16 (69%)`.

A gained q03 (which the hybrid path missed) and lost q11, because the aggregate column is built on
the hybrid path and cannot rank what that path never retrieves. Union across the three columns is
still 12/16. **The acceptance bar was not met:** q06, q08, q12 and q14 remain MISS under all three.

Corpus scan of those four explains why, and it is not a ranking problem:

- **q12 is bad golden data.** Its keyword `Mm5` occurs in **56** chunks, so any of them "passes"
  evidence while no ranking can single out the selection API. The question needs a discriminating
  keyword before it can measure anything.
- **q06 / q08 / q14: evidence is fragmented or mislabelled.** 14, 8 and 51 chunks contain their
  keywords; the strongest matches are page fragments (`MEC003 p23 part 1/2`, `MEC070 p26 part 1/2`,
  `MEC083 p54 part 2/3`) or belong to documents whose `doc_id` names a different spec (MEC070 is a
  copy of GS MEC 003 V4.1.1). This is B1 (identity) and B2 (row-group chunks), not a wider `k`.
- The scan also found `MEC021 p14 part 3/3` holding **14 words** — the degenerate tail chunk
  predicted from `chunk_page_text`'s slicing, confirming B2.

Conclusion: A improved the shape of what is retrieved (ranks and complete passages) and one net
hit, and the four stubborn misses are precisely the ones B targets. B proceeds, with q12 rewritten
first so the next measurement is honest.

## 5c. End-to-end proof (`--answer`), run 2026-10-06

`uv run python -m etsi_mec_agent.search "Which reference point connects the MEC platform to the MEC
orchestrator?" --use-bm25 --answer --answer-context 15 --per-doc 2` printed
`[ANSWER CONTEXT] 15 excerpts, up to 2 per document (~5169 words), 9 page-part group(s) merged`, and
every reference-point row arrived whole — the pre-change run's `|Mm9:|The Mm9 reference` truncation is
gone. **A's last acceptance clause passes.**

The answer is still wrong, and the cause is B1, not A. It says **Mm5** and justifies it with "the MEC
orchestrator (referred to as the MEC Platform Manager in ETSI GS MEC 003)". The corpus contradicts that
in every edition it holds: `Mm3` = MEO ↔ MEC platform manager, `Mm5` = MEC platform manager ↔ MEC
platform (read from the stored payloads at `MEC003` p16, `gs_MEC_003_v2.2.1` p15, `MEC070` p17–18). No
edition defines a platform↔orchestrator reference point, so the defensible answer is "none — management
reaches the platform through the platform manager".

Why the model could not tell them apart: four of the five shown excerpts are the *same table* taken from
**three editions across four `doc_id`s** — `MEC003` (V3.1.1), `gs_MEC_003_v2.2.1` (V2.2.1), and
`MEC023` + `MEC070`, which md5 identically (`cf0052ad…`) and are both GS MEC 003 **V4.1.1** under
filenames that name other specs. `--per-doc` caps by `doc_id`, so byte-identical documents stored under
different names each claim their own slots; the 15-excerpt budget is spent on near-duplicate tables and
the one distinguishing row is outvoted by three differently-worded copies. This is B1 (spec identity:
dedupe by content, one `is_current` edition per `spec_id`) showing up as a wrong answer rather than as
a low recall number.

## 5d. Result after B1 identity stamping, measured 2026-10-06

Same command, same 16 questions, on the collection `scripts/backfill_spec_identity.py` stamped
(4669 chunks: 3481 `is_current=true`, 1188 superseded) and `search.py::build_filter` now filters by
default, with `_dedup_docs` spending the per-doc quota per `content_md5`:

`recall@5: colbert=11/16 (69%)  hybrid=11/16 (69%)  aggregate@15: 12/16 (75%)`, every question
reporting `evidence ok`.

Against §5b that is colbert 10→11, hybrid 10→11, aggregate 11→12, while 1188 chunks became
unreachable and another 957 collapse into a shared quota — recall rose because the duplicates and
stale editions stopped outvoting the discriminating row. Union coverage went 12/16 → 14/16: only q08
and q14 are missed by all three paths.

- **q06 moved MISS/MISS → aggregate @2.** Rewritten around `Nnef_TrafficInfluence` (the "TCR" it
  asked about occurs in no chunk of the corpus), it is the one question A's acceptance bar named and
  the stitch mechanism is what surfaced it.
- **q12 moved the other way:** rewritten around "selects the MEC host", a phrase in 2 chunks, it is
  colbert @3 and MISS on both hybrid paths.
- **q08 and q14 stay MISS everywhere, and they are now genuinely retrieval failures:** their keywords
  sit in current MEC-003 V4.1.1 (`LCM proxy` in 5 current chunks, `service registration` in 18,
  `service discovery` in 27). This is B2 row/clause chunking, not golden data and not a wider `k`.
- **A's bar is still not met:** the criterion was two of q06/q08/q12/q14 converting, and only q06 did.

The end-to-end proof changed character, though. The same Mm3/Mm5 question that produced "Mm5, because
the MEC orchestrator is referred to as the MEC Platform Manager" now answers **Mm3**, cites it to
GS MEC 010-2 V4.1.1 with the table quoted whole, and distinguishes it from Mp1. The corpus holds no
reference point between the MEC platform and the orchestrator, so naming the manager-mediated path is
the best answer available — and it is attributed to the spec the text really comes from.

---

## 6. Out of scope here

- Replacing the parser with Docling/unstructured (approach C): new heavy dependencies on a 16 GB
  host, slower ingest, and B already delivers row integrity. Revisit only if B's eval shows table
  fidelity is still binding.
- Sending diagram **pixels** to the model. Today's answer cites figure filenames it cannot read;
  multimodal answering needs a pinned vision model, which AGENTS.md forbids changing without an
  explicit request, plus VLM captioning at ingest.
- Re-downloading the 29 spec numbers never fetched. The downloader is repaired; fetching is a
  network run the user triggers separately.
- Cross-document diagram dedup (hash→path) so overview decks keep reprinted figures.

## 7. Risks

Free-tier routing (`openrouter/free`) may degrade on a ~3.5k-word prompt; if answers get vaguer
while recall rises, the fix is a pinned model or map-reduce, not a smaller context. Re-ingest
replaces the live collection: take the existing volume-backup route first and keep the
`etsi_mec_specs_v2` + alias pattern so a failed run is one alias call from recovery.
