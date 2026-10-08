"""Page markdown -> chunks that never cut a table row and never lose a table header.

Stdlib-only by design, like `dedup.py`: `scripts/check_chunking.py` asserts on this without
importing `ingest.py`, which pulls fastembed (and ONNX runtime) at module level.

`chunk_page_text`'s word window did that job badly: slicing by words flattened every line of
markdown into one long space-separated run, so a 48-pipe reference-point table arrived as pipe
soup that could end mid-cell.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_ROW_RE = re.compile(r"^\s*\|")
_SEP_RE = re.compile(r"^\s*\|[\s:|-]+$")
_HEADING_RE = re.compile(r"^\s*#+\s*(.*)$")
_CLAUSE_RE = re.compile(r"\d+(?:\.\d+)+")
_IMG_RE = re.compile(r"^\s*!\[.*?\]\(.*?\)\s*$")


@dataclass(frozen=True)
class PageChunk:
    """One candidate excerpt: its markdown, whether it is table or prose, and its clause."""

    text: str
    block_kind: str
    clause: str


def estimate_tokens(text: str) -> int:
    """Words plus the markdown punctuation that tokenizes as its own tokens.

    ColBERT's passage ceiling is 512 tokens and 380 words of table text can blow it, because
    every pipe and `<br>` is a token of its own. This is an estimate, not the real tokenizer:
    the bound it enforces is deliberately below the ceiling (see `max_tokens`).
    """
    return (
        len(text.split())
        + text.count("|")
        + 2 * len(re.findall(r"<br\s*/?>", text, re.IGNORECASE))
    )


def _table_units(lines: list[str]) -> list[tuple[str, str]]:
    """(row, header) for one run of `|` lines, where header is the rows above its separator.

    A run can hold several sub-tables; each `|---|---|` closes the header above it, so the row
    that names a clause (`|7.2.2|Reference points related to the MEC management|`) becomes the
    header of what follows instead of a body row.
    """
    units, pending, header = [], [], ""
    for i, line in enumerate(lines):
        next_is_sep = i + 1 < len(lines) and _SEP_RE.match(lines[i + 1])
        if _SEP_RE.match(line):
            header = "\n".join(pending + [line])
            pending = []
        elif next_is_sep:
            pending.append(line)
        else:
            units.append((line, header))
    units.extend((p, header) for p in pending)
    return units


def _units(text: str) -> list[tuple[str, str, str, str]]:
    """(kind, body, header, clause) in document order; a row or paragraph is never split here."""
    units: list[tuple[str, str, str, str]] = []
    clause = ""
    para: list[str] = []
    rows: list[str] = []

    def flush_para():
        if para:
            units.append(("prose", "\n".join(para), "", clause))
            para.clear()

    def flush_rows():
        nonlocal clause
        if not rows:
            return
        for body, header in _table_units(rows):
            # A sub-table is introduced by a row whose first cell is only a clause number
            # (`|7.2.2|Reference points related to the MEC management|`), and that row lands in
            # the header rather than the body, so both have to be asked.
            for line in ((header.splitlines() or [""])[0], body):
                cell = line.strip().strip("|").split("|")[0].strip()
                if _CLAUSE_RE.fullmatch(cell):
                    clause = cell
            units.append(("table", body, header, clause))
        rows.clear()

    for line in text.splitlines():
        if _ROW_RE.match(line):
            flush_para()
            rows.append(line)
            continue
        flush_rows()
        stripped = line.strip()
        if not stripped:
            flush_para()
            continue
        heading = _HEADING_RE.match(line)
        if heading:
            flush_para()
            found = _CLAUSE_RE.search(heading.group(1))
            if found:
                clause = found.group(0)
            units.append(("prose", line, "", clause))
            continue
        if _IMG_RE.match(line):
            flush_para()
            units.append(("prose", line, "", clause))
            continue
        para.append(line)

    flush_para()
    flush_rows()
    return units


# ETSI's per-page furniture, which says nothing about the page it sits on.
_FOOTER_RE = re.compile(r"^_+ETSI_+$")
_HEADER_RE = re.compile(r"^ETSI\s+(?:GS|GR|TS|ISG)\s+MEC.*\bV\d+\.\d+\.\d+\b", re.IGNORECASE)
_PAGENO_RE = re.compile(r"^\d{1,4}$")
# No underscore in this class: it would eat the emphasis marks of the `_ETSI_` footer and leave the
# word ETSI behind looking like content.
_MARKUP_RE = re.compile(r"^[\s#*>`~.-]+|[\s#*>`~.-]+$")


def _carries_signal(text: str) -> bool:
    """Does any line of this chunk say something once the page furniture is removed?

    An image reference counts as signal: a figure-only page is exactly what --diagrams-only is for.
    """
    for line in text.splitlines():
        if "![" in line:
            return True
        bare = _MARKUP_RE.sub("", line.strip())
        if bare and not (_FOOTER_RE.match(bare) or _HEADER_RE.match(bare) or _PAGENO_RE.match(bare)):
            return True
    return False


def chunk_page(text: str, max_words: int = 300, overlap: int = 40, max_tokens: int = 420,
               min_words: int = 150) -> list[PageChunk]:
    """Pack indivisible units into chunks, cutting only between rows or paragraphs.

    A page whose text is nothing but ETSI's per-page furniture (footer, running title, page number)
    yields no chunk.

    `max_tokens` is what ColBERT's 512-token passage limit actually constrains; `max_words`
    stays for prose because that is the size the rest of the pipeline was measured at. A table
    row longer than both limits is kept whole rather than sliced: cutting a row is cutting a
    cell, and ingest's per-chunk fallback skips the rare giant instead.

    `min_words` is the floor that stops a clause change from shattering a dense table: measured
    on the re-ingested corpus, splitting at every clause left 4456 chunks sitting on pages broken
    into more than 8 parts, 15-word fragments that neither rank nor stitch back usefully.
    """
    def fits(candidate: str) -> bool:
        return len(candidate.split()) <= max_words and estimate_tokens(candidate) <= max_tokens

    chunks: list[PageChunk] = []
    cur: list[tuple[str, str, str]] = []   # (piece, kind, clause of the unit it came from)

    def pieces():
        return [p for p, _, _ in cur]

    def close():
        if not cur:
            return
        body = "\n".join(pieces())
        table_clauses = [cl for _, kind, cl in cur if kind == "table"]
        # A page holding nothing but ETSI's footer, running title and page number yields no chunk:
        # that furniture says nothing about the page, yet left alone it is retrievable evidence —
        # measured on the live collection, a footer-only chunk ranked second for
        # "What is the MEP in ETSI MEC?".
        if _carries_signal(body):
            chunks.append(PageChunk(
                body,
                "table" if table_clauses else "prose",
                table_clauses[0] if table_clauses else cur[0][2],
            ))
        cur.clear()

    for kind, body, header, clause in _units(text):
        if kind == "prose" and not fits(body):
            # Long prose keeps the old word window: its 40-word overlap is what dedup._stitch
            # trims when the parts are read back in order.
            words = body.split()
            step = max(1, max_words - overlap)
            for i in range(0, len(words), step):
                close()
                chunks.append(PageChunk(" ".join(words[i : i + max_words]), "prose", clause))
            continue

        # A clause change ends a chunk only once the chunk is substantial, so a chunk labelled
        # 7.2.2 holds 7.2.2 rows without dense tables degenerating into row confetti.
        if cur and kind == "table" and sum(len(p.split()) for p, _, _ in cur) >= min_words:
            held = [cl for _, k, cl in cur if k == "table"]
            if held and held[-1] != clause:
                close()

        piece = body
        if kind == "table" and header and header not in "\n".join(pieces()):
            piece = f"{header}\n{body}"
        if cur and not fits("\n".join(pieces() + [piece])):
            close()
            piece = f"{header}\n{body}" if (kind == "table" and header) else body
        cur.append((piece, kind, clause))

    close()
    return chunks