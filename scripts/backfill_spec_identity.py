"""Stamp content-derived spec identity onto the chunks already in the collection.

    uv run python scripts/backfill_spec_identity.py              # report, writes nothing
    uv run python scripts/backfill_spec_identity.py --apply      # set_payload, no re-embedding
    uv run python scripts/backfill_spec_identity.py --self-check # asserts on the parser, offline

The filenames in data/specs/ do not identify their contents: MEC041.pdf is GS MEC 040,
MEC012.pdf is GR MEC-DEC 025, and 16 separate files all hold GS MEC 009. Identity is read
from the ETSI title line on the PDF cover instead, so a query can prefer one edition per
spec rather than letting three copies of the same table compete for the --answer-context
budget (see docs/superpowers/specs/2026-10-06-corpus-wide-answering-design.md §5c).
"""

from __future__ import annotations

import argparse
import hashlib
import re
from collections import defaultdict
from pathlib import Path

import pymupdf
from qdrant_client import models

from etsi_mec_agent.store import get_qdrant_client
from etsi_mec_agent.config import settings

HEADER = re.compile(
    r"ETSI\s+(?:GS|GR|TS|ISG)\s+(MEC(?:-DEC)?\s+\d+(?:-\d+)?)\s+"
    r"V(\d+)\.(\d+)\.(\d+)\s*\((\d{4})-(\d{2})\)"
)


def cover_identity(pdf: Path) -> dict | None:
    """{spec_id, edition, pub_date, key} from the cover pages, None if unnumbered."""
    with pymupdf.open(str(pdf)) as doc:
        head = " ".join(" ".join(doc[i].get_text() for i in range(min(3, doc.page_count))).split())
    m = HEADER.search(head)
    if not m:
        return None
    edition = f"V{m.group(2)}.{m.group(3)}.{m.group(4)}"
    return {
        "spec_id": re.sub(r"\s+", "-", m.group(1)),
        "edition": edition,
        "pub_date": f"{m.group(5)}-{m.group(6)}",
        "key": tuple(int(p) for p in edition[1:].split(".")),
    }


def scan(specs_dir: Path) -> dict:
    """md5 -> {stems, ident}; then flag which groups hold the newest edition of a spec."""
    groups: dict[str, list[str]] = defaultdict(list)
    for pdf in sorted(specs_dir.glob("*.pdf")):
        groups[hashlib.md5(pdf.read_bytes()).hexdigest()].append(pdf.stem)

    rows = {md5: {"stems": stems, "ident": cover_identity(specs_dir / f"{stems[0]}.pdf")}
            for md5, stems in groups.items()}

    newest: dict[str, tuple] = {}
    for row in rows.values():
        ident = row["ident"]
        if ident and ident["key"] > newest.get(ident["spec_id"], ()):
            newest[ident["spec_id"]] = ident["key"]
    for row in rows.values():
        ident = row["ident"]
        row["is_current"] = bool(ident) and ident["key"] == newest[ident["spec_id"]]
    return rows


def mislabeled(stem: str, ident: dict) -> bool:
    """True when the filename names a different spec number than the content does."""
    want = re.search(r"\d{3}", ident["spec_id"])
    got = re.search(r"MEC(\d{3})", stem)
    return bool(want and got and want.group(0) != got.group(1))


def collect(client) -> tuple[dict[str, int], dict[str, list]]:
    """One pass over the collection: chunks per doc_id, and doc_id -> point ids."""
    counts: dict[str, int] = defaultdict(int)
    ids: dict[str, list] = defaultdict(list)
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=settings.qdrant_index,
            with_payload=["doc_id"],
            limit=256,
            offset=offset,
        )
        for p in points:
            doc = (p.payload or {}).get("doc_id", "")
            counts[doc] += 1
            ids[doc].append(p.id)
        if offset is None:
            return counts, ids


def report(rows: dict, counts: dict[str, int]) -> None:
    by_spec: dict[str, list] = defaultdict(list)
    for md5, row in rows.items():
        by_spec[row["ident"]["spec_id"] if row["ident"] else "?"].append((md5, row))

    old_chunks = dup_chunks = 0
    for spec in sorted(by_spec):
        editions = sorted(by_spec[spec], key=lambda x: (x[1]["ident"] or {}).get("key", ()), reverse=True)
        for md5, row in editions:
            ident = row["ident"]
            if not ident:
                print(f"  {'?':<14} no number on cover: {row['stems']}")
                continue
            live = {s: counts[s] for s in row["stems"] if s in counts}
            n = sum(live.values())
            if not row["is_current"]:
                old_chunks += n
            if len(row["stems"]) > 1:
                dup_chunks += n * (len(row["stems"]) - 1) // len(row["stems"])
            flag = "current" if row["is_current"] else "SUPERSEDED"
            lies = [s for s in row["stems"] if mislabeled(s, ident)]
            print(f"  {spec:<14} {ident['edition']:<9} {ident['pub_date']}  {flag:<10} "
                  f"{n:>4} chunks  {row['stems']}"
                  + (f"   <-- filename lies for {lies}" if lies else ""))

    print(f"\n{len(rows)} unique documents across {sum(len(r['stems']) for r in rows.values())} files, "
          f"{sum(counts.values())} indexed chunks.")
    print(f"~{old_chunks} chunks are a superseded edition, ~{dup_chunks} are byte-duplicates of a "
          f"doc_id already counted.")


def self_check() -> None:
    cases = {
        "ETSI GS MEC 003 V4.1.1 (2025-05)": ("MEC-003", "V4.1.1", "2025-05"),
        "ETSI GR MEC-DEC 025 V2.1.1 (2019-06)": ("MEC-DEC-025", "V2.1.1", "2019-06"),
        "ETSI GS MEC 010-2 V2.1.1 (2019-11)": ("MEC-010-2", "V2.1.1", "2019-11"),
    }
    for line, want in cases.items():
        head = " ".join(line.split())
        m = HEADER.search(head)
        assert m, line
        got = (re.sub(r"\s+", "-", m.group(1)), f"V{m.group(2)}.{m.group(3)}.{m.group(4)}",
               f"{m.group(5)}-{m.group(6)}")
        assert got == want, (got, want)
    assert HEADER.search("Mobile-Edge Computing Introductory Technical White Paper") is None

    rows = scan(Path("data/specs"))
    mec003 = {r["ident"]["edition"] for r in rows.values()
              if r["ident"] and r["ident"]["spec_id"] == "MEC-003"}
    assert "V4.1.1" in mec003 and "V2.2.1" in mec003, mec003
    current = [r for r in rows.values() if r["is_current"]]
    assert all(len({s for s in r["stems"]}) >= 1 for r in current)
    newest = next(r for r in rows.values() if r["ident"] and r["ident"]["edition"] == "V4.1.1"
                  and r["ident"]["spec_id"] == "MEC-003")
    assert newest["is_current"] and "MEC023" in newest["stems"] and "MEC070" in newest["stems"]
    assert any(r["ident"] is None for r in rows.values()), "unnumbered docs must stay unknown"
    print(f"self-check OK: {len(rows)} unique documents, {len(current)} current groups")


def verify() -> None:
    """Assert search_specs() itself drops superseded editions, with the embedders stubbed.

    fastembed is replaced before etsi_mec_agent.search is imported, so this runs against the
    live collection without loading ONNX; the vectors are arbitrary because the assert is
    about which chunks a filtered query can return, not about their rank.
    """
    import sys
    import types

    stub = types.ModuleType("fastembed")
    stub.LateInteractionTextEmbedding = stub.TextEmbedding = object
    sys.modules["fastembed"] = stub
    import etsi_mec_agent.search as search

    class _V(list):
        def tolist(self):
            return list(self)

    search.get_embedders = lambda: (
        types.SimpleNamespace(embed=lambda t: [_V([0.1] * settings.dense_dim) for _ in t]),
        types.SimpleNamespace(query_embed=lambda t: [_V([[0.0] * 128] * 4)]),
    )

    docs = search.search_specs("reference point MEC platform orchestrator", top_k=5, use_bm25=True)
    leaked = [d.meta.get("doc_id") for d in docs if (d.meta or {}).get("is_current") is False]
    assert not leaked, f"current_only leaked superseded editions: {leaked}"

    all_editions = search.search_specs("reference point MEC platform orchestrator",
                                       top_k=5, use_bm25=True, current_only=False)
    stale = [d.meta.get("doc_id") for d in all_editions if (d.meta or {}).get("is_current") is False]
    assert stale, "the same query must reach a superseded edition once the filter is off"
    print(f"\n[verify] filtered {len(docs)} excerpts, none superseded; --all-editions returns "
          f"{len(stale)} of them: {stale}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write spec_id/edition/is_current payloads")
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="assert search_specs() drops superseded editions against the live collection")
    ap.add_argument("--specs", default="data/specs")
    args = ap.parse_args()

    if args.self_check:
        self_check()
        return

    if args.verify:
        verify()
        return

    rows = scan(Path(args.specs))
    client = get_qdrant_client()
    counts, ids = collect(client)
    missing = sorted(s for r in rows.values() for s in r["stems"] if s not in counts)
    if missing:
        print(f"doc_ids on disk but not indexed: {missing}")
    report(rows, counts)

    if not args.apply:
        print("\ndry run — nothing written. Re-run with --apply.")
        return

    # Point ids in batches: a filter-scoped update re-reads the whole collection and
    # runs past the client's request timeout on 4.7k points.
    for md5, row in rows.items():
        ident = row["ident"]
        payload = {
            "spec_id": ident["spec_id"] if ident else None,
            "edition": ident["edition"] if ident else None,
            "pub_date": ident["pub_date"] if ident else None,
            "content_md5": md5,
            # A white paper or slide deck is superseded by nothing, so it stays current.
            "is_current": row["is_current"] if ident else True,
        }
        point_ids = [pid for stem in row["stems"] for pid in ids.get(stem, [])]
        for i in range(0, len(point_ids), 400):
            client.set_payload(collection_name=settings.qdrant_index, payload=payload,
                               points=point_ids[i:i + 400], wait=True)
        print(f"  {payload['spec_id'] or '?':<14} {payload['edition'] or '-':<9} "
              f"current={str(payload['is_current']):<5} {len(point_ids)} chunks")

    for field, kind in (("is_current", models.PayloadSchemaType.BOOL),
                        ("spec_id", models.PayloadSchemaType.KEYWORD),
                        ("content_md5", models.PayloadSchemaType.KEYWORD)):
        client.create_payload_index(settings.qdrant_index, field, field_schema=kind)
    print("\nStamped every chunk and indexed is_current + spec_id.")


if __name__ == "__main__":
    main()
