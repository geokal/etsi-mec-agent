"""Report and (on --apply) delete byte-identical PDFs in data/specs.

    uv run python scripts/dedupe_spec_pdfs.py            # dry run, writes nothing
    uv run python scripts/dedupe_spec_pdfs.py --apply    # deletes the copies

106 files are 54 documents: the old downloader saved a neighbour's PDF under the spec number it
was asked for, so the same bytes sit under several names. ingest.py now skips them by md5, which
is why the collection is right while the folder is not.

The keeper of each group is the copy the collection actually indexed — stored chunks carry
`filename` and `diagram_paths` pointing at it, so deleting that one would silently orphan them.
Only when no copy is indexed does it fall back to the name matching the number on the cover.
Qdrant must be reachable for --apply; needs no model.
"""
import argparse
import json
import os
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from etsi_mec_agent.identity import cover_identity, file_md5

SPECS_DIR = Path(__file__).resolve().parents[1] / "data" / "specs"


def label_of(pdf: Path) -> str | None:
    """MEC003 / MEC010-2 / MEC-DEC063 — the number the *cover* declares, not the filename."""
    ident = cover_identity(pdf)
    if not ident:
        return None
    return ident["spec_id"].replace("MEC-DEC-", "MEC-DEC").replace("MEC-", "MEC", 1)


def indexed_filenames() -> set[str]:
    """Every filename stored points at, straight from Qdrant (payload only, no embedders)."""
    url = f"http://{os.getenv('QDRANT_HOST', 'localhost')}:{os.getenv('QDRANT_PORT', '6333')}" \
          f"/collections/{os.getenv('QDRANT_INDEX', 'etsi_mec_specs')}/points/scroll"
    names: set[str] = set()
    offset = None
    while True:
        body = {"limit": 500, "with_payload": ["filename"]}
        if offset:
            body["offset"] = offset
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"content-type": "application/json"})
        res = json.load(urllib.request.urlopen(req, timeout=30))["result"]
        names.update(p["payload"].get("filename", "") for p in res["points"])
        offset = res.get("next_page_offset")
        if not offset:
            break
    return names


def groups(dir: Path) -> dict[str, list[Path]]:
    by_md5: dict[str, list[Path]] = defaultdict(list)
    for pdf in sorted(dir.glob("*.pdf")):
        by_md5[file_md5(pdf)].append(pdf)
    return {md5: files for md5, files in by_md5.items() if len(files) > 1}


def keepers(files: list[Path], indexed: set[str]) -> list[Path]:
    """Never delete a copy the index points at; if none is, keep the cover-correct name."""
    used = [p for p in files if p.name in indexed]
    if used:
        return used
    for p in files:
        label = label_of(p)
        if label and re.sub(r"\D", "", p.stem) == re.sub(r"\D", "", label):
            return [p]
    return [min(files, key=lambda p: (len(p.name), p.name))]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="delete the duplicate copies")
    ap.add_argument("--dir", type=Path, default=SPECS_DIR)
    args = ap.parse_args()

    dupes = groups(args.dir)
    if not dupes:
        print(f"{args.dir}: no byte-identical PDFs left.")
        return

    try:
        indexed = indexed_filenames()
    except Exception as exc:
        if args.apply:
            # Deleting blind is how stored chunks get orphaned.
            print(f"[ABORT] cannot read the collection to check which files are indexed: {exc}\n"
                  f"        Start Qdrant, or run the dry report without --apply.")
            sys.exit(1)
        print(f"[WARN] Qdrant unreachable ({exc}) — dry run only, keepers chosen from PDF covers.\n")
        indexed = set()

    freed = 0
    removed = 0
    for md5, files in sorted(dupes.items(), key=lambda kv: kv[1][0].name):
        keep = keepers(files, indexed)
        for p in files:
            if p in keep:
                continue
            freed += p.stat().st_size
            removed += 1
            print(f"  {', '.join(k.name for k in keep):<28} <- {p.name} ({p.stat().st_size // 1024} KB)")
            if args.apply:
                p.unlink()

    print(f"\n{len(dupes)} documents held more than one copy; {removed} files, "
          f"{freed // 1024 // 1024} MB " + ("deleted" if args.apply else "removable (dry run — re-run with --apply)")
          + f"; {len(indexed)} filenames are referenced by the stored chunks")


if __name__ == "__main__":
    main()
