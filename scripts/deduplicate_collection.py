"""
scripts/deduplicate_collection.py

Deduplicate the Qdrant collection IN-PLACE — no re-embedding needed.

The monitor regex bug caused the same PDF to be downloaded under multiple
filenames (e.g. MEC059 content stored as MEC059, MEC069, MEC077, MEC079).
Each ingest run created fresh UUIDs for those chunks, leaving the collection
with 3-4× more points than it should have.

This script:
  1. Scrolls all points in batches (payload only, no vectors).
  2. Groups them by a text fingerprint (markdown images stripped).
  3. Within each duplicate group, elects ONE survivor:
       - Prefer the point whose filename matches a canonical MECxxx.pdf pattern.
       - Among ties, prefer the one with the lowest UUID (stable tie-break).
  4. Deletes all non-survivor points.
  5. Prints a summary of how many points were kept / removed.

Estimated runtime: 5–15 min for 10 000 points on localhost Qdrant.
No embeddings are computed — pure payload reads + delete calls.

Usage:
    uv run python scripts/deduplicate_collection.py
    uv run python scripts/deduplicate_collection.py --dry-run
    uv run python scripts/deduplicate_collection.py --batch-size 512
"""

import argparse
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from etsi_mec_agent.config import settings
from etsi_mec_agent.dedup import _IMG_RE
from etsi_mec_agent.store import get_qdrant_client

# _IMG_RE is shared with etsi_mec_agent.dedup: it strips markdown image references so that
# "![](MEC059.pdf-0035-02.png)" vs "![](MEC079.pdf-0035-02.png)" don't produce different
# fingerprints for identical text. The fingerprint below is still deliberately this script's
# own (600 chars + whitespace normalisation) because it drives a destructive pass.
_MEC_RE = re.compile(r'^MEC\d{3}\.pdf$', re.IGNORECASE)


def _fingerprint(text: str) -> str:
    """Strip images, normalise whitespace, take first 600 chars."""
    cleaned = _IMG_RE.sub("", text)
    return " ".join(cleaned.split())[:600]


def _canonical_score(filename: str) -> int:
    """
    Lower is better. Points whose filename looks like a canonical MECxxx.pdf
    get score 0; all others get score 1. Used to prefer the 'real' copy.
    """
    return 0 if _MEC_RE.match(filename or "") else 1


def scroll_all_points(client, batch_size: int):
    """Yield all points (payload only, no vectors) from the collection."""
    offset = None
    total = 0
    while True:
        points, offset = client.scroll(
            collection_name=settings.qdrant_index,
            limit=batch_size,
            offset=offset,
            with_payload=["text", "doc_id", "filename", "page"],
            with_vectors=False,
        )
        for p in points:
            yield p
            total += 1
        if offset is None:
            break
    return total


def deduplicate(batch_size: int, dry_run: bool):
    client = get_qdrant_client()

    info = client.get_collection(settings.qdrant_index)
    total_points = info.points_count
    print(f"[dedup] Collection '{settings.qdrant_index}': {total_points} points")
    print(f"[dedup] Scrolling all points (batch_size={batch_size}) …\n")

    t0 = time.time()

    # Group point IDs by text fingerprint
    # fingerprint → [(canonical_score, point_id, filename)]
    groups: dict[str, list] = defaultdict(list)
    scanned = 0

    for point in scroll_all_points(client, batch_size):
        text = point.payload.get("text", "")
        fp   = _fingerprint(text)
        fname = point.payload.get("filename", "")
        groups[fp].append((_canonical_score(fname), str(point.id), point.id))
        scanned += 1
        if scanned % 1000 == 0:
            elapsed = time.time() - t0
            print(f"  scanned {scanned}/{total_points} ({elapsed:.1f}s) …", flush=True)

    elapsed = time.time() - t0
    print(f"\n[dedup] Scanned {scanned} points in {elapsed:.1f}s.")
    print(f"[dedup] Unique fingerprints: {len(groups)}")

    # Identify survivors and victims
    survivors = 0
    victims: list = []       # list of point IDs to delete

    dup_groups = 0
    for fp, entries in groups.items():
        if len(entries) == 1:
            survivors += 1
            continue
        # Sort: canonical score first (0=good), then UUID string for stable ordering
        entries.sort(key=lambda e: (e[0], e[1]))
        survivor_id = entries[0][2]
        survivors += 1
        dup_groups += 1
        for _, _, pid in entries[1:]:
            victims.append(pid)

    print(f"[dedup] Duplicate groups: {dup_groups}")
    print(f"[dedup] Points to DELETE: {len(victims)}")
    print(f"[dedup] Points to KEEP:   {survivors}")

    if not victims:
        print("\n[dedup] Collection is already clean — nothing to delete.")
        return

    if dry_run:
        print(f"\n[dry-run] Would delete {len(victims)} points. Re-run without --dry-run to apply.")
        return

    # Delete victims in batches
    print(f"\n[dedup] Deleting {len(victims)} points …")
    delete_batch = 256
    deleted = 0
    t1 = time.time()
    for i in range(0, len(victims), delete_batch):
        batch = victims[i : i + delete_batch]
        client.delete(
            collection_name=settings.qdrant_index,
            points_selector=batch,
        )
        deleted += len(batch)
        print(f"  deleted {deleted}/{len(victims)} …", flush=True)

    elapsed = time.time() - t1
    print(f"\n[dedup] Done. Deleted {deleted} duplicate points in {elapsed:.1f}s.")
    print(f"[dedup] Collection now has ~{survivors} points.")
    print(f"[dedup] Savings: {len(victims) / max(scanned, 1) * 100:.1f}% reduction in collection size.")


def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate the Qdrant MEC collection in-place (no re-embedding needed)."
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Points to fetch per Qdrant scroll call (default: 256)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be deleted without actually deleting anything",
    )
    args = parser.parse_args()
    deduplicate(batch_size=args.batch_size, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
