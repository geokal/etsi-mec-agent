"""
scripts/backfill_clip.py

Backfill CLIP image embeddings into the existing Qdrant collection.

Because the collection was created without a 'clip' named vector, this script:
  1. Adds the 'clip' named vector (512-dim Cosine) to the collection schema.
  2. Scrolls all points that have has_diagram=true and non-empty diagram_paths.
  3. For each such point, embeds the first diagram image with CLIP.
  4. Updates the point in Qdrant with the new 'clip' vector.

Run once after ingesting all PDFs — no need to recreate the collection.

Usage:
    uv add transformers torch Pillow
    uv run python scripts/backfill_clip.py

Options:
    --batch-size N   Points to update per Qdrant call (default: 16)
    --limit N        Max points to process (default: all)
    --dry-run        Print what would be done without touching Qdrant
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from etsi_mec_agent.config import settings
from etsi_mec_agent.store import get_qdrant_client

CLIP_DIM = 512


def add_clip_vector_to_schema(client):
    """Add the 'clip' named vector to the Qdrant collection if not already present."""
    from qdrant_client import models

    info = client.get_collection(settings.qdrant_index)
    existing_vectors = info.config.params.vectors

    if "clip" in existing_vectors:
        print("[schema] 'clip' named vector already exists — skipping schema update.")
        return

    print(f"[schema] Adding 'clip' (512-dim Cosine) to collection '{settings.qdrant_index}' …")
    client.update_collection(
        collection_name=settings.qdrant_index,
        vectors_config={
            "clip": models.VectorParams(
                size=CLIP_DIM,
                distance=models.Distance.COSINE,
            )
        },
    )
    print("[schema] Done.")


def scroll_diagram_points(client, limit: int | None):
    """
    Scroll all points that have at least one diagram_path.
    Returns list of (point_id, diagram_paths) tuples.
    """
    from qdrant_client import models

    results = []
    offset = None
    batch = 256

    while True:
        fetch = batch if limit is None else min(batch, limit - len(results))
        points, offset = client.scroll(
            collection_name=settings.qdrant_index,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="has_diagram",
                        match=models.MatchValue(value=True),
                    )
                ]
            ),
            limit=fetch,
            offset=offset,
            with_payload=["diagram_paths", "doc_id", "page"],
            with_vectors=False,
        )
        for p in points:
            paths = p.payload.get("diagram_paths", [])
            if paths:
                results.append((p.id, paths))
        if offset is None or (limit is not None and len(results) >= limit):
            break

    return results


def backfill(batch_size: int, limit: int | None, dry_run: bool):
    from qdrant_client import models
    from etsi_mec_agent.tools.clip_embed import embed_images

    client = get_qdrant_client()

    # Step 1 — extend collection schema
    if not dry_run:
        add_clip_vector_to_schema(client)
    else:
        print("[dry-run] Would add 'clip' vector to schema.")

    # Step 2 — find all diagram points
    print("\n[backfill] Scanning for points with diagrams …")
    diagram_points = scroll_diagram_points(client, limit)
    print(f"[backfill] Found {len(diagram_points)} points with diagram paths.")

    if not diagram_points:
        print("[backfill] Nothing to do.")
        return

    # Step 3 — embed and update in batches
    t0 = time.time()
    updated = 0
    skipped = 0

    for batch_start in range(0, len(diagram_points), batch_size):
        batch = diagram_points[batch_start: batch_start + batch_size]

        point_ids = []
        clip_vecs = []

        for point_id, paths in batch:
            # Use the first diagram image for this chunk
            img_path = paths[0]
            if not Path(img_path).exists():
                print(f"  [SKIP] {img_path} not found on disk.", flush=True)
                skipped += 1
                continue

            vecs = embed_images([img_path])
            if vecs[0] is None:
                print(f"  [SKIP] Could not embed {img_path}.", flush=True)
                skipped += 1
                continue

            point_ids.append(point_id)
            clip_vecs.append(vecs[0])

        if not point_ids:
            continue

        if dry_run:
            print(f"  [dry-run] Would update {len(point_ids)} points with CLIP vectors.")
        else:
            # Qdrant: update named vectors for existing points
            client.update_vectors(
                collection_name=settings.qdrant_index,
                points=[
                    models.PointVectors(id=pid, vector={"clip": vec})
                    for pid, vec in zip(point_ids, clip_vecs)
                ],
            )

        updated += len(point_ids)
        elapsed = time.time() - t0
        rate = updated / elapsed if elapsed > 0 else 0
        remaining = len(diagram_points) - batch_start - len(batch)
        eta = remaining / rate if rate > 0 else 0
        print(
            f"  [{batch_start + len(batch)}/{len(diagram_points)}] "
            f"Updated {updated} | Skipped {skipped} | "
            f"{rate:.1f} pts/s | ETA {eta:.0f}s",
            flush=True,
        )

    elapsed = time.time() - t0
    print(f"\n[backfill] Complete. {updated} points updated, {skipped} skipped in {elapsed:.1f}s.")
    print(f"[backfill] You can now search diagrams with --image-query (see search.py)")


def main():
    parser = argparse.ArgumentParser(description="Backfill CLIP embeddings for diagram chunks in Qdrant.")
    parser.add_argument("--batch-size", type=int, default=16, help="Qdrant update batch size (default: 16)")
    parser.add_argument("--limit",      type=int, default=None, help="Max points to process (default: all)")
    parser.add_argument("--dry-run",    action="store_true",    help="Print actions without modifying Qdrant")
    args = parser.parse_args()

    backfill(
        batch_size=args.batch_size,
        limit=args.limit,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
