"""Delete the chunks that ETSI's page furniture alone produced — no re-embed, no re-ingest.

    uv run python scripts/drop_no_signal_chunks.py            # report only
    uv run python scripts/drop_no_signal_chunks.py --apply    # delete

`chunking._carries_signal` refuses such a chunk now, but the live collection still holds the ones the
old chunker wrote. Measured 2026-10-08: 10 of 5264 points, every one of them body-less (e.g.
`**_ETSI_**`), and one of them ranked second for "What is the MEP in ETSI MEC?".

The predicate is imported from the chunker rather than restated here, so this script cannot disagree
with ingest about what counts as junk. Payloads only, no vectors, no models.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from etsi_mec_agent.chunking import _carries_signal
from etsi_mec_agent.config import settings
from etsi_mec_agent.store import get_qdrant_client


def find_no_signal(client, batch_size: int) -> tuple[list, int, list]:
    """(ids of points whose text is page furniture only, points scanned, up to 3 examples)."""
    victims, examples, scanned, offset = [], [], 0, None
    while True:
        points, offset = client.scroll(
            collection_name=settings.qdrant_index,
            limit=batch_size,
            offset=offset,
            with_payload=["text", "spec_id", "edition", "page"],
            with_vectors=False,
        )
        for p in points:
            payload = p.payload or {}
            scanned += 1
            if not _carries_signal(payload.get("text", "")):
                victims.append(p.id)
                if len(examples) < 3:
                    meta = f"{payload.get('spec_id')} {payload.get('edition')} p.{payload.get('page')}"
                    examples.append(f"{meta}: {payload.get('text', '')!r}")
        if offset is None:
            return victims, scanned, examples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delete chunks whose text is only ETSI's page furniture (dry run by default).")
    parser.add_argument("--apply", action="store_true", help="delete the points, not just report them")
    parser.add_argument("--batch-size", type=int, default=256, help="points per scroll (default: 256)")
    args = parser.parse_args()

    client = get_qdrant_client()
    victims, scanned, examples = find_no_signal(client, args.batch_size)
    print(f"[no-signal] {len(victims)} of {scanned} points in '{settings.qdrant_index}' "
          f"carry no signal.")
    for line in examples:
        print(f"  e.g. {line}")
    if not victims:
        return
    if not args.apply:
        print("[no-signal] dry run: re-run with --apply to delete them.")
        return

    client.delete(collection_name=settings.qdrant_index, points_selector=victims)
    # Re-scan rather than trust the response: the deletion either reached the collection or it did not.
    left, rescan, _ = find_no_signal(client, args.batch_size)
    print(f"[no-signal] deleted {len(victims)} points; {rescan} scanned, {len(left)} still carry no signal.")
    if left:
        sys.exit(1)


if __name__ == "__main__":
    main()
