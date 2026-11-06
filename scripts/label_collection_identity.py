"""Stamp the embedding identity onto a collection that predates the identity guard.

    uv run python scripts/label_collection_identity.py            # report only
    uv run python scripts/label_collection_identity.py --apply    # write the metadata

The guard in `store.verify_collection_identity` refuses to query a collection that does not say
which provider made its vectors, so this has to run before the guard goes live — otherwise every
query against the existing corpus aborts. It writes collection metadata only: no point is touched,
nothing is re-embedded. It does NOT check that the stored vectors really came from the provider in
your .env; it records the claim you make by running it, so run it only against a collection you
know was ingested with these models.

    uv run python scripts/label_collection_identity.py --collection etsi_mec_prototype --apply
"""
import argparse
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root / "scripts"))

from etsi_mec_agent.config import settings
from etsi_mec_agent.embeddings import get_provider
from etsi_mec_agent.store import _identity_problem, get_qdrant_client
from migrate_add_sparse import resolve_collection


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--collection", default=settings.qdrant_index,
                        help=f"collection or alias to label (default: {settings.qdrant_index})")
    parser.add_argument("--apply", action="store_true",
                        help="write the metadata; without it this only reports")
    args = parser.parse_args()

    client = get_qdrant_client()
    name = resolve_collection(client, args.collection)
    expected = get_provider().identity()
    stored = client.get_collection(name).config.metadata or {}

    print(f"collection '{args.collection}' -> '{name}'")
    print(f"  provider identity: {expected}")
    print(f"  stored metadata:   {stored or '(none)'}")

    problem = _identity_problem(stored, expected)
    if problem is None:
        print("[ok] already labelled with this provider's identity — nothing to do.")
        return
    print(f"[gap] {problem}")

    if not args.apply:
        print("[dry-run] re-run with --apply to write the metadata above onto "
              f"'{name}' (points untouched).")
        return

    client.update_collection(collection_name=name, metadata=expected)
    after = client.get_collection(name).config.metadata or {}
    if after != expected:
        raise SystemExit(f"ABORT: '{name}' metadata reads back as {after} — label did not stick")
    print(f"[applied] '{name}' now carries the identity; the guard will accept queries against it.")


if __name__ == "__main__":
    main()
