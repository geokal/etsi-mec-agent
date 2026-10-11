"""Migrate the collection to add the 'sparse' term-frequency vector.

Qdrant cannot add a new named vector to an existing collection, so this
copies every point (dense + colbert vectors and payload — no re-embedding)
into a fresh collection that declares all three vectors, then swaps the
collection alias back to the original name.

    uv run python scripts/migrate_add_sparse.py
    uv run python scripts/migrate_add_sparse.py --target etsi_mec_specs_v3
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qdrant_client import models

from etsi_mec_agent.config import settings
from etsi_mec_agent.search import _token_sparse
from etsi_mec_agent.store import ensure_collection, get_qdrant_client

NEW = settings.qdrant_index + "_v2"


def resolve_collection(client, name: str) -> str:
    """The collection a name actually refers to, following an alias one hop.

    `QDRANT_INDEX` names an alias in the live setup, and `delete_collection` on an alias name does
    not do what its caller means. Resolving first keeps the deletion pointed at a real collection.
    """
    for alias in client.get_aliases().aliases:
        if alias.alias_name == name:
            return alias.collection_name
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", default=NEW,
                        help=f"collection to copy into (default: {NEW})")
    args = parser.parse_args()
    target = args.target

    client = get_qdrant_client()
    alias = settings.qdrant_index                 # the name everyone searches by, alias or not
    source = resolve_collection(client, alias)    # the collection behind it

    if client.collection_exists(target):
        raise SystemExit(
            f"ABORT: '{target}' already exists. Copying into it would mix these points with the "
            f"ones already there — in the live setup that name is the rollback collection. Pass "
            f"--target <fresh name>, or delete '{target}' once it is genuinely no longer needed.")
    if target == source:
        raise SystemExit(f"ABORT: --target '{target}' is the collection being migrated")

    src = client.get_collection(source).points_count
    ensure_collection(collection_name=target)

    copied = 0
    offset = None
    while True:
        pts, offset = client.scroll(
            collection_name=source, limit=8,  # small: ColBERT multivectors OOM the client buffer at 64
            with_payload=True, with_vectors=True, offset=offset,
        )
        if pts:
            client.upsert(collection_name=target, points=[
                models.PointStruct(
                    id=p.id,
                    vector={**p.vector, "sparse": _token_sparse(p.payload.get("text", ""))},
                    payload=p.payload,
                )
                for p in pts
            ])
            copied += len(pts)
            print(f"  copied {copied}/{src}", flush=True)
        if offset is None:
            break

    if copied != src:
        raise SystemExit(f"ABORT: copied {copied} but source had {src} — alias NOT switched")

    move = [models.CreateAliasOperation(
        create_alias=models.CreateAlias(collection_name=target, alias_name=alias))]
    if alias != source:
        # Re-point, then delete. `create_alias` moves an alias that already exists, and deleting a
        # collection drops its aliases silently — so deleting first leaves a window (and, across a
        # crash, an indefinite state) in which everyone's QDRANT_INDEX names nothing at all.
        # Both facts were probed against Qdrant 1.19.1 rather than assumed.
        client.update_collection_aliases(change_aliases_operations=move)
        client.delete_collection(source)
    else:
        # Here the name IS a collection, and Qdrant refuses an alias colliding with a collection
        # name (409), so the old collection has to go before the name can be reused.
        client.delete_collection(source)
        client.update_collection_aliases(change_aliases_operations=move)
    print(f"[done] alias '{alias}' -> '{target}': {copied} points now carry sparse vectors")


if __name__ == "__main__":
    main()
