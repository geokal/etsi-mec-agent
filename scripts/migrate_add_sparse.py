"""Migrate the collection to add a 'sparse' BM25 vector.

Qdrant cannot add a new named vector to an existing collection, so this
copies every point (dense + colbert vectors and payload — no re-embedding)
into a fresh collection that declares all three vectors, then swaps the
collection alias back to the original name.

    uv run python scripts/migrate_add_sparse.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qdrant_client import models

from etsi_mec_agent.config import settings
from etsi_mec_agent.search import _token_sparse
from etsi_mec_agent.store import ensure_collection, get_qdrant_client

NEW = settings.qdrant_index + "_v2"


def main():
    client = get_qdrant_client()
    old = settings.qdrant_index
    src = client.get_collection(old).points_count
    ensure_collection(collection_name=NEW)

    copied = 0
    offset = None
    while True:
        pts, offset = client.scroll(
            collection_name=old, limit=8,  # small: ColBERT multivectors OOM the client buffer at 64
            with_payload=True, with_vectors=True, offset=offset,
        )
        if pts:
            client.upsert(collection_name=NEW, points=[
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

    client.delete_collection(old)
    client.update_collection_aliases(change_aliases_operations=[
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(collection_name=NEW, alias_name=old)
        )
    ])
    print(f"[done] alias '{old}' -> '{NEW}': {copied} points now carry sparse vectors")


if __name__ == "__main__":
    main()
