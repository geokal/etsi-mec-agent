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
from etsi_mec_agent.embeddings import _token_sparse
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


def _embed_text(payload: dict) -> str:
    """Rebuild the text ingest embedded, so a migrated sparse vector matches a fresh one.

    ingest.py prefixes every chunk with `spec_id edition clause heading` before tokenising it, but
    the stored payload keeps only the body under "text". Reassembling that prefix from the identity
    fields is what stops a migrated chunk losing its cover-derived spec number in the sparse vector.
    """
    context = " ".join(filter(None, [
        payload.get("spec_id"), payload.get("edition"),
        payload.get("clause"), payload.get("heading"),
    ]))
    body = payload.get("text", "")
    return f"{context}\n{body}" if context else body


def main():
    """Copy every point into a collection that declares 'sparse', then move the alias onto it."""
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

    src_col = client.get_collection(source)
    # A copy carries the vectors it copies, so the target inherits the SOURCE's identity claim:
    # stamping the current provider's would label local vectors as cloud-made (or the reverse)
    # and wave through exactly the mismatch the guard exists to catch.
    source_identity = src_col.config.metadata
    if not source_identity:
        raise SystemExit(
            f"ABORT: '{source}' records no embedding identity, so nothing can say which provider "
            f"made the vectors about to be copied. Label it first — "
            f"`uv run python scripts/label_collection_identity.py --apply` with QDRANT_INDEX "
            f"pointed at '{source}' — then re-run.")
    # The target is declared from DENSE_DIM/COLBERT_DIM but the vectors come from the source, so a
    # mismatch fails the first upsert *after* the target exists — and the "already exists" refusal
    # above then blocks every retry under that name. Read the stored params, not the identity claim:
    # a claim can be wrong, the stored width is what the upsert actually hits.
    for name, want, env in (("dense", settings.dense_dim, "DENSE_DIM"),
                            ("colbert", settings.colbert_dim, "COLBERT_DIM")):
        got = src_col.config.params.vectors[name].size
        if got != want:
            raise SystemExit(
                f"ABORT: '{source}' stores {got}-d '{name}' vectors but '{target}' would be created "
                f"{want}-d, so the copy would fail on its first upsert and leave the target behind. "
                f"Set {env}={got} and re-run.")
    ensure_collection(collection_name=target, metadata=source_identity)
    src = src_col.points_count

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
                    vector={**p.vector, "sparse": _token_sparse(_embed_text(p.payload))},
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
