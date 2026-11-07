from qdrant_client import QdrantClient, models
from etsi_mec_agent.config import settings
from etsi_mec_agent.embeddings import get_provider


def get_qdrant_client() -> QdrantClient:
    """Connect to Qdrant vector database."""
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None,
                            # Without this the client embeds models.Document locally via fastembed
                            # instead of sending the text to the cluster (qdrant_client.py:387).
                            cloud_inference=settings.embedding_mode == "cloud")
    if settings.qdrant_path:
        return QdrantClient(path=settings.qdrant_path)
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port,
                       api_key=settings.qdrant_api_key or None)


def _identity_problem(stored: dict | None, expected: dict) -> str | None:
    """Why a collection cannot answer queries embedded by `expected`, or None if it can."""
    if not stored:
        return ("no embedding identity recorded — run "
                "`uv run python scripts/label_collection_identity.py --apply` to stamp it")
    for key, want in expected.items():
        got = stored.get(key)
        if got != want:
            return f"{key} mismatch: collection holds {got!r}, provider embeds {want!r}"
    return None


def verify_collection_identity(client=None, collection_name: str | None = None):
    """Abort before any query if the stored vectors were not made by the current provider.

    A model swap silently re-embeds queries into another space: retrieval returns ranked noise
    with no error, so the mismatch has to be reported as a hard failure naming the key.
    """
    name = collection_name or settings.qdrant_index
    client = client or get_qdrant_client()
    stored = client.get_collection(name).config.metadata
    problem = _identity_problem(stored, get_provider().identity())
    if problem:
        raise RuntimeError(f"collection {name!r}: {problem}")


def ensure_collection(recreate: bool = False, collection_name: str | None = None,
                      metadata: dict | None = None):
    """
    Initialize the Qdrant hybrid collection with:
    - 'dense': Cosine vectors, `settings.dense_dim` wide, for fast candidate filtering
    - 'colbert': MaxSim multi-vectors, `settings.colbert_dim` wide per token, for table matching
    - 'sparse': raw term frequencies, IDF weighting applied server-side (Modifier.IDF)

    collection_name overrides settings.qdrant_index (used by the sparse migration).
    metadata overrides the identity stamped at birth: a raw-vector copy passes the source's claim
    through, because the vectors it carries were not made by the current provider.
    """
    name = collection_name or settings.qdrant_index
    # Resolved before the delete below. Constructing the provider can fail (EMBEDDING_MODE=cloud
    # without QDRANT_URL), and failing after the wipe would leave an empty collection behind.
    identity = metadata if metadata is not None else get_provider().identity()
    client = get_qdrant_client()
    exists = client.collection_exists(name)

    if recreate and exists:
        print(f"Deleting existing collection '{name}'...")
        client.delete_collection(name)
        exists = False

    if not exists:
        print(f"Creating Hybrid (Dense + ColBERT + Sparse) collection '{name}' in Qdrant...")
        client.create_collection(
            collection_name=name,
            vectors_config={
                "dense": models.VectorParams(
                    size=settings.dense_dim,
                    distance=models.Distance.COSINE,
                ),
                "colbert": models.VectorParams(
                    size=settings.colbert_dim,
                    distance=models.Distance.DOT,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                    on_disk=True,  # Keeps RAM usage low by storing ColBERT multi-vectors on disk
                ),
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF),
            },
            # Self-describing at birth: the guard below compares this against the current
            # provider before any query, so a model swap without re-ingest aborts loudly.
            metadata=identity,
        )
        client.create_payload_index(
            collection_name=name,
            field_name="has_diagram",
            field_schema=models.PayloadSchemaType.BOOL,  # was KEYWORD — booleans never matched
        )
        print(f"[SUCCESS] Hybrid collection '{name}' created.")
