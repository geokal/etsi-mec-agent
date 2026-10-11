from qdrant_client import QdrantClient, models
from etsi_mec_agent.config import settings


def get_qdrant_client() -> QdrantClient:
    """Connect to Qdrant vector database."""
    if settings.qdrant_path:
        return QdrantClient(path=settings.qdrant_path)
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def ensure_collection(recreate: bool = False, collection_name: str | None = None):
    """
    Initialize the Qdrant hybrid collection with:
    - 'dense': standard 384-d Cosine vectors for fast candidate filtering
    - 'colbert': 128-d multi-vectors with MaxSim comparator for token-level table matching
    - 'sparse': raw term frequencies, IDF weighting applied server-side (Modifier.IDF)

    collection_name overrides settings.qdrant_index (used by the sparse migration).
    """
    name = collection_name or settings.qdrant_index
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
        )
        client.create_payload_index(
            collection_name=name,
            field_name="has_diagram",
            field_schema=models.PayloadSchemaType.BOOL,  # was KEYWORD — booleans never matched
        )
        print(f"[SUCCESS] Hybrid collection '{name}' created.")
