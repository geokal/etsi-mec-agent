from qdrant_client import QdrantClient, models
from etsi_mec_agent.config import settings


def get_qdrant_client() -> QdrantClient:
    """Connect to Qdrant vector database."""
    if settings.qdrant_path:
        return QdrantClient(path=settings.qdrant_path)
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def ensure_collection(recreate: bool = False):
    """
    Initialize the Qdrant hybrid collection with:
    - 'dense': standard 384-d Cosine vectors for fast candidate filtering
    - 'colbert': 128-d multi-vectors with MaxSim comparator for token-level table matching
    """
    client = get_qdrant_client()
    exists = client.collection_exists(settings.qdrant_index)

    if recreate and exists:
        print(f"Deleting existing collection '{settings.qdrant_index}'...")
        client.delete_collection(settings.qdrant_index)
        exists = False

    if not exists:
        print(f"Creating Hybrid (Dense + ColBERT) collection '{settings.qdrant_index}' in Qdrant...")
        client.create_collection(
            collection_name=settings.qdrant_index,
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
        )
        client.create_payload_index(
            collection_name=settings.qdrant_index,
            field_name="has_diagram",
            field_schema=models.PayloadSchemaType.BOOL,  # was KEYWORD — booleans never matched
        )
        print(f"[SUCCESS] Hybrid collection '{settings.qdrant_index}' created.")
