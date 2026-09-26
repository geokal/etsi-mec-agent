import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # Qdrant connection
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))
    qdrant_path: str = os.getenv("QDRANT_PATH", "")
    qdrant_index: str = os.getenv("QDRANT_INDEX", "etsi_mec_specs")

    # FastEmbed Dense Embedding (for fast global retrieval)
    dense_model: str = os.getenv("DENSE_MODEL", "BAAI/bge-small-en-v1.5")
    dense_dim: int = 384
    embedding_dim: int = 384  # alias used by Haystack store and graph scripts

    # FastEmbed ColBERT Multi-Vector Late-Interaction (for precise table/token matching)
    colbert_model: str = os.getenv("COLBERT_MODEL", "colbert-ir/colbertv2.0")
    colbert_dim: int = 128

    # Extracted diagrams storage
    diagrams_dir: str = os.getenv("DIAGRAMS_DIR", "data/diagrams")


settings = Settings()
