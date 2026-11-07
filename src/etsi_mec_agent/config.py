import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """The whole runtime configuration: every field is env-backed and resolved once at import."""

    # Qdrant connection
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))
    qdrant_path: str = os.getenv("QDRANT_PATH", "")
    qdrant_index: str = os.getenv("QDRANT_INDEX", "etsi_mec_specs")
    # Remote Qdrant (e.g. Qdrant Cloud free tier): when QDRANT_URL is set it wins over host/port.
    qdrant_url: str = os.getenv("QDRANT_URL", "")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")

    # Answer LLM — any OpenAI-compatible endpoint; default stays the OpenRouter free router.
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "openrouter/free")

    # Embedding runtime: "local" (FastEmbed ONNX) is the only one shipped so far; every other
    # runtime arrives behind this switch with its own collection identity.
    embedding_mode: str = os.getenv("EMBEDDING_MODE", "local")

    # FastEmbed Dense Embedding (for fast global retrieval)
    dense_model: str = os.getenv("DENSE_MODEL", "BAAI/bge-small-en-v1.5")
    dense_dim: int = int(os.getenv("DENSE_DIM", "384"))

    # FastEmbed ColBERT Multi-Vector Late-Interaction (for precise table/token matching)
    colbert_model: str = os.getenv("COLBERT_MODEL", "colbert-ir/colbertv2.0")
    colbert_dim: int = int(os.getenv("COLBERT_DIM", "128"))

    # Extracted diagrams storage
    diagrams_dir: str = os.getenv("DIAGRAMS_DIR", "data/diagrams")


settings = Settings()
