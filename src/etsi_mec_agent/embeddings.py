"""Every vector-producing decision lives here: which runtime embeds, and the
vector-space signature the collection identity guard compares.

`EMBEDDING_MODE` is the single switch; unset (or "local") is the FastEmbed ONNX path,
byte-for-byte what the pipeline shipped before providers existed.
"""

import functools
import re as _re
import zlib

from qdrant_client import models

from etsi_mec_agent.config import settings


# lru_cache: Streamlit reruns the whole script per interaction and the ONNX embedders cost
# ~30–60 s cold; the CLI calls this once per process either way.
@functools.lru_cache(maxsize=1)
def get_embedders():
    """Load CPU-optimized ONNX models for Dense and ColBERT embeddings, once per process."""
    # Lazy: the sparse-only scripts (migrate_add_sparse, check_migrate_sparse) import this
    # module without ever needing ONNX, and fastembed stays an optional package.
    try:
        from fastembed import LateInteractionTextEmbedding, TextEmbedding
    except ImportError as exc:
        raise ImportError("fastembed is required to embed locally (EMBEDDING_MODE=local)") from exc
    print(f"Loading Dense embedder ({settings.dense_model})...", flush=True)
    dense_model = TextEmbedding(settings.dense_model)

    print(f"Loading ColBERT late-interaction embedder ({settings.colbert_model})...", flush=True)
    colbert_model = LateInteractionTextEmbedding(settings.colbert_model)

    return dense_model, colbert_model


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split into word tokens."""
    return _re.findall(r"[a-z0-9]+", text.lower())


def _token_sparse(text: str) -> models.SparseVector:
    """Raw term-frequency sparse vector; token ids are crc32 hashes so no shared
    vocabulary file is needed between ingest and query time. Qdrant applies IDF
    weighting at query time via Modifier.IDF on the collection config."""
    tf: dict[int, int] = {}
    for tok in _tokenize(text):
        i = zlib.crc32(tok.encode()) & 0x7FFFFFFF
        tf[i] = tf.get(i, 0) + 1
    return models.SparseVector(indices=list(tf), values=[float(v) for v in tf.values()])


class OnnxProvider:
    """The local FastEmbed path, verbatim: this is what an unset EMBEDDING_MODE has always done."""

    def query_vectors(self, text: str) -> dict:
        """Raw query vectors: dense and ColBERT as plain lists, sparse as a SparseVector."""
        dense_model, colbert_model = get_embedders()
        return {
            "dense": list(dense_model.embed([text]))[0].tolist(),
            "colbert": list(colbert_model.query_embed(text))[0].tolist(),
            "sparse": _token_sparse(text),
        }

    def embed_dense_batch(self, texts) -> list:
        """Point-ready dense vectors: float lists that go straight into PointStruct.vector."""
        dense_model, _ = get_embedders()
        return [v.tolist() for v in dense_model.embed(texts)]

    def embed_colbert_batch(self, texts) -> list:
        """Point-ready ColBERT multivectors: one list of per-token vectors per text."""
        _, colbert_model = get_embedders()
        return [v.tolist() for v in colbert_model.passage_embed(texts)]

    def identity(self) -> dict:
        """The vector-space signature: two collections match only if every key matches."""
        return _identity("local")


def _identity(mode: str) -> dict:
    """Both providers' signature, read off the same settings so their keys always line up."""
    return {
        "mode": mode,
        "dense_model": settings.dense_model,
        "dense_dim": settings.dense_dim,
        "colbert_model": settings.colbert_model,
        "colbert_dim": settings.colbert_dim,
        "sparse": "crc32-term-frequency",
    }


class CloudInferenceProvider:
    """Server-side embeddings: text goes to the cluster, Qdrant Cloud makes the vectors.

    No ONNX loads here — the client sends `models.Document` and the cluster's inference service
    resolves it. `store.get_qdrant_client` must pass `cloud_inference=True` for that to happen:
    with the flag off, qdrant-client 1.19.1 embeds the same Documents *locally* through fastembed
    (qdrant_client.py:387), which needs the model on the host and quietly defeats the point.

    Sparse is still built here, client-side: the stored sparse vectors use crc32 token ids with
    server-side IDF, and the cloud's BM25 model would put new vectors in a different id space.
    """

    def query_vectors(self, text: str) -> dict:
        """Raw query vectors as Documents: the cluster embeds them, sparse stays client-side."""
        return {
            "dense": models.Document(text=text, model=settings.dense_model),
            "colbert": models.Document(text=text, model=settings.colbert_model),
            "sparse": _token_sparse(text),
        }

    def embed_dense_batch(self, texts) -> list:
        """Point-ready dense Documents: one per text, resolved by the cluster's inference."""
        return [models.Document(text=t, model=settings.dense_model) for t in texts]

    def embed_colbert_batch(self, texts) -> list:
        """Point-ready ColBERT Documents: the multivector is made server-side too."""
        return [models.Document(text=t, model=settings.colbert_model) for t in texts]

    def identity(self) -> dict:
        """The vector-space signature: the same keys as the local provider, mode 'cloud'."""
        return _identity("cloud")


@functools.lru_cache(maxsize=1)
def get_provider():
    """The one mode branch in the codebase."""
    if settings.embedding_mode in ("", "local"):
        return OnnxProvider()
    if settings.embedding_mode == "cloud":
        if not settings.qdrant_url:
            raise ValueError(
                "EMBEDDING_MODE=cloud needs QDRANT_URL (and QDRANT_API_KEY) pointing at a Qdrant "
                "Cloud cluster: a local Docker Qdrant has no inference service")
        return CloudInferenceProvider()
    raise ValueError(
        f"EMBEDDING_MODE={settings.embedding_mode!r} is not a supported runtime "
        "(use 'local' for FastEmbed ONNX or 'cloud' for Qdrant Cloud Inference)"
    )
