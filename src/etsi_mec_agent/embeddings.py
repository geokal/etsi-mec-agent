"""Every vector-producing decision lives here: which runtime embeds, and the
vector-space signature the collection identity guard compares.

`EMBEDDING_MODE` is the single switch; unset (or "local") is the FastEmbed ONNX path,
byte-for-byte what the pipeline shipped before providers existed.
"""

import functools
import re as _re
import zlib

from fastembed import LateInteractionTextEmbedding, TextEmbedding
from qdrant_client import models

from etsi_mec_agent.config import settings


# lru_cache: Streamlit reruns the whole script per interaction and the ONNX embedders cost
# ~30–60 s cold; the CLI calls this once per process either way.
@functools.lru_cache(maxsize=1)
def get_embedders():
    """Load CPU-optimized ONNX models for Dense and ColBERT embeddings, once per process."""
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
        dense_model, colbert_model = get_embedders()
        return {
            "dense": list(dense_model.embed([text]))[0].tolist(),
            "colbert": list(colbert_model.query_embed(text))[0].tolist(),
            "sparse": _token_sparse(text),
        }

    def embed_dense_batch(self, texts) -> list:
        dense_model, _ = get_embedders()
        return list(dense_model.embed(texts))

    def embed_colbert_batch(self, texts) -> list:
        _, colbert_model = get_embedders()
        return list(colbert_model.passage_embed(texts))

    def identity(self) -> dict:
        """The vector-space signature: two collections match only if every key matches."""
        return {
            "mode": "local",
            "dense_model": settings.dense_model,
            "dense_dim": settings.dense_dim,
            "colbert_model": settings.colbert_model,
            "colbert_dim": settings.colbert_dim,
            "sparse": "crc32-term-frequency",
        }


@functools.lru_cache(maxsize=1)
def get_provider():
    """The one mode branch in the codebase."""
    if settings.embedding_mode in ("", "local"):
        return OnnxProvider()
    raise ValueError(
        f"EMBEDDING_MODE={settings.embedding_mode!r} is not a supported runtime "
        "(only 'local' ships so far)"
    )
