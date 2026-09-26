"""
tools/clip_embed.py

CLIP (Contrastive Language–Image Pre-Training) embedder for ETSI MEC diagram images.

- Runs fully on CPU via PyTorch (no NVIDIA GPU required).
- Uses openai/clip-vit-base-patch32 (~600 MB, downloaded once and cached).
- Produces a 512-dim float vector per image that can be stored as a named
  Qdrant vector alongside the existing 'dense' and 'colbert' vectors.

Usage in Python:
    from etsi_mec_agent.tools.clip_embed import embed_images, embed_text_for_image_search
    vecs = embed_images(["data/diagrams/MEC099.pdf-0020-13.png"])
    # vecs[0] is a list of 512 floats

Usage from CLI (test a single image):
    uv run python -m etsi_mec_agent.tools.clip_embed data/diagrams/MEC099.pdf-0020-13.png
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Lazy model singleton — loads once per process
# ---------------------------------------------------------------------------
_model = None
_processor = None


def _load_clip():
    """Load the CLIP model and processor (cached after first call)."""
    global _model, _processor
    if _model is not None:
        return _model, _processor

    try:
        from transformers import CLIPModel, CLIPProcessor
        import torch  # noqa: F401  (just verify torch is present)
    except ImportError as exc:
        raise ImportError(
            "CLIP requires transformers and torch.\n"
            "Install with:  uv add transformers torch Pillow"
        ) from exc

    print("[CLIP] Loading openai/clip-vit-base-patch32 (first run downloads ~600 MB)…", flush=True)
    _model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    _model.eval()
    print("[CLIP] Model ready.", flush=True)
    return _model, _processor


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_images(image_paths: List[str]) -> List[Optional[List[float]]]:
    """
    Embed a batch of image file paths with CLIP.

    Returns a list of the same length as *image_paths*.
    Each element is either a list of 512 floats (the CLIP image embedding)
    or ``None`` if the file could not be opened.
    """
    import torch
    from PIL import Image

    model, processor = _load_clip()

    results: List[Optional[List[float]]] = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[CLIP] Cannot open {path}: {e}", flush=True)
            results.append(None)
            continue

        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            vec = model.get_image_features(**inputs)   # (1, 512)
            # L2-normalise so cosine similarity == dot product
            vec = vec / vec.norm(dim=-1, keepdim=True)
        results.append(vec[0].cpu().tolist())

    return results


def embed_text_for_image_search(text: str) -> List[float]:
    """
    Embed a natural-language text query into the CLIP joint embedding space
    so it can be compared directly against image embeddings.

    Example:
        vec = embed_text_for_image_search("sequence diagram showing Mp1 handshake")
        # Use this vector to query the 'clip' named vector in Qdrant
    """
    import torch

    model, processor = _load_clip()
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        vec = model.get_text_features(**inputs)         # (1, 512)
        vec = vec / vec.norm(dim=-1, keepdim=True)
    return vec[0].cpu().tolist()


# ---------------------------------------------------------------------------
# Quick CLI test
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python -m etsi_mec_agent.tools.clip_embed <image_path> [<image_path2> …]")
        sys.exit(1)

    paths = sys.argv[1:]
    vecs = embed_images(paths)
    for path, vec in zip(paths, vecs):
        if vec is None:
            print(f"[FAIL] {path}")
        else:
            print(f"[OK]   {path}  →  dim={len(vec)}  norm≈{sum(v*v for v in vec)**0.5:.4f}")


if __name__ == "__main__":
    main()
