"""
scripts/graph_visualize.py

Builds a semantic relationship graph across all ETSI MEC spec chunks:
  - Nodes = MEC specification documents (e.g. MEC001, MEC002 …)
  - Edges = semantic similarity between specs (thicker edge = more similar)
  - Layout = force-directed (similar specs pulled together)

Run with:
    uv run python scripts/graph_visualize.py
    uv run python scripts/graph_visualize.py --top-k 8 --threshold 0.6 --limit 3000
"""

import argparse
import collections
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
MISSING = []
try:
    import numpy as np
except ImportError:
    MISSING.append("numpy")
try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
except ImportError:
    MISSING.append("matplotlib")
try:
    import networkx as nx
except ImportError:
    MISSING.append("networkx")

if MISSING:
    print(f"[ERROR] Missing packages: {', '.join(MISSING)}")
    print(f"       Run:  uv add {' '.join(MISSING)}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from etsi_mec_agent.config import settings
from etsi_mec_agent.store import get_qdrant_client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def scroll_all_points(client, limit: int):
    """Scroll through the collection and return up to `limit` points with dense vectors."""
    points, offset = [], None
    batch = min(256, limit)
    while len(points) < limit:
        batch_size = min(batch, limit - len(points))
        result, offset = client.scroll(
            collection_name=settings.qdrant_index,
            limit=batch_size,
            offset=offset,
            with_vectors=["dense"],
            with_payload=True,
        )
        points.extend(result)
        if offset is None:
            break
    return points


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def build_doc_graph(points, top_k: int, threshold: float):
    """
    Build a weighted graph at the *document* level.

    Strategy:
      1. For each chunk find its top-K nearest chunks from OTHER docs
         (using cosine similarity over the dense vectors).
      2. Accumulate similarity scores per (doc_a, doc_b) pair.
      3. Edge weight = mean similarity of the top connections between two docs.
    """
    # Group vectors by doc_id
    doc_vecs: dict[str, list] = collections.defaultdict(list)
    for p in points:
        doc_id = p.payload.get("doc_id", "UNKNOWN")
        vec = p.vector.get("dense") if isinstance(p.vector, dict) else None
        if vec:
            doc_vecs[doc_id].append(np.array(vec, dtype="float32"))

    doc_ids = sorted(doc_vecs.keys())
    print(f"[graph] {len(doc_ids)} distinct docs, {len(points)} chunks loaded")

    # Compute doc-level mean vectors for a quick overview graph
    doc_means: dict[str, np.ndarray] = {}
    for doc_id, vecs in doc_vecs.items():
        mat = np.stack(vecs)
        doc_means[doc_id] = mat.mean(axis=0)

    # Build edge weights between all pairs of docs
    G = nx.Graph()
    G.add_nodes_from(doc_ids)

    for i, doc_a in enumerate(doc_ids):
        for doc_b in doc_ids[i + 1:]:
            sim = cosine_similarity(doc_means[doc_a], doc_means[doc_b])
            if sim >= threshold:
                G.add_edge(doc_a, doc_b, weight=sim)

    # Also add chunk-level top-K edges for richness
    # (sample at most 200 chunks per doc to keep it tractable)
    print("[graph] computing chunk-level top-K cross-doc connections …")
    pair_sims: dict[tuple, list] = collections.defaultdict(list)

    for doc_a, vecs_a in doc_vecs.items():
        sample_a = vecs_a[:200]
        for doc_b, vecs_b in doc_vecs.items():
            if doc_a >= doc_b:
                continue
            sample_b = vecs_b[:200]
            # Matrix multiply for fast pairwise cosine
            mat_a = np.stack(sample_a)
            mat_b = np.stack(sample_b)
            norms_a = np.linalg.norm(mat_a, axis=1, keepdims=True) + 1e-9
            norms_b = np.linalg.norm(mat_b, axis=1, keepdims=True) + 1e-9
            sims = (mat_a / norms_a) @ (mat_b / norms_b).T  # (len_a, len_b)
            # Top-K per row
            top_sims = np.sort(sims, axis=1)[:, -top_k:].flatten()
            pair_sims[(doc_a, doc_b)].extend(top_sims.tolist())

    for (doc_a, doc_b), sims in pair_sims.items():
        mean_sim = float(np.mean(sims))
        if mean_sim >= threshold:
            if G.has_edge(doc_a, doc_b):
                # average the two estimates
                old = G[doc_a][doc_b]["weight"]
                G[doc_a][doc_b]["weight"] = (old + mean_sim) / 2
            else:
                G.add_edge(doc_a, doc_b, weight=mean_sim)

    return G


def draw_graph(G: nx.Graph, output_path: str):
    if G.number_of_nodes() == 0:
        print("[graph] No nodes – nothing to draw.")
        return

    print(f"[graph] {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Node sizes proportional to degree
    degrees = dict(G.degree())
    max_deg = max(degrees.values()) if degrees else 1
    node_sizes = [300 + 1200 * (degrees[n] / max_deg) for n in G.nodes()]

    # Edge widths proportional to weight
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(weights) if weights else 1
    edge_widths = [0.5 + 4.0 * (w / max_w) for w in weights]
    edge_alphas = [0.3 + 0.5 * (w / max_w) for w in weights]

    # Color nodes by doc family (MEC0xx → group by tens)
    color_map = matplotlib.colormaps["tab20"]
    def node_color(doc_id):
        digits = "".join(c for c in doc_id if c.isdigit())
        group = (int(digits) // 10) % 20 if digits else 0
        return color_map(group)
    node_colors = [node_color(n) for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    pos = nx.spring_layout(G, weight="weight", k=2.5, seed=42, iterations=100)

    # Draw edges with per-edge alpha
    for (u, v), width, alpha in zip(G.edges(), edge_widths, edge_alphas):
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)],
            width=width, alpha=alpha,
            edge_color="white", ax=ax,
        )

    nx.draw_networkx_nodes(
        G, pos, node_size=node_sizes,
        node_color=node_colors, alpha=0.9, ax=ax,
    )
    nx.draw_networkx_labels(
        G, pos, font_size=7, font_color="white",
        font_weight="bold", ax=ax,
    )

    # Edge weight labels (optional – comment out if too noisy)
    edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels,
        font_size=5, font_color="#aaaaaa", ax=ax,
    )

    ax.set_title(
        "ETSI MEC Specification Semantic Relationship Graph\n"
        "(node size = connectivity, edge thickness = semantic similarity)",
        color="white", fontsize=13, pad=20,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[graph] saved → {output_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ETSI MEC semantic relationship graph")
    parser.add_argument("--limit",     type=int,   default=3000,  help="Max chunks to load from Qdrant (default: 3000)")
    parser.add_argument("--top-k",     type=int,   default=5,     help="Top-K chunk pairs per doc pair (default: 5)")
    parser.add_argument("--threshold", type=float, default=0.55,  help="Min cosine similarity to draw an edge (default: 0.55)")
    parser.add_argument("--output",    type=str,   default="etsi_mec_graph.png", help="Output image file")
    args = parser.parse_args()

    client = get_qdrant_client()
    print(f"[graph] loading up to {args.limit} chunks from Qdrant …")
    points = scroll_all_points(client, args.limit)
    print(f"[graph] loaded {len(points)} chunks")

    G = build_doc_graph(points, top_k=args.top_k, threshold=args.threshold)
    draw_graph(G, args.output)


if __name__ == "__main__":
    main()
