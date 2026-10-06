import argparse
import hashlib
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import List

import pymupdf
import pymupdf4llm
from fastembed import LateInteractionTextEmbedding, TextEmbedding
from qdrant_client.models import PointStruct

from etsi_mec_agent.chunking import chunk_page
from etsi_mec_agent.config import settings
from etsi_mec_agent.identity import stamp as identity_stamp
from etsi_mec_agent.search import _token_sparse
from etsi_mec_agent.store import ensure_collection, get_qdrant_client

# Ensure stdout flushes immediately in real-time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


def get_embedders():
    """Load CPU-optimized ONNX models for Dense and ColBERT embeddings."""
    print(f"Loading Dense embedder ({settings.dense_model})...", flush=True)
    dense_model = TextEmbedding(settings.dense_model)

    print(f"Loading ColBERT late-interaction embedder ({settings.colbert_model})...", flush=True)
    colbert_model = LateInteractionTextEmbedding(settings.colbert_model)

    return dense_model, colbert_model


def extract_primary_heading(markdown_text: str) -> str:
    """Extract the first markdown heading if present in the page text."""
    for line in markdown_text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return ""


_SEEN_IMAGE_HASHES: set = set()   # module-level: dedupes boilerplate images across all PDFs in one run


def _image_is_worth_keeping(path: Path) -> bool:
    """Return False (and delete the file) for blank, sub-200px, or byte-duplicate images."""
    from PIL import Image  # lazy import: Pillow arrives via fastembed

    keep = False
    try:
        with Image.open(path) as im:
            lo, hi = im.convert("L").getextrema()
            keep = im.width >= 200 and im.height >= 200 and (hi - lo) > 5
    except Exception:
        keep = False
    if keep:
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest in _SEEN_IMAGE_HASHES:
            keep = False
        else:
            _SEEN_IMAGE_HASHES.add(digest)
    if not keep:
        path.unlink(missing_ok=True)
    return keep


def _prune_bad_image_refs(text: str, diagrams_dir: Path) -> str:
    """Drop markdown image refs whose files were discarded by the quality filter."""

    def _keep(m):
        ref = Path(m.group(1))
        p = ref if ref.is_absolute() else diagrams_dir / ref.name
        return m.group(0) if _image_is_worth_keeping(p) else ""

    return re.sub(r"!\[.*?\]\((.*?)\)", _keep, text)

def _figure_size_ok(rect, page_rect) -> bool:
    if rect.width < 100 or rect.height < 60:
        return False  # rules, bullets, text-fragment clusters
    if rect.width > 0.95 * page_rect.width and rect.height > 0.95 * page_rect.height:
        return False  # page-spanning frames / watermarks
    return True


def _merged_tile_groups(page) -> list:
    """Bboxes of 2+ adjacent raster images: a big figure that pymupdf4llm
    exports as separate strips becomes one renderable rectangle again."""
    boxes = sorted(
        (pymupdf.Rect(im["bbox"]) for im in page.get_image_info()),
        key=lambda r: (r.y0, r.x0),
    )
    groups: list = []  # each: [rect, tile_count]
    for b in boxes:
        grown = pymupdf.Rect(b.x0 - 5, b.y0 - 5, b.x1 + 5, b.y1 + 5)
        if groups and grown.intersects(groups[-1][0]):
            groups[-1][0] |= b
            groups[-1][1] += 1
        else:
            groups.append([pymupdf.Rect(b), 1])
    return [g[0] for g in groups if g[1] > 1]


# ponytail: cluster_drawings also matches vector table grids, which pass as "figures";
# upgrade path is Docling's figure/table classification if the noise matters.
def render_vector_figures(pdf_path: Path, diagrams_dir: Path) -> dict:
    """Render each page's vector-drawing clusters and fragmented raster tile
    groups as whole PNGs.

    pymupdf4llm's write_images only exports embedded rasters, so ETSI's
    vector-drawn architecture diagrams (MEC002/003 reference models) need
    this separate render pass; large raster figures that arrive as stacked
    strips are re-merged into one image. Returns {page_number: [png paths]}.
    """
    figures: dict = {}
    doc = pymupdf.open(str(pdf_path))
    try:
        for pno in range(len(doc)):
            page = doc[pno]
            pr = page.rect
            rects = [r for r in page.cluster_drawings() if _figure_size_ok(r, pr)]
            rects += [r for r in _merged_tile_groups(page) if _figure_size_ok(r, pr)]
            for idx, rect in enumerate(rects):
                out = diagrams_dir / f"{pdf_path.name}-{pno + 1:04d}-fig-{idx:02d}.png"
                page.get_pixmap(dpi=200, clip=rect).save(str(out))
                if _image_is_worth_keeping(out):
                    figures.setdefault(pno + 1, []).append(str(out))
    finally:
        doc.close()
    return figures


def ingest_single_pdf(
    pdf_path: Path,
    dense_model: TextEmbedding,
    colbert_model: LateInteractionTextEmbedding,
    client,
    batch_size: int = 16,      # raised from 4 → 16: ~3-4x faster on CPU ONNX, still safe
    extract_diagrams: bool = True,
    skip_existing: bool = False,
    dry_run: bool = False,
    show_chunks: int = 0,
) -> int:
    """Extract markdown from a PDF using pymupdf4llm, crop diagrams, chunk safely, compute dual vectors, and upsert."""
    doc_name = pdf_path.stem
    # Identity is read off the cover, never off the filename: 59 of this corpus's filenames name
    # a spec whose text they do not hold. is_current starts True for everything — which edition a
    # spec number is on is only knowable once the whole corpus is in, so
    # scripts/backfill_spec_identity.py re-decides it as a post-ingest pass.
    ident = identity_stamp(pdf_path)
    ident["content_md5"] = hashlib.md5(pdf_path.read_bytes()).hexdigest()
    t0 = time.time()

    if skip_existing:
        # Check if this doc_id already has points in the collection
        existing, _ = client.scroll(
            collection_name=settings.qdrant_index,
            scroll_filter=__import__("qdrant_client").models.Filter(
                must=[__import__("qdrant_client").models.FieldCondition(
                    key="doc_id",
                    match=__import__("qdrant_client").models.MatchValue(value=doc_name),
                )]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        if existing:
            print(f"    [SKIP] {pdf_path.name} already indexed ({doc_name}). Use --recreate-index to force.", flush=True)
            return 0

    vector_figs: dict = {}
    if extract_diagrams:
        diagrams_dir = Path(settings.diagrams_dir)
        diagrams_dir.mkdir(parents=True, exist_ok=True)
        print(f"    Extracting Markdown & cropping diagrams with PyMuPDF4LLM across {pdf_path.name}...", flush=True)
        try:
            page_chunks = pymupdf4llm.to_markdown(
                str(pdf_path),
                page_chunks=True,
                write_images=True,
                image_path=str(diagrams_dir),
                image_format="png",
                image_size_limit=0.05,
                dpi=150,
            )
            print(f"    Parsed {len(page_chunks)} pages successfully. Preparing chunks...", flush=True)
        except Exception as e:
            print(f"    [ERROR] Failed parsing {pdf_path.name} with PyMuPDF: {e}", flush=True)
            return 0
        if not dry_run:
            vector_figs = render_vector_figures(pdf_path, diagrams_dir)
            n_vec = sum(len(v) for v in vector_figs.values())
            if n_vec:
                print(f"    Rendered {n_vec} vector figure(s) from drawing clusters.", flush=True)
    else:
        print(f"    Extracting Markdown (text-only) with PyMuPDF4LLM across {pdf_path.name}...", flush=True)
        try:
            page_chunks = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
            print(f"    Parsed {len(page_chunks)} pages successfully. Preparing chunks...", flush=True)
        except Exception as e:
            print(f"    [ERROR] Failed parsing {pdf_path.name} with PyMuPDF: {e}", flush=True)
            return 0

    texts_to_embed = []   # what gets embedded: context prefix + body
    bodies = []           # what gets stored and read back: the page markdown unchanged
    metadata_list = []

    for chunk in page_chunks:
        raw_text = chunk.get("text", "").strip()
        if not raw_text:
            continue

        page_meta = chunk.get("metadata", {})
        page_num = page_meta.get("page_number", page_meta.get("page", 0) + 1)
        heading = extract_primary_heading(raw_text)

        if extract_diagrams and not dry_run:
            raw_text = _prune_bad_image_refs(raw_text, diagrams_dir)
        vec_paths = vector_figs.get(page_num, [])

        # Row- and clause-aware packing: a table row is never cut and every table chunk keeps its
        # header, replacing the word window that flattened a page into pipe soup.
        page_parts = chunk_page(raw_text, max_words=300, overlap=40)
        for sub_idx, pc in enumerate(page_parts, 1):
            diagram_paths = re.findall(r'!\[.*?\]\((.*?)\)', pc.text)
            if sub_idx == 1:
                # page-level vector renders attach once, not to every sub-chunk
                diagram_paths = diagram_paths + vec_paths
            # Stored as POSIX so a Linux consumer of the same payload resolves the file.
            diagram_paths = [Path(p).as_posix() for p in diagram_paths]
            has_diagram = len(diagram_paths) > 0

            # B3: the document side carries the context a bare clause number needs; the query
            # side has nothing to add, so the two are deliberately asymmetric.
            context = " ".join(filter(None, [ident["spec_id"], ident["edition"], pc.clause, heading]))
            texts_to_embed.append(f"{context}\n{pc.text}" if context else pc.text)
            bodies.append(pc.text)
            metadata_list.append({
                "doc_id": doc_name,
                "filename": pdf_path.name,
                "page": page_num,
                "chunk_part": sub_idx,
                "total_parts": len(page_parts),
                "heading": heading,
                "clause": pc.clause,
                "block_kind": pc.block_kind,
                "has_diagram": has_diagram,
                "diagram_paths": diagram_paths,
                **ident,
            })

    if not texts_to_embed:
        print(f"    [WARNING] No text extracted from {pdf_path.name}", flush=True)
        return 0

    total_chunks = len(texts_to_embed)
    total_batches = (total_chunks + batch_size - 1) // batch_size
    diagram_chunks = sum(1 for m in metadata_list if m["has_diagram"])

    if dry_run:
        print(f"    [DRY-RUN] Would index {total_chunks} chunk(s) ({diagram_chunks} with diagrams) "
              f"across {len(page_chunks)} page(s) from '{pdf_path.name}'. No embeddings, no upsert.", flush=True)
        for meta, body in list(zip(metadata_list, bodies))[:show_chunks]:
            print(f"      --- page {meta['page']} part {meta['chunk_part']}/{meta['total_parts']} "
                  f"[{meta['block_kind']}, clause {meta['clause'] or '-'}, "
                  f"{meta['spec_id']} {meta['edition'] or 'unnumbered'}]", flush=True)
            for line in body.splitlines():
                print(f"      {line[:160]}", flush=True)
        return 0

    print(f"    Generated {total_chunks} chunk(s) ({diagram_chunks} containing diagrams). "
          f"Computing Dense & ColBERT embeddings in {total_batches} batch(es) of {batch_size}...", flush=True)

    indexed = 0

    # Stream: embed → upsert one batch at a time — never holds all vectors in RAM
    for b_idx, i in enumerate(range(0, len(texts_to_embed), batch_size), 1):
        batch_texts = texts_to_embed[i : i + batch_size]
        batch_bodies = bodies[i : i + batch_size]
        batch_meta  = metadata_list[i : i + batch_size]

        if b_idx % 5 == 0 or b_idx == 1 or b_idx == total_batches:
            print(f"      [Progress] Batch {b_idx}/{total_batches} "
                  f"(chunk {min(i + batch_size, total_chunks)}/{total_chunks})...", flush=True)

        try:
            dense_embeddings   = list(dense_model.embed(batch_texts))
            colbert_embeddings = list(colbert_model.passage_embed(batch_texts))
        except Exception as e:
            print(f"    [WARNING] Batch embedding error: {e}. Trying individual fallback...", flush=True)
            dense_embeddings   = []
            colbert_embeddings = []
            valid_indices      = []
            for idx, single_text in enumerate(batch_texts):
                try:
                    d = list(dense_model.embed([single_text]))[0]
                    c = list(colbert_model.passage_embed([single_text]))[0]
                    dense_embeddings.append(d)
                    colbert_embeddings.append(c)
                    valid_indices.append(idx)
                except Exception as inner_e:
                    print(f"    [SKIPPED] Oversized chunk on page {batch_meta[idx].get('page')}: {inner_e}", flush=True)
            batch_texts = [batch_texts[idx] for idx in valid_indices]
            batch_bodies = [batch_bodies[idx] for idx in valid_indices]
            batch_meta  = [batch_meta[idx]  for idx in valid_indices]

        # Embedded text and stored text differ on purpose (B3): the vector side carries the spec
        # and clause context a bare "7.2.1" needs, the payload side stays the page markdown.
        batch_points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector={"dense": d_vec.tolist(), "colbert": c_vec.tolist(),
                        "sparse": _token_sparse(embed_text)},
                payload={"text": body, **meta},
            )
            for embed_text, body, meta, d_vec, c_vec in zip(
                batch_texts, batch_bodies, batch_meta, dense_embeddings, colbert_embeddings
            )
        ]

        # Upsert immediately — no global points list needed
        if batch_points:
            client.upsert(collection_name=settings.qdrant_index, points=batch_points)
            indexed += len(batch_points)

    elapsed = time.time() - t0
    print(f"    --> Indexed {indexed} chunk(s) ({diagram_chunks} with diagrams) "
          f"across {len(page_chunks)} page(s) in {elapsed:.1f}s into '{settings.qdrant_index}'.", flush=True)
    return indexed



def ingest_pdfs(
    sources: List[Path],
    recreate_index: bool = False,
    extract_diagrams: bool = True,
    skip_existing: bool = False,
    dry_run: bool = False,
    show_chunks: int = 0,
) -> int:
    """Ingest a list of ETSI MEC PDFs using PyMuPDF4LLM and Hybrid Qdrant."""
    valid_files = [p.resolve() for p in sources if p.is_file() and p.suffix.lower() == ".pdf"]

    if not valid_files:
        print("[WARNING] No valid PDF files found to ingest.", flush=True)
        return 0

    if not dry_run:
        ensure_collection(recreate=recreate_index)
    client = get_qdrant_client()
    # Dry-run: skip model loading entirely — nothing gets embedded or written.
    dense_model, colbert_model = (None, None) if dry_run else get_embedders()

    if recreate_index:
        # A wipe empties the collection, so --skip-existing would skip nothing;
        # normalize the flag so the per-doc scroll check can't fire against the pre-wipe data.
        if skip_existing:
            print("[INFO] --recreate-index wipes the collection first, so --skip-existing has nothing to skip.", flush=True)
        skip_existing = False
    if dry_run and recreate_index:
        try:
            points = client.get_collection(settings.qdrant_index).points_count
            print(f"[DRY-RUN] A real run would DELETE '{settings.qdrant_index}' ({points} points) before re-ingesting.", flush=True)
        except Exception:
            print(f"[DRY-RUN] A real run would DELETE '{settings.qdrant_index}' (if present) before re-ingesting.", flush=True)

    total_chunks = 0
    total_start = time.time()

    print(f"\nIngesting {len(valid_files)} ETSI MEC PDF(s) with PyMuPDF4LLM + ColBERT:", flush=True)
    print("=" * 70, flush=True)

    seen_content: dict[str, str] = {}   # md5 -> filename: these 106 files are 54 documents
    skipped = 0
    for idx, pdf_path in enumerate(valid_files, 1):
        digest = hashlib.md5(pdf_path.read_bytes()).hexdigest()
        if digest in seen_content:
            skipped += 1
            print(f"[{idx}/{len(valid_files)}] [SKIP] {pdf_path.name} is byte-identical to "
                  f"{seen_content[digest]}", flush=True)
            continue
        seen_content[digest] = pdf_path.name
        print(f"[{idx}/{len(valid_files)}] {pdf_path.name}", flush=True)
        chunks = ingest_single_pdf(
            pdf_path, dense_model, colbert_model, client,
            extract_diagrams=extract_diagrams,
            skip_existing=skip_existing,
            dry_run=dry_run,
            show_chunks=show_chunks,
        )
        total_chunks += chunks

    total_time = time.time() - total_start
    print("=" * 70, flush=True)
    print(f"[COMPLETED] Indexed {total_chunks} chunks from {len(valid_files) - skipped} documents "
          f"({skipped} byte-identical duplicates skipped) in {total_time:.1f}s.", flush=True)
    if not dry_run:
        print("[NEXT] Identity starts out all-current: run "
              "`uv run python scripts/backfill_spec_identity.py --apply` to mark superseded editions.",
              flush=True)
    return total_chunks


def main():
    parser = argparse.ArgumentParser(description="Ingest ETSI MEC PDFs into Qdrant using PyMuPDF4LLM & ColBERT.")
    parser.add_argument(
        "path",
        type=str,
        nargs="?",
        default="data/specs",
        help="Path to PDF or directory (default: data/specs)",
    )
    parser.add_argument(
        "--recreate-index",
        action="store_true",
        help="Recreate Qdrant collection before ingestion (clears old data)",
    )
    parser.add_argument(
        "--no-diagrams",
        action="store_true",
        help="Disable physical diagram image extraction (text-only mode)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip PDFs whose doc_id already has points in Qdrant (safe for incremental updates)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk only: report what would be indexed (honors --skip-existing), write nothing to Qdrant",
    )
    parser.add_argument(
        "--show-chunks",
        type=int,
        default=0,
        metavar="N",
        help="With --dry-run, print the first N candidate chunks per document (no models, no writes)",
    )

    args = parser.parse_args()
    target_path = Path(args.path)

    if not target_path.exists():
        print(f"[ERROR] Path does not exist: {target_path}", file=sys.stderr)
        sys.exit(1)

    if target_path.is_file():
        pdf_files = [target_path]
    else:
        pdf_files = sorted(list(target_path.glob("*.pdf")))

    if not pdf_files:
        print(f"[WARNING] No PDF files found in {target_path.resolve()}", file=sys.stderr)
        sys.exit(0)

    ingest_pdfs(
        pdf_files,
        recreate_index=args.recreate_index,
        extract_diagrams=not args.no_diagrams,
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
        show_chunks=args.show_chunks,
    )


if __name__ == "__main__":
    main()

