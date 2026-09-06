"""
page_images.py — renders PDF pages to PNG files, once, for multimodal use.

Shared by every ingest_*.py script: turning a page range into images is
identical regardless of which source document it's for. Images are cached
on disk exactly like everything else in this project (embeddings, the
downloaded PDFs) — re-running ingestion skips any page whose image already
exists.

Why PyMuPDF and not pypdf: pypdf can only pull OUT images the PDF already
embeds as separate raster objects — it can't reliably reconstruct a
vector-drawn chart (lines, shapes, and text drawn directly on the page,
which is how most PDF-generated charts are actually built). PyMuPDF instead
renders the whole page exactly as a PDF viewer would — a screenshot of the
real page, charts and all, regardless of how they were drawn.
"""

from pathlib import Path

import pymupdf

from config import IMAGE_DPI, IMAGES_DIR


def render_chunk_pages(
    pdf_path: Path, source_name: str, chunk_id: str, page_start: int, page_end: int
) -> list[str]:
    """
    Render every page in [page_start, page_end] (1-based, inclusive) of
    pdf_path to a PNG under data/images/<source_name>/. Returns the saved
    paths as STRINGS (Chroma metadata only accepts simple scalar types —
    the caller joins this list into one "|"-delimited string to store).

    Skips rendering a page whose image file already exists — the cache.
    """
    out_dir = IMAGES_DIR / source_name
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    doc = None
    for page_num in range(page_start, page_end + 1):
        out_path = out_dir / f"{chunk_id}_p{page_num:04d}.png"
        if not out_path.exists():
            if doc is None:
                doc = pymupdf.open(str(pdf_path))
            pix = doc[page_num - 1].get_pixmap(dpi=IMAGE_DPI)  # PyMuPDF pages are 0-based
            pix.save(str(out_path))
        paths.append(str(out_path))
    if doc is not None:
        doc.close()
    return paths
