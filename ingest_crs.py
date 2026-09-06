"""
ingest_crs.py — standalone, re-runnable ingestion for the CRS report
"FY2026 Defense Budget: Funding for Selected Weapon Systems" (R48860).

This is the THIRD data source, and it needs a genuinely different chunking
strategy from the other two:
    - GAO (ingest_gao.py): one chunk PER PROGRAM (~62 profiles), detected
      from a "Common Name:" field.
    - Weapons Book (ingest_weapons_book.py): one chunk PER PROGRAM PAGE
      (~95 pages), detected from a "N-M" section number.
    - CRS (this file): one chunk PER CATEGORY SECTION (9 sections). This
      document is a continuous narrative report, not per-program
      profiles — "Aircraft and Related Weapon Systems" is one ~6-page
      section covering dozens of aircraft programs together, with a
      funding table threaded through the prose. Splitting it any finer
      would cut sentences and tables in half for no benefit.

Section boundaries below were VERIFIED against the actual PDF text (not
guessed from the table of contents' printed page numbers, which have a
+4 offset vs. the real PDF page index) — see the conversation history for
how these were confirmed. If CRS republishes this report with a different
layout, re-verify these page numbers before trusting them.

Run it:            python ingest_crs.py
Force a rebuild:   python ingest_crs.py --rebuild
"""

import argparse
import sys

import requests
from pypdf import PdfReader

from config import CRS_PDF_PATH, CRS_PDF_URL, CRS_COLLECTION_NAME
from page_images import render_chunk_pages
from vector_store import VectorStore

# (section_name, first_pdf_page) — 1-based, verified directly against the
# extracted text. Each section runs up to (but not including) the next
# one's start page; the last section runs to the end of the document.
SECTIONS = [
    ("Introduction", 5),
    ("Aircraft and Related Weapon Systems", 10),
    ("Communications and Space-Based Systems", 16),
    ("Ground Systems", 20),
    ("Hypersonic Weapons", 23),
    ("Missile Defense", 25),
    ("Missiles and Munitions", 28),
    ("Shipbuilding and Maritime Systems", 32),
    ("Appendix: FY2026 Funding in the FY2025 Reconciliation Law", 36),
]


def download_pdf():
    """Download the CRS report once. Skips if already on disk."""
    if CRS_PDF_PATH.exists():
        print(f"PDF already downloaded: {CRS_PDF_PATH}")
        return
    print(f"Downloading {CRS_PDF_URL} ...")
    resp = requests.get(CRS_PDF_URL, timeout=120)
    resp.raise_for_status()
    CRS_PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    CRS_PDF_PATH.write_bytes(resp.content)
    print(f"Saved {len(resp.content):,} bytes to {CRS_PDF_PATH}")


def build_chunks() -> list[dict]:
    """
    Slice the PDF into one chunk per entry in SECTIONS, using each
    section's verified start page and the next section's start page as
    the exclusive end boundary (last section runs to the document's end).
    """
    reader = PdfReader(str(CRS_PDF_PATH))
    n_pages = len(reader.pages)
    print(f"Parsed PDF: {n_pages} pages")

    page_texts = [p.extract_text() or "" for p in reader.pages]

    chunks = []
    for i, (name, start_page) in enumerate(SECTIONS):
        end_page = SECTIONS[i + 1][1] - 1 if i + 1 < len(SECTIONS) else n_pages
        start_idx, end_idx = start_page - 1, end_page - 1  # to 0-based

        text = "\n".join(page_texts[p] for p in range(start_idx, end_idx + 1))
        if len(text.strip()) < 100:
            continue

        chunk_id = f"crs-s{i:02d}"
        image_paths = render_chunk_pages(
            CRS_PDF_PATH, "crs", chunk_id, page_start=start_page, page_end=end_page
        )
        chunks.append({
            "id": chunk_id,
            "text": text,
            "metadata": {
                "program": name,   # a category section, not a single program —
                                    # same field name as the other sources so
                                    # tools.py's citation code needs no changes
                "page_start": start_page,
                "page_end": end_page,
                "source": "CRS Report R48860: FY2026 Defense Budget: Funding for Selected Weapon Systems",
                "image_paths": "|".join(image_paths),
            },
        })

    print(f"Built {len(chunks)} chunks (one per report section)")
    return chunks


def main():
    parser = argparse.ArgumentParser(description="Ingest the CRS report into Chroma")
    parser.add_argument("--rebuild", action="store_true",
                         help="wipe the existing collection and re-embed from scratch")
    args = parser.parse_args()

    download_pdf()

    store = VectorStore(collection_name=CRS_COLLECTION_NAME)
    if args.rebuild:
        print("--rebuild: wiping existing collection")
        store.reset()

    if store.count() > 0:
        print(f"Collection already has {store.count()} chunks — nothing to do.")
        print("(Use --rebuild if you changed the chunking logic.)")
        sys.exit(0)

    chunks = build_chunks()

    print("Embedding and storing...")
    store.add_documents(
        texts=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
        ids=[c["id"] for c in chunks],
    )
    print(f"Done. Collection now holds {store.count()} chunks in data/chroma/.")


if __name__ == "__main__":
    main()
