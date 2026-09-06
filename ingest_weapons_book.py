"""
ingest_weapons_book.py — standalone, re-runnable ingestion for DoD's own
"Program Acquisition Costs by Weapon System" (informally "the Weapons Book").

This is the SECOND data source, plugging in exactly where ingest_gao.py's
TODO comments said it would — same pipeline shape, same VectorStore
wrapper, a completely separate Chroma collection so the two sources never
mix.

Why the chunking differs from ingest_gao.py:
    GAO's report self-labels every profile with a clean "Common Name:"
    field. The Weapons Book doesn't — program names are woven into prose
    in wildly different sentence shapes ("The X program procures...",
    "X are uncrewed systems...", "To address gaps in..."). Regex-extracting
    a perfect name from every page is fragile and would undermine the one
    thing that actually matters: RELIABLE citations.

    So this script anchors citations on what the document DOES label
    reliably — its own section numbers (e.g. "5-12") and category headers
    (e.g. "Missiles and Munitions") — and treats the program name as a
    best-effort label only. If that guess fails, the citation (category +
    section + page) is still exact.

Run it:            python ingest_weapons_book.py
Force a rebuild:   python ingest_weapons_book.py --rebuild
"""

import argparse
import re
from pathlib import Path

from pypdf import PdfReader

from config import WEAPONS_BOOK_YEARS, WEAPONS_BOOK_COLLECTION_NAME
from page_images import render_chunk_pages
from pdf_fetch import download_pdf
from vector_store import VectorStore

SECTION_RE = re.compile(r"^\d+-\d+$")

# Best-effort program-name guesser — tries common sentence openings. If
# nothing matches, build_chunks() falls back to the category name, which
# is always exact (never a guess that could be wrong).
NAME_RE = re.compile(
    r"^(?:The\s+)?([A-Z][\w\-\./&(),\u2019\u2018\s]{2,90}?)"
    r"(?:\s+(?:is|are|program|provides?|procures?|consists?|comprises?|replaces?|portfolio|element|class|will)\b)"
)


def guess_program_name(body_text: str, category: str) -> str:
    """Best-effort program name from the page's opening sentence; falls
    back to the (always-reliable) category name if nothing matches."""
    m = NAME_RE.match(body_text.strip())
    if m:
        name = re.sub(r"\s+", " ", m.group(1)).strip(" ,")
        if len(name) >= 3:
            return name
    return f"{category} (overview)"


def build_chunks(pdf_path: Path, fiscal_year: str) -> list[dict]:
    """
    Parse the PDF and chunk it by SECTION NUMBER: every content page
    carries a small "N-M" label (e.g. "5-12") in its header; consecutive
    pages sharing the same label are grouped into one chunk (a program's
    budget table occasionally spills onto a second page with the same
    label).

    Every chunk's metadata:
        program     - best-effort label (see guess_program_name)
        category    - the chapter/category heading (always exact)
        section     - the document's own "N-M" section number (always exact)
        page_start / page_end - PDF page numbers (always exact)
        source      - which document this came from
        fiscal_year - "FY2024" etc. — lets retrieval scope to a year
    """
    reader = PdfReader(str(pdf_path))
    n_pages = len(reader.pages)
    print(f"Parsed PDF: {n_pages} pages")

    page_texts = [p.extract_text() or "" for p in reader.pages]

    page_info = []
    for text in page_texts:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        section, category, body = None, None, ""
        for idx, line in enumerate(lines[:4]):
            if SECTION_RE.fullmatch(line):
                section = line
                category = lines[idx - 1] if idx > 0 else "Weapons Book"
                body = " ".join(lines[idx + 1: idx + 4])
                break
        page_info.append((section, category, body))

    chunks = []

    def add_chunk(start, end, section, category, body):
        text = "\n".join(page_texts[p] for p in range(start, end + 1))
        if len(text.strip()) < 100:
            return
        # Year-prefixed for the same reason as ingest_gao.py: two different
        # years' PDFs can share a section number on a different page, and
        # an un-prefixed id would let one year's chunk silently overwrite
        # another's.
        chunk_id = f"weapons-{fiscal_year.lower()}-p{start + 1:04d}"
        image_paths = render_chunk_pages(
            pdf_path, f"weapons_book-{fiscal_year.lower()}", chunk_id,
            page_start=start + 1, page_end=end + 1,
        )
        chunks.append({
            "id": chunk_id,
            "text": text,
            "metadata": {
                "program": guess_program_name(body, category),
                "category": category,
                "section": section,
                "page_start": start + 1,
                "page_end": end + 1,
                "source": f"DoD {fiscal_year} Program Acquisition Costs by Weapon System (Weapons Book)",
                "fiscal_year": fiscal_year,
                "image_paths": "|".join(image_paths),
            },
        })

    i, n_programs = 0, 0
    while i < n_pages:
        section, category, body = page_info[i]
        if section is None:
            i += 1
            continue  # front matter / blank pages — not part of the corpus
        j = i
        while j + 1 < n_pages and page_info[j + 1][0] == section:
            j += 1
        add_chunk(i, j, section, category, body)
        n_programs += 1
        i = j + 1

    print(f"Built {len(chunks)} chunks ({n_programs} sections across the document)")
    return chunks


def main():
    parser = argparse.ArgumentParser(description="Ingest DoD Weapons Books (all configured years) into Chroma")
    parser.add_argument("--rebuild", action="store_true",
                         help="wipe the existing collection and re-embed every year from scratch")
    parser.add_argument(
        "--year", action="append", dest="years", choices=list(WEAPONS_BOOK_YEARS),
        help="ingest only this fiscal year (repeatable); default is all configured years",
    )
    args = parser.parse_args()
    years = args.years or list(WEAPONS_BOOK_YEARS)

    store = VectorStore(collection_name=WEAPONS_BOOK_COLLECTION_NAME)
    if args.rebuild:
        print("--rebuild: wiping existing collection")
        store.reset()

    for fiscal_year in years:
        entry = WEAPONS_BOOK_YEARS[fiscal_year]
        print(f"\n=== {fiscal_year} ===")
        download_pdf(entry["path"], entry["url"], label=f"Weapons Book {fiscal_year}")

        if not args.rebuild and store.has_metadata_value("fiscal_year", fiscal_year):
            print(f"{fiscal_year} already ingested — skipping (use --rebuild to redo it).")
            continue

        chunks = build_chunks(entry["path"], fiscal_year)
        print(f"Embedding and storing {fiscal_year}...")
        store.add_documents(
            texts=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
            ids=[c["id"] for c in chunks],
        )
        print(f"{fiscal_year} done — {len(chunks)} chunks added.")

    print(f"\nCollection now holds {store.count()} chunks total in data/chroma/.")


if __name__ == "__main__":
    main()