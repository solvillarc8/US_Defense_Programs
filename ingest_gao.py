"""
ingest_gao.py — standalone, re-runnable ingestion script for the GAO report.

Pipeline (matches the first diagram we drew):
    PDF  ->  parse pages  ->  chunk (~1 chunk per program profile)
         ->  embed + store in Chroma (embedding happens inside VectorStore)

Run it:            python ingest_gao.py
Force a rebuild:   python ingest_gao.py --rebuild

Design notes:
  * Re-runnable & cheap: if the collection already has documents, we skip
    everything (your embedding cache). --rebuild wipes and starts over.
  * One script per data source: next week we'll add ingest_weapons_book.py
    and ingest_crs.py as siblings following this exact same shape.
"""

import argparse
import re
from pathlib import Path

from pypdf import PdfReader

from config import GAO_YEARS, PAGES_PER_CHUNK
from page_images import render_chunk_pages
from pdf_fetch import download_pdf
from vector_store import VectorStore


def guess_program_name(page_text: str) -> str:
    """
    Best-effort guess of the program name for a chunk.

    GAO program profiles carry a literal "Common Name: <program>" field
    (e.g. "Common Name: CVN 78"), so we look for that first — it's the
    cleanest possible label. For non-profile pages (overview chapters,
    appendices) we fall back to the first plausible heading line; the page
    numbers in the metadata still give a valid citation either way.
    """
    # Preferred: the standardized "Common Name:" field on profile pages.
    match = re.search(r"Common Name:\s*([^\n]{2,60})", page_text)
    if match:
        return match.group(1).strip()

    # Fallback: first shortish heading-like line at the top of the chunk.
    for line in page_text.splitlines()[:8]:
        line = line.strip()
        if not line:
            continue
        if 3 < len(line) < 80 and re.search(r"[A-Za-z]{3}", line):
            if re.match(r"(?i)^(page|gao-|appendix|contents|figure|table)", line):
                continue
            if "/gid" in line:        # pypdf glyph-extraction artifacts
                continue
            if line[0].islower():     # sentence fragment, not a heading
                continue
            return line
    return "GAO Weapon Systems Assessment (section)"


def extract_common_name(page_text: str) -> str | None:
    """
    Every GAO program-profile page carries a standardized header like
    "Navy Program Type: MDAP Common Name: CVN 78". If this page has one,
    return the program's short name; otherwise None (front matter,
    overview chapters, appendices).
    """
    match = re.search(r"Common Name:\s*([^\n]{2,60})", page_text)
    return match.group(1).strip() if match else None


def is_bibliography_page(page_text: str) -> bool:
    """
    "Related GAO Products" pages are bare lists of OTHER reports' titles,
    not this document's own assessment. A model asked to answer from one
    will paraphrase a citation's title as if it were an actual finding
    (e.g. "Program Continues to Encounter Production Issues" -> "the
    program continues to face production issues") — a real hallucination
    we hit in testing. Cheapest fix: never store these pages at all, so
    they can never be retrieved and summarized in the first place.
    """
    return bool(re.search(r"^\s*Related GAO Products\s*$", page_text, re.MULTILINE))


def build_chunks(pdf_path: Path, fiscal_year: str, report: str) -> list[dict]:
    """
    Parse the PDF and chunk it SEMANTICALLY:
      * consecutive pages sharing the same "Common Name" header are grouped
        into ONE chunk -> exactly one chunk per program profile (~62), which
        is the ideal retrieval unit for this document;
      * pages without a Common Name (overview, appendices) fall back to
        fixed windows of PAGES_PER_CHUNK pages.

    Every chunk keeps the metadata that citations are built from:
        program      - program short name (or best-effort section heading)
        page_start   - first PDF page in the chunk (1-based)
        page_end     - last PDF page in the chunk
        source       - which document this came from
        fiscal_year  - "FY2024" etc. — lets retrieval scope to a year
                       (see VectorStore.query()'s `where` param)
    """
    reader = PdfReader(str(pdf_path))
    n_pages = len(reader.pages)
    print(f"Parsed PDF: {n_pages} pages")

    page_texts = [p.extract_text() or "" for p in reader.pages]
    page_names = [extract_common_name(t) for t in page_texts]

    chunks = []

    def add_chunk(start: int, end: int, program: str | None):
        """Append pages start..end (0-based, inclusive) as one chunk."""
        text = "\n".join(page_texts[p] for p in range(start, end + 1))
        if len(text.strip()) < 100:      # skip near-empty pages (covers etc.)
            return
        if is_bibliography_page(text):   # "Related GAO Products" — see is_bibliography_page()
            return
        # Year-prefixed so the same page number in two different years'
        # PDFs never collides into the same chunk id (which would silently
        # overwrite one year's chunk with another's on ingest).
        chunk_id = f"gao-{fiscal_year.lower()}-p{start + 1:04d}"
        # Render this chunk's page(s) to PNG (cached — see page_images.py).
        # This is what lets the app SHOW the real source page, and lets a
        # vision-capable tool actually READ its charts/tables.
        image_paths = render_chunk_pages(
            pdf_path, f"gao-{fiscal_year.lower()}", chunk_id, page_start=start + 1, page_end=end + 1
        )
        chunks.append(
            {
                "id": chunk_id,
                "text": text,
                "metadata": {
                    "program": program or guess_program_name(text),
                    "page_start": start + 1,          # humans count from 1
                    "page_end": end + 1,
                    "source": f"{report} Weapon Systems Annual Assessment",
                    "fiscal_year": fiscal_year,
                    # Chroma metadata must be a plain string — "|"-join, split on read.
                    "image_paths": "|".join(image_paths),
                },
            }
        )

    i = 0
    n_profiles = 0
    while i < n_pages:
        name = page_names[i]
        if name:
            # Profile: absorb every consecutive page with the same name.
            j = i
            while j + 1 < n_pages and page_names[j + 1] == name:
                j += 1
            add_chunk(i, j, name)
            n_profiles += 1
        else:
            # Non-profile pages: fixed window until the next profile starts.
            j = i
            while (
                j + 1 < n_pages
                and page_names[j + 1] is None
                and (j + 1 - i) < PAGES_PER_CHUNK
            ):
                j += 1
            add_chunk(i, j, None)
        i = j + 1

    print(f"Built {len(chunks)} chunks ({n_profiles} program profiles + "
          f"{len(chunks) - n_profiles} overview/appendix sections)")
    return chunks


def main():
    parser = argparse.ArgumentParser(description="Ingest GAO reports (all configured years) into Chroma")
    parser.add_argument(
        "--rebuild", action="store_true",
        help="wipe the existing collection and re-embed every year from scratch",
    )
    parser.add_argument(
        "--year", action="append", dest="years", choices=list(GAO_YEARS),
        help="ingest only this fiscal year (repeatable); default is all configured years",
    )
    args = parser.parse_args()
    years = args.years or list(GAO_YEARS)

    store = VectorStore()
    if args.rebuild:
        print("--rebuild: wiping existing collection")
        store.reset()

    for fiscal_year in years:
        entry = GAO_YEARS[fiscal_year]
        print(f"\n=== {fiscal_year} ({entry['report']}) ===")
        download_pdf(entry["path"], entry["url"], label=f"GAO {fiscal_year}")

        # THE CACHE CHECK, now per-year: a multi-year collection means
        # "the collection has SOME chunks" no longer means "this year is
        # done" — check for this specific year's chunks instead.
        if not args.rebuild and store.has_metadata_value("fiscal_year", fiscal_year):
            print(f"{fiscal_year} already ingested — skipping (use --rebuild to redo it).")
            continue

        chunks = build_chunks(entry["path"], fiscal_year, entry["report"])
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
