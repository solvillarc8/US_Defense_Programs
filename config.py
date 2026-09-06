"""
config.py — the ONE place where all tunable settings live.

Why this file exists:
    Every other module (ingestion, vector store, agent, UI) imports its
    settings from here. When we later swap the model tier for evaluation
    runs, change the chunk size, or move the data folder, we edit ONE line
    here and nothing else needs to change.
"""

from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# PROJECT_ROOT is the folder this file lives in. Using Path(__file__) means
# the paths keep working no matter which folder you launch Python from.
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # downloaded source PDFs live here
CHROMA_DIR = DATA_DIR / "chroma"      # the persistent vector DB lives here

# The GAO Weapon Systems Annual Assessment PDF (our only corpus tonight).
GAO_PDF_PATH = RAW_DIR / "gao-26-108457.pdf"
GAO_PDF_URL = "https://www.gao.gov/assets/gao-26-108457.pdf"

# ---------------------------------------------------------------------------
# Multi-year coverage — GAO + Weapons Book, FY2024 through FY2026
# ---------------------------------------------------------------------------
# Each entry gives the fixed report identifier (GAO's changes every year;
# the Weapons Book's doesn't) plus its download path/URL. GAO's server
# blocks plain downloads every year we've tried — pdf_fetch.py's Wayback
# Machine fallback handles that automatically at ingest time. Both sources
# keep the same per-program chunking shape year to year, so a new year is
# "add an entry here", not new ingestion logic — CRS is NOT included: its
# FY2024 report (R47582) is a structurally different document from
# FY2026's (R48860), so it isn't a drop-in extension of ingest_crs.py.
GAO_YEARS = {
    "FY2024": {"path": RAW_DIR / "gao-24-106831.pdf", "url": "https://www.gao.gov/assets/gao-24-106831.pdf", "report": "GAO-24-106831"},
    "FY2025": {"path": RAW_DIR / "gao-25-107569.pdf", "url": "https://www.gao.gov/assets/gao-25-107569.pdf", "report": "GAO-25-107569"},
    "FY2026": {"path": GAO_PDF_PATH, "url": GAO_PDF_URL, "report": "GAO-26-108457"},
}

# ---------------------------------------------------------------------------
# Model (the LLM that powers the agent)
# ---------------------------------------------------------------------------
MODEL_NAME = "gpt-4o-mini"
MAX_TOKENS = 1024                     # cap on each answer's length
TRANSCRIBE_MODEL = "whisper-1"        # speech-to-text for voice input (week 3)

# Final groundedness review. Tool-backed drafts are checked once against the
# exact retrieved text before they are returned to the UI. This catches
# plausible but unsupported synthesis that prompt rules alone did not stop.
# OFF for interactive use: it is a second full LLM call after the streamed
# answer, which roughly doubled per-turn latency. The prompt rules (3a/3b/4)
# and the tool-level coverage check remain the grounding safeguards.
GROUNDING_REVIEW_ENABLED = False
GROUNDING_REVIEW_MODEL = MODEL_NAME
GROUNDING_REVIEW_MAX_TOKENS = 1024
GROUNDING_REVIEW_SOURCE_CHAR_LIMIT = 10000

# ---------------------------------------------------------------------------
# Ingestion / chunking
# ---------------------------------------------------------------------------
# Each GAO program profile is a standardized ~2-page spread. Our chunking
# strategy is "one chunk = one program profile". PAGES_PER_CHUNK is the
# fallback window if profile detection ever fails.
PAGES_PER_CHUNK = 2

# ---------------------------------------------------------------------------
# Vector store / retrieval
# ---------------------------------------------------------------------------
COLLECTION_NAME = "gao_reports"       # Chroma collection for this source.
                                      # TODO(week 2): add "weapons_book",
                                      # "crs_reports" collections alongside.
TOP_K = 4                             # how many chunks retrieval returns

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
# We use Chroma's built-in DEFAULT embedding model (all-MiniLM-L6-v2).
# It runs locally on the laptop CPU and is completely free — so "embed once,
# never re-pay" is automatic: Chroma persists vectors to CHROMA_DIR on disk.
# TODO(later): if retrieval quality needs a boost, swap in a paid embedder
# (e.g. Voyage) inside vector_store.py — nothing outside that file changes.


# ---------------------------------------------------------------------------
# Reranker 
# ---------------------------------------------------------------------------
# The reranker re-scores a shortlist for accuracy. RERANK_ENABLED is a switch
# so we can turn it OFF later for the "rerank vs no-rerank" evaluation.
RERANK_ENABLED = True
HYBRID_SEARCH_ENABLED = True 
RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"   # FlashRank model, ~34MB, local, free
FETCH_K = 12          # how many chunks to over-fetch BEFORE reranking
                      # (12, not 20: reranking cost scales with this and
                      # the corpora are now 3x larger with multi-year data)

# Default retrieval scope when the user picks no year in the UI: the
# latest report year only. tools.set_year_scope() overrides per request.
DEFAULT_FISCAL_YEARS = ["FY2026"]
AVAILABLE_FISCAL_YEARS = ["FY2024", "FY2025", "FY2026"]
# (TOP_K above stays 4 — that's how many survive AFTER reranking)

# ---------------------------------------------------------------------------
# Weapons Book — DoD's own official view (second data source)
# ---------------------------------------------------------------------------
WEAPONS_BOOK_PDF_PATH = RAW_DIR / "fy2026_weapons.pdf"
WEAPONS_BOOK_PDF_URL = "https://comptroller.war.gov/Portals/45/Documents/defbudget/FY2026/FY2026_Weapons.pdf"
WEAPONS_BOOK_COLLECTION_NAME = "weapons_book"

# See GAO_YEARS above for why only GAO + Weapons Book span multiple years.
WEAPONS_BOOK_YEARS = {
    "FY2024": {"path": RAW_DIR / "fy2024_weapons.pdf", "url": "https://comptroller.war.gov/Portals/45/Documents/defbudget/FY2024/FY2024_Weapons.pdf"},
    "FY2025": {"path": RAW_DIR / "fy2025_weapons.pdf", "url": "https://comptroller.war.gov/Portals/45/Documents/defbudget/FY2025/FY2025_Weapons.pdf"},
    "FY2026": {"path": WEAPONS_BOOK_PDF_PATH, "url": WEAPONS_BOOK_PDF_URL},
}

# ---------------------------------------------------------------------------
# Multimodal — chart/page images (week 3)
# ---------------------------------------------------------------------------
# Every chunk's page(s) get rendered to PNG once at ingestion time (same
# "cache forever" philosophy as embeddings) so the app can (a) DISPLAY the
# real source page next to any answer, and (b) let a vision-capable model
# actually READ charts/tables that plain text extraction can't structure
# correctly (e.g. a donut chart's numbers, a schedule timeline's dates).
IMAGES_DIR = DATA_DIR / "images"
IMAGE_DPI = 130            # good text/chart legibility at a modest file size

# ---------------------------------------------------------------------------
# CRS Report — nonpartisan congressional analysis (third data source)
# ---------------------------------------------------------------------------
# "FY2026 Defense Budget: Funding for Selected Weapon Systems" (R48860).
# Unlike GAO (one profile per program) and the Weapons Book (one page per
# program), this document is a continuous narrative report organized into
# a handful of broad category sections, each covering many programs at
# once — so its chunking is "one chunk per section", not per program. See
# ingest_crs.py for the verified section page boundaries.
CRS_PDF_PATH = RAW_DIR / "crs-r48860.pdf"
CRS_PDF_URL = "https://www.congress.gov/crs_external_products/R/PDF/R48860/R48860.6.pdf"
CRS_COLLECTION_NAME = "crs_reports"

SOURCE_DOCUMENT_LINKS = [
    # One entry per GAO year — a generic "GAO Weapon Systems Annual
    # Assessment" match would have made every year's citation link open
    # the WRONG year's PDF (they all share that phrase; only the report
    # number differs), so each year needs its own exact match.
    *[
        {"match": entry["report"], "label": f"GAO Weapon Systems Annual Assessment {fy}", "url": entry["url"]}
        for fy, entry in GAO_YEARS.items()
    ],
    *[
        {"match": f"DoD {fy} Program Acquisition", "label": f"DoD {fy} Weapons Book", "url": entry["url"]}
        for fy, entry in WEAPONS_BOOK_YEARS.items()
    ],
    {"match": "CRS Report R48860", "label": "CRS FY2026 Defense Budget report", "url": CRS_PDF_URL},
    {"match": "USAspending.gov", "label": "USAspending.gov live contract awards", "url": "https://www.usaspending.gov/", "live": True},
]

# ---------------------------------------------------------------------------
# Memory — the agent's persistent watchlist/notes (week 3)
# ---------------------------------------------------------------------------
# A tiny JSON file, not a database — this only ever holds a short list of
# tracked programs and notes, so a database would be overkill. Persists
# across app restarts (unlike st.session_state, which resets on refresh).
MEMORY_PATH = DATA_DIR / "user_memory.json"

# ---------------------------------------------------------------------------
# Cross-source divergence analysis
# ---------------------------------------------------------------------------
DIVERGENCE_MODEL = MODEL_NAME
DIVERGENCE_MAX_TOKENS = 900
DIVERGENCE_SOURCE_CHAR_LIMIT = 10000

# ---------------------------------------------------------------------------
# USAspending.gov — live federal contract-award data
# ---------------------------------------------------------------------------
# The model cannot choose the URL, method, agency, result count, or an
# unbounded response size; it supplies only a short keyword and lookback.
USASPENDING_API_URL = "https://api.usaspending.gov/api/v2/search/spending_by_award/"
USASPENDING_TIMEOUT_SECONDS = 20
USASPENDING_DEFAULT_MONTHS_BACK = 12
USASPENDING_MAX_MONTHS_BACK = 60
USASPENDING_RESULT_LIMIT = 8
USASPENDING_MAX_RESPONSE_BYTES = 2_000_000
USASPENDING_AGENCY_NAME = "Department of Defense"
USASPENDING_CONTRACT_TYPE_CODES = ["A", "B", "C", "D"]
