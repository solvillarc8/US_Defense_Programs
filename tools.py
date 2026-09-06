"""
tools.py — the agent's TOOL REGISTRY.

This is the pattern that makes the whole project extensible:
    Each registry entry pairs
      - "definition": what the MODEL sees (name, description, input schema).
        OpenAI reads the description to decide WHEN to use the tool.
      - "function":   the Python we RUN when Claude asks for it.

    Adding a capability next week (search_usaspending, search_weapons_book,
    read_chart...) = write one function + append one entry HERE.
    agent.py and app.py never change, because they only iterate this list.
"""

import base64
import os
import re
import threading

import openai
import requests

from config import (
    MODEL_NAME, TOP_K, WEAPONS_BOOK_COLLECTION_NAME, CRS_COLLECTION_NAME,
    DEFAULT_FISCAL_YEARS, AVAILABLE_FISCAL_YEARS,
)
from divergence import analyze_divergence
from vector_store import VectorStore
from langsmith import traceable
import memory as memory_store
from usaspending import search_contract_awards

# ---------------------------------------------------------------------------
# Year scope — enforced HERE, at the tool level, not by asking the model.
# ---------------------------------------------------------------------------
# The UI sets this before each question. Every GAO / Weapons Book search is
# hard-filtered to these years regardless of what the model passes as
# fiscal_year (a model-supplied year outside the scope is ignored). With no
# UI selection the scope is DEFAULT_FISCAL_YEARS (the latest report year).
_year_scope: list[str] = list(DEFAULT_FISCAL_YEARS)


def set_year_scope(years: list[str] | None) -> None:
    """years=[] or None -> default (latest year). Unknown years are dropped."""
    valid = [y for y in (years or []) if y in AVAILABLE_FISCAL_YEARS]
    global _year_scope
    _year_scope = valid or list(DEFAULT_FISCAL_YEARS)


def get_year_scope() -> list[str]:
    return list(_year_scope)


def _year_where(requested: str | None) -> dict:
    """The Chroma `where` clause for a search: the model's requested year if
    it lies inside the active scope, else the whole scope."""
    if requested in _year_scope:
        return {"fiscal_year": {"$in": [requested]}}
    return {"fiscal_year": {"$in": list(_year_scope)}}


def warm_up() -> None:
    """Pay every cold-start cost once (embedding model, BM25 index, reranker
    for each corpus) so the first real question isn't the slow one. app.py
    calls this once per server process."""
    for store in (_get_store(), _get_weapons_store(), _get_crs_store()):
        store.query("warm up", top_k=1)

# All three VectorStore singletons below point at the same on-disk
# CHROMA_DIR (different collections within it). agent.py now runs a
# round's tool calls concurrently, so two DIFFERENT stores can each hit
# their first-ever creation at the same instant — a real crash was hit
# during testing where chromadb's tenant/client init raced across two
# PersistentClient instances sharing that directory. ONE shared lock
# serializes creation across all three (per-store locks wouldn't have
# helped — the conflict is at the shared storage level, not per-collection).
# Once a store exists, concurrent .query() calls proceed without contention.
_chroma_init_lock = threading.Lock()

# Matches program-designator-shaped tokens in a query, e.g. "F-35", "CVN 78",
# "XM30", "B-21". Used to catch cases where the model would otherwise answer
# about a named program using results that never actually mention it (see
# _coverage_note below).
_DESIGNATOR_RE = re.compile(r"\b[A-Z]{1,6}[-\s]?\d{1,4}[A-Za-z]{0,3}\b")


def _query_designators(query: str) -> list[str]:
    matches = _DESIGNATOR_RE.findall(query)
    # Fiscal-year labels match the broad designator shape (letters + digits)
    # but are not weapon programs. Treating FY2026 as a program caused valid
    # CRS budget queries to receive a false "source does not cover it" note.
    return [m for m in matches if not re.fullmatch(r"FY[-\s]?\d{4}", m, re.IGNORECASE)]


def _designator_found_by_program(designator: str, program_names: list[str]) -> bool:
    """
    Checks against each result's PROGRAM NAME (the metadata field that
    identifies what the chunk is actually about), not its raw text — a
    chunk can mention "F-35" once in passing (e.g. a related program that's
    compatible with it) without being a chunk ABOUT the F-35. Matching on
    the program name is what actually distinguishes "this source covers
    that program" from "that program's name appears somewhere on the page."

    Only valid for sources chunked ONE-PROGRAM-PER-CHUNK (GAO, Weapons
    Book) — see _designator_found_by_text for section-chunked sources.
    """
    norm = re.sub(r"[-\s]", "", designator).lower()
    return any(norm in re.sub(r"[-\s]", "", p).lower() for p in program_names)


def _designator_found_by_text(designator: str, texts: list[str]) -> bool:
    """
    For sources chunked one-SECTION-per-chunk covering many programs at
    once (CRS), "program name" isn't a program at all — it's a category
    like "Appendix" or "Aircraft and Related Weapon Systems", so it can
    NEVER match a program designator and the by-program check always
    false-flags real coverage as missing. Text presence is the correct
    check here instead: unlike GAO's bibliography pages (many report
    TITLES listed in passing, none actually assessed), a CRS section that
    mentions a program by name is actually discussing it in prose.
    """
    norm = re.sub(r"[-\s]", "", designator).lower()
    return any(norm in re.sub(r"[-\s]", "", t).lower() for t in texts)


def _coverage_note(query: str, hits: list[dict], check_by: str = "program") -> str:
    """
    We hit a real hallucination bug in testing: asked about the F-35, the
    model got back results for OTHER programs (nothing wrong with that —
    embedding search returns the closest match even when nothing is truly
    relevant) and answered as if those results covered the F-35, inventing
    plausible-sounding claims. A system-prompt instruction alone didn't
    reliably stop this. So instead we check mechanically whether a result
    actually covers the designator the user asked about. If not, we say so
    directly in the tool output data itself, which the model treats as
    ground truth far more reliably than a competing instruction.

    check_by="program" for one-program-per-chunk sources (GAO, Weapons
    Book); check_by="text" for one-section-per-chunk sources (CRS), where
    program metadata is a category name, not a program.
    """
    designators = _query_designators(query)
    if not designators:
        return ""
    if check_by == "text":
        texts = [h["text"] for h in hits]
        missing = [d for d in designators if not _designator_found_by_text(d, texts)]
    else:
        program_names = [h["metadata"].get("program", "") for h in hits]
        missing = [d for d in designators if not _designator_found_by_program(d, program_names)]
    if not missing:
        return ""
    names = ", ".join(missing)
    return (
        f"\n\n[RETRIEVAL NOTE: none of the results above is a program chunk "
        f"for {names}. The designator may appear incidentally in another "
        f"program's text, but this source most likely does not substantively "
        f"cover {names} — do not describe what it 'typically' says or "
        f"'likely' covers about it. If the question specifically asks about "
        f"{names}, say plainly you can't find it in this source.]"
    )


def _citation_safe_text(text: str) -> str:
    """Remove PDF-internal page labels that conflict with citation metadata.

    Retrieval metadata uses physical PDF pages, which are also the pages shown
    in the UI.  Extracted text contains printed report page labels (often 12
    pages lower in GAO), and the model was observed citing those instead of the
    authoritative result header.  Preserve the content while removing only
    standalone labels such as "Page 85" from model-visible text.
    """
    return re.sub(
        r"^\s*Page\s+(?:[ivxlcdm]+|\d+)\s*$", "", text,
        flags=re.MULTILINE | re.IGNORECASE,
    )

# The vector store is created lazily (first use) and reused afterwards,
# so importing this module stays instant. agent.py now runs a round's tool
# calls concurrently (see agent.py's ThreadPoolExecutor use), which means
# two DIFFERENT tools can hit their first-ever call at the same instant —
# without a lock, two threads creating a chromadb.PersistentClient against
# the same on-disk store at once raced and corrupted client init (a real
# crash hit during testing). The lock only guards CREATION; once a store
# exists, concurrent .query() calls proceed without contention.
_store: VectorStore | None = None


def _get_store() -> VectorStore:
    global _store
    if _store is None:
        with _chroma_init_lock:
            if _store is None:
                _store = VectorStore()
    return _store


def _image_paths_from_metadata(meta: dict) -> list[str]:
    """image_paths is stored as a '|'-joined string (Chroma metadata can't
    hold a list) — split it back out. Empty/missing -> no images."""
    raw = meta.get("image_paths", "")
    return raw.split("|") if raw else []

_crs_store: VectorStore | None = None


def _get_crs_store() -> VectorStore:
    global _crs_store
    if _crs_store is None:
        with _chroma_init_lock:
            if _crs_store is None:
                _crs_store = VectorStore(collection_name=CRS_COLLECTION_NAME)
    return _crs_store


@traceable(name="search_crs_reports")
def search_crs_reports(query: str) -> dict:
    """
    Semantic search over the CRS report "FY2026 Defense Budget: Funding for
    Selected Weapon Systems" — a nonpartisan CONGRESSIONAL analysis, distinct
    from both GAO's independent watchdog view and DoD's own official view.
    Each result covers a whole category section (e.g. "Aircraft and Related
    Weapon Systems"), not a single program.
    """
    # CRS exists for FY2026 only. If the active year scope excludes FY2026,
    # this source is simply out of scope for the request.
    if "FY2026" not in _year_scope:
        return {
            "text": (f"CRS report not available for {', '.join(_year_scope)} — "
                     "the CRS source covers FY2026 only and is excluded by the "
                     "active year filter."),
            "citations": [],
        }
    hits = _get_crs_store().query(query, top_k=TOP_K)
    if not hits:
        return {"text": "No results found in the CRS report.", "citations": []}
    parts, citations = [], []
    for i, h in enumerate(hits, 1):
        m = h["metadata"]
        header = (
            f"[Result {i} | section: {m['program']} | "
            f"pages {m['page_start']}-{m['page_end']} | {m['source']}]"
        )
        parts.append(f"{header}\n{_citation_safe_text(h['text'])}")
        citations.append({
            "program": m["program"], "page_start": m["page_start"],
            "page_end": m["page_end"], "source": m["source"],
            "image_paths": _image_paths_from_metadata(m),
        })
    text = "\n\n---\n\n".join(parts) + _coverage_note(query, hits, check_by="text")
    return {"text": text, "citations": citations}

# ---------------------------------------------------------------------------
# Tool 1: search_gao_reports  (tonight's only tool)
# ---------------------------------------------------------------------------
@traceable(name="search_gao_reports")
def search_gao_reports(query: str, fiscal_year: str | None = None) -> dict:
    """
    Semantic search over the ingested GAO report(s). fiscal_year (e.g.
    "FY2024") scopes retrieval to that year's document when the collection
    spans multiple years; omit it to search across all ingested years.
    Returns {"text": <for the model>, "citations": <exact page data for the UI>}.
    """
    where = _year_where(fiscal_year)
    hits = _get_store().query(query, top_k=TOP_K, where=where)
    if not hits:
        return {"text": f"No results found in the GAO corpus for {', '.join(_year_scope)}.", "citations": []}
    parts, citations = [], []
    for i, h in enumerate(hits, 1):
        m = h["metadata"]
        header = (
            f"[Result {i} | section: {m['program']} | "
            f"pages {m['page_start']}-{m['page_end']} | {m['source']}]"
        )
        parts.append(f"{header}\n{_citation_safe_text(h['text'])}")
        citations.append({
            "program": m["program"], "page_start": m["page_start"],
            "page_end": m["page_end"], "source": m["source"],
            "fiscal_year": m.get("fiscal_year"),
            "image_paths": _image_paths_from_metadata(m),
        })
    text = "\n\n---\n\n".join(parts) + _coverage_note(query, hits)
    return {"text": text, "citations": citations}

# ---------------------------------------------------------------------------
# Tool 2: search_weapons_book  (second source — DoD's own official view)
# ---------------------------------------------------------------------------
_weapons_store: VectorStore | None = None


def _get_weapons_store() -> VectorStore:
    global _weapons_store
    if _weapons_store is None:
        with _chroma_init_lock:
            if _weapons_store is None:
                _weapons_store = VectorStore(collection_name=WEAPONS_BOOK_COLLECTION_NAME)
    return _weapons_store


@traceable(name="search_weapons_book")
def search_weapons_book(query: str, fiscal_year: str | None = None) -> dict:
    """
    Semantic search over DoD's own "Program Acquisition Costs by Weapon
    System" — the OFFICIAL budget view, as opposed to GAO's independent
    watchdog view from search_gao_reports. fiscal_year (e.g. "FY2024")
    scopes retrieval to that year's document; omit it to search across all
    ingested years.
    """
    where = _year_where(fiscal_year)
    hits = _get_weapons_store().query(query, top_k=TOP_K, where=where)
    if not hits:
        return {"text": f"No results found in the Weapons Book for {', '.join(_year_scope)}.", "citations": []}
    parts, citations = [], []
    for i, h in enumerate(hits, 1):
        m = h["metadata"]
        header = (
            f"[Result {i} | program: {m['program']} | category: {m['category']} | "
            f"section {m['section']} | pages {m['page_start']}-{m['page_end']} | {m['source']}]"
        )
        parts.append(f"{header}\n{_citation_safe_text(h['text'])}")
        citations.append({
            "program": m["program"], "page_start": m["page_start"],
            "page_end": m["page_end"], "source": m["source"],
            "fiscal_year": m.get("fiscal_year"),
            "image_paths": _image_paths_from_metadata(m),
        })
    text = "\n\n---\n\n".join(parts) + _coverage_note(query, hits)
    return {"text": text, "citations": citations}


@traceable(name="check_source_divergence")
def check_source_divergence(program: str) -> dict:
    """Retrieve all three static sources and classify their differences."""
    gao = search_gao_reports(program)
    weapons = search_weapons_book(program)
    crs = search_crs_reports(program)
    try:
        analysis = analyze_divergence(program, {
            "GAO": gao["text"], "Weapons Book": weapons["text"],
            "CRS": crs["text"],
        })
    except openai.OpenAIError as exc:
        analysis = f"Divergence analysis unavailable: {type(exc).__name__}."

    citations, seen = [], set()
    for result in (gao, weapons, crs):
        for citation in result["citations"]:
            key = (citation["source"], citation["program"],
                   citation["page_start"], citation["page_end"])
            if key not in seen:
                seen.add(key)
                citations.append(citation)
    return {
        "text": f"DEDICATED CROSS-SOURCE DIVERGENCE CHECK: {program}\n\n{analysis}",
        "citations": citations,
    }


@traceable(name="search_usaspending_awards")
def search_usaspending_awards(
    query: str,
    months_back: int = 12,
    start_date: str | None = None,
    end_date: str | None = None,
    exclude_keywords: list[str] | None = None,
    exclude_recipients: list[str] | None = None,
) -> dict:
    """
    Read-only live search of bounded DoD prime contract awards.

    start_date/end_date give an exact window (e.g. the user asks for "2023
    only" or "between March and June 2024") and take priority over
    months_back when both are given. exclude_keywords/exclude_recipients
    let the user narrow OUT results — e.g. "F-35 awards but not Lockheed"
    excludes that recipient rather than searching for it.
    """
    try:
        return search_contract_awards(
            query, months_back, start_date=start_date, end_date=end_date,
            exclude_keywords=exclude_keywords, exclude_recipients=exclude_recipients,
        )
    except (requests.RequestException, ValueError, TypeError) as exc:
        return {
            "text": f"USAspending live data unavailable: {type(exc).__name__}: {exc}",
            "citations": [],
        }


# ---------------------------------------------------------------------------
# Tool 3: read_program_chart  (multimodal — actually LOOKS at the page)
# ---------------------------------------------------------------------------
@traceable(name="read_program_chart")
def read_program_chart(query: str) -> dict:
    """
    Finds the most relevant page (via the same retrieval pipeline as the
    search tools) and sends its IMAGE to a vision-capable model, asking it
    to read the chart/table structure directly. Plain text extraction can
    grab the raw numbers off a page like this but loses which number
    belongs to which chart element (e.g. a donut chart's two labeled
    rings, a schedule timeline's dated milestones) — vision preserves that
    structure because it's looking at the actual layout.
    """
    hits = _get_store().query(query, top_k=1)
    if not hits or not _image_paths_from_metadata(hits[0]["metadata"]):
        return {"text": "No chart image found for that query.", "citations": []}

    m = hits[0]["metadata"]
    image_paths = _image_paths_from_metadata(m)
    image_path = image_paths[0]   # the page most likely to hold the chart

    if not os.environ.get("OPENAI_API_KEY"):
        return {"text": "Vision reading unavailable: no OpenAI API key configured.",
                "citations": []}

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=600,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    "This is a page from a US defense budget document. Read "
                    "any charts, donut/ring graphics, tables, or schedule "
                    "timelines on this page precisely. For each chart, state "
                    "its title, every labeled number with what it measures, "
                    "and (for a timeline) each milestone with its date. Do "
                    "not summarize loosely — report the actual figures you see."
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ],
        }],
    )
    description = response.choices[0].message.content or ""

    header = (
        f"[Chart reading | program: {m['program']} | "
        f"pages {m['page_start']}-{m['page_end']} | {m['source']}]"
    )
    citation = {
        "program": m["program"], "page_start": m["page_start"],
        "page_end": m["page_end"], "source": m["source"],
        "image_paths": image_paths,
    }
    return {"text": f"{header}\n{description}", "citations": [citation]}


# ---------------------------------------------------------------------------
# Tool 5/6: track_program / recall_watchlist  (persistent memory)
# ---------------------------------------------------------------------------
@traceable(name="track_program")
def track_program(program: str) -> dict:
    """
    Adds a program to the persistent watchlist (data/user_memory.json).
    Unlike chat history, this survives a browser refresh or app restart —
    it's real memory, not just this conversation's scratch state.
    """
    mem = memory_store.track_program(program)
    return {
        "text": f"Added '{program}' to your watchlist. "
                f"Currently tracking: {', '.join(mem['tracked_programs']) or '(none)'}.",
        "citations": [],
    }


@traceable(name="recall_watchlist")
def recall_watchlist() -> dict:
    """Returns the current persistent watchlist and any saved notes."""
    mem = memory_store.load_memory()
    if not mem["tracked_programs"] and not mem["notes"]:
        return {"text": "Nothing is currently being tracked.", "citations": []}
    lines = []
    if mem["tracked_programs"]:
        lines.append("Tracked programs: " + ", ".join(mem["tracked_programs"]))
    if mem["notes"]:
        lines.append("Notes: " + "; ".join(mem["notes"]))
    return {"text": "\n".join(lines), "citations": []}


# ---------------------------------------------------------------------------
# THE REGISTRY — the agent iterates this list; it names no tool directly.
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "definition": {
            "name": "check_source_divergence",
            "description": (
                "Run the same program query against GAO, the DoD Weapons Book, "
                "and CRS, then explicitly classify agreements, disagreements, "
                "different emphasis, and source gaps. Use for source comparisons "
                "or divergence; do not make that judgment yourself."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"program": {
                    "type": "string", "description": "Program name or designator"
                }},
                "required": ["program"],
            },
        },
        "function": check_source_divergence,
    },
    {
        "definition": {
            "name": "search_usaspending_awards",
            "description": (
                "Search current USAspending.gov prime contract awards made by "
                "the Department of Defense. Use for recent/live awards, "
                "recipients, obligations, or contract activity. Supports an "
                "exact date window and excluding specific keywords or "
                "recipients when the user asks to narrow OUT results."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Program or contractor keyword"},
                    "months_back": {"type": "integer", "minimum": 1, "maximum": 60},
                    "start_date": {"type": "string", "description": "YYYY-MM-DD; overrides months_back if given with end_date"},
                    "end_date": {"type": "string", "description": "YYYY-MM-DD; overrides months_back if given with start_date"},
                    "exclude_keywords": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Terms that must NOT appear in an award's description or recipient name",
                    },
                    "exclude_recipients": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Recipient/contractor names to exclude entirely",
                    },
                },
                "required": ["query"],
            },
        },
        "function": search_usaspending_awards,
    },
    {
        "definition": {
            "name": "search_gao_reports",
            "description": (
                "Search the GAO Weapon Systems Annual Assessment, an "
                "independent watchdog review of major US DoD weapon "
                "acquisition programs. Covers FY2024, FY2025, and FY2026 — "
                "each a separate year's report. Call this for ANY question "
                "about a weapon program's cost, schedule, performance, "
                "technology maturity, or risks. Returns text excerpts, each "
                "tagged with its section name and page numbers for citation. "
                "NEVER use this tool directly for a source-comparison or "
                "divergence request; use check_source_divergence instead."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A search phrase describing what to look for, "
                            "e.g. 'F-35 cost overruns and schedule delays'"
                        ),
                    },
                    "fiscal_year": {
                        "type": "string", "enum": ["FY2024", "FY2025", "FY2026"],
                        "description": "Scope to one year's report; omit to search all ingested years.",
                    },
                },
                "required": ["query"],
            },
        },
        "function": search_gao_reports,
    },
    {
        "definition": {
            "name": "search_weapons_book",
            "description": (
                "Search DoD's own 'Program Acquisition Costs by Weapon "
                "System' (the official budget request document, informally "
                "'the Weapons Book'). Covers FY2024, FY2025, and FY2026 — "
                "each a separate year's edition. Call this for the "
                "Department's OWN stated view of a program's funding, "
                "category, or budget request — as opposed to GAO's "
                "independent watchdog assessment. Useful for comparing "
                "official program facts. NEVER use this tool directly for a "
                "source-comparison or divergence request; use "
                "check_source_divergence instead."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A search phrase, e.g. 'CVN 78 aircraft carrier funding'",
                    },
                    "fiscal_year": {
                        "type": "string", "enum": ["FY2024", "FY2025", "FY2026"],
                        "description": "Scope to one year's edition; omit to search all ingested years.",
                    },
                },
                "required": ["query"],
            },
        },
        "function": search_weapons_book,
    },
    {
        "definition": {
            "name": "read_program_chart",
            "description": (
                "Read a program's chart or table by LOOKING at the actual "
                "page image (multimodal), not just its extracted text. Call "
                "this when the user specifically asks about a program's cost "
                "breakdown chart, quantity chart, schedule/milestone "
                "timeline, or wants precise figures from a graphic — cases "
                "where the visual structure (which number belongs to which "
                "chart element) matters and plain text search might not "
                "capture it correctly. For general questions, prefer "
                "search_gao_reports / search_weapons_book instead."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The program whose chart to read, e.g. 'CVN 78 cost chart'",
                    }
                },
                "required": ["query"],
            },
        },
        "function": read_program_chart,
    },
    {
        "definition": {
            "name": "search_crs_reports",
            "description": (
                "Search the CRS (Congressional Research Service) report "
                "'FY2026 Defense Budget: Funding for Selected Weapon "
                "Systems' — a NONPARTISAN analysis prepared for Congress, "
                "distinct from GAO's independent watchdog view and DoD's "
                "own official Weapons Book. Covers funding, authorization, "
                "and appropriation status by category (aircraft, missile "
                "defense, shipbuilding, etc.), including congressional "
                "action (House/Senate bills, enacted law) not covered by "
                "the other two sources. NEVER use this tool directly for a "
                "source-comparison or divergence request; use "
                "check_source_divergence instead."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A search phrase, e.g. 'F-35 funding and congressional action'",
                    }
                },
                "required": ["query"],
            },
        },
        "function": search_crs_reports,
    },
    {
        "definition": {
            "name": "track_program",
            "description": (
                "Add a weapon program to the user's persistent watchlist, "
                "saved across sessions (not just this conversation). Call "
                "this when the user says they want to track, follow, watch, "
                "or keep an eye on a specific program."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "program": {
                        "type": "string",
                        "description": "The program's name/designator, e.g. 'CVN 78' or 'F-35'",
                    }
                },
                "required": ["program"],
            },
        },
        "function": track_program,
    },
    {
        "definition": {
            "name": "recall_watchlist",
            "description": (
                "List the programs and notes currently saved in the user's "
                "persistent watchlist. Call this when the user asks what "
                "they're tracking/following, or references 'my watchlist'."
            ),
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
        "function": recall_watchlist,
    },
]


# ---------------------------------------------------------------------------
# Helpers the agent uses (these never need to change when tools are added)
# ---------------------------------------------------------------------------
def get_tool_definitions() -> list[dict]:
    """The list of definitions sent to the model with every request,
    wrapped in OpenAI's function-tool format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["definition"]["name"],
                "description": t["definition"]["description"],
                "parameters": t["definition"]["input_schema"],
            },
        }
        for t in TOOLS
    ]
def run_tool(name: str, tool_input: dict) -> dict:
    """Dispatch a tool call. Always returns {"text": ..., "citations": ...}."""
    for t in TOOLS:
        if t["definition"]["name"] == name:
            result = t["function"](**tool_input)
            if isinstance(result, dict):
                return {"text": result.get("text", ""), "citations": result.get("citations", [])}
            return {"text": result, "citations": []}
    return {"text": f"Error: unknown tool '{name}'", "citations": []}
