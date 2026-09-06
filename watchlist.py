"""
watchlist.py — live status checks for programs on the persistent watchlist.

Surfaces, per tracked program:
  - genuinely NEW USAspending contract awards (an Award ID not seen on the
    previous check — not just "every award in the window again")
  - whether the current GAO / Weapons Book excerpt for that program has
    changed since the last check (a real text fingerprint, not a guessed
    dollar delta — this app's own grounding rules exist specifically
    because ambiguous multi-column budget tables produce wrong numbers
    when a model tries to compute "the change" itself)

Deliberately NOT run on every Streamlit rerun — these are live network and
LLM-adjacent calls. app.py triggers a refresh only when the user asks for
one, and the result (plus the updated snapshot) is persisted via memory.py
so the next visit can tell what's new since then.
"""

import re

from tools import search_gao_reports, search_weapons_book
from usaspending import search_contract_awards
import memory as memory_store

_AWARD_ID_RE = re.compile(r"^Award ID:\s*(.+)$", re.MULTILINE)

# GAO/Weapons Book now span multiple years (FY2024-FY2026). An unscoped
# search ranks across all of them, so which year's chunk comes back "first"
# for a given program can shift between calls even when nothing actually
# changed — that made gao_report_changed/weapons_report_changed fire
# spuriously. Pinning both checks to the current year makes the fingerprint
# comparison meaningful again: it's now genuinely "did THIS year's report
# text change", not "did search happen to rank a different year higher".
_CURRENT_FISCAL_YEAR = "FY2026"


def _fingerprint(text: str) -> str:
    return str(hash(text))


def _award_ids(awards_text: str) -> set[str]:
    return {m.strip() for m in _AWARD_ID_RE.findall(awards_text) if m.strip() and m.strip() != "not reported"}


def refresh_program(program: str) -> dict:
    """Fetch a fresh live status for one program and update its snapshot."""
    mem = memory_store.load_memory()
    prior = mem["watchlist_snapshots"].get(program, {})

    try:
        awards = search_contract_awards(query=program)
        awards_error = None
    except Exception as exc:  # network/API errors shouldn't crash the page
        awards, awards_error = {"text": ""}, f"{type(exc).__name__}: {exc}"

    gao = search_gao_reports(program, fiscal_year=_CURRENT_FISCAL_YEAR)
    weapons = search_weapons_book(program, fiscal_year=_CURRENT_FISCAL_YEAR)

    current_award_ids = _award_ids(awards.get("text", ""))
    prior_award_ids = set(prior.get("award_ids", []))
    new_award_ids = current_award_ids - prior_award_ids if prior else set()

    gao_fp = _fingerprint(gao.get("text", ""))
    weapons_fp = _fingerprint(weapons.get("text", ""))

    status = {
        "program": program,
        "awards_error": awards_error,
        "new_award_count": len(new_award_ids),
        "new_award_ids": sorted(new_award_ids),
        "total_award_count": len(current_award_ids),
        "gao_report_changed": bool(prior) and prior.get("gao_fp") not in (None, gao_fp),
        "weapons_report_changed": bool(prior) and prior.get("weapons_fp") not in (None, weapons_fp),
        "gao_citation": (gao.get("citations") or [None])[0],
        "weapons_citation": (weapons.get("citations") or [None])[0],
        "is_first_check": not bool(prior),
    }

    mem["watchlist_snapshots"][program] = {
        "award_ids": sorted(current_award_ids),
        "gao_fp": gao_fp,
        "weapons_fp": weapons_fp,
    }
    memory_store.save_memory(mem)
    return status


def refresh_watchlist() -> list[dict]:
    """Refresh every tracked program. Returns one status dict each."""
    mem = memory_store.load_memory()
    return [refresh_program(p) for p in mem["tracked_programs"]]
