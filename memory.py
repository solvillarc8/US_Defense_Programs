"""
memory.py — the agent's persistent watchlist/notes.

This is what makes the agent remember something ACROSS conversations, not
just across turns within one (st.session_state already does that, but it
resets the moment the browser tab refreshes). The model itself decides
when to write here, via the track_program / add_note tools in tools.py —
this file only owns the read/write mechanics.

Storage is one small JSON file (see MEMORY_PATH in config.py). No database:
the whole point is a short list of tracked programs and notes, so a file
is the simplest thing that could work.
"""

import json

from config import MEMORY_PATH

_EMPTY = {"tracked_programs": [], "notes": [], "watchlist_snapshots": {}}


def load_memory() -> dict:
    if not MEMORY_PATH.exists():
        return dict(_EMPTY)
    try:
        data = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return dict(_EMPTY)
    return {
        "tracked_programs": data.get("tracked_programs", []),
        "notes": data.get("notes", []),
        # Last-seen fingerprints per tracked program, so the watchlist can
        # tell you what's NEW since you last checked instead of re-showing
        # the same live snapshot every time. See watchlist.py.
        "watchlist_snapshots": data.get("watchlist_snapshots", {}),
    }


def save_memory(memory: dict) -> None:
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    MEMORY_PATH.write_text(json.dumps(memory, indent=2), encoding="utf-8")


def track_program(program: str) -> dict:
    """Add a program to the watchlist (no-op if already tracked)."""
    memory = load_memory()
    if program not in memory["tracked_programs"]:
        memory["tracked_programs"].append(program)
        save_memory(memory)
    return memory


def untrack_program(program: str) -> dict:
    """Remove a program from the watchlist (no-op if not tracked)."""
    memory = load_memory()
    if program in memory["tracked_programs"]:
        memory["tracked_programs"].remove(program)
        save_memory(memory)
    return memory


def add_note(note: str) -> dict:
    """Save a short standing note (e.g. a preference to remember)."""
    memory = load_memory()
    memory["notes"].append(note)
    save_memory(memory)
    return memory


def memory_summary() -> str:
    """
    One short paragraph for the system prompt, so the agent has ambient
    awareness of what's being tracked WITHOUT having to call a tool first.
    Empty string if there's nothing to report (keeps the prompt clean).
    """
    memory = load_memory()
    if not memory["tracked_programs"] and not memory["notes"]:
        return ""
    parts = []
    if memory["tracked_programs"]:
        parts.append("Tracked programs: " + ", ".join(memory["tracked_programs"]))
    if memory["notes"]:
        parts.append("Notes: " + "; ".join(memory["notes"]))
    return "\n".join(parts)
