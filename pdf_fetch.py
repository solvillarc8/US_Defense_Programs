"""
pdf_fetch.py — shared "download once, cache forever" helper for source PDFs.

Both gao.gov and the DoD comptroller site block plain requests some of the
time (bot detection returns a 403 with an HTML error page instead of the
PDF). Discovered and worked around repeatedly this project: the fix is to
try the direct URL first, and if that fails, fall back to a Wayback Machine
snapshot of the same URL — which is never bot-blocked.
"""

from pathlib import Path

import requests


def download_pdf(path: Path, url: str, label: str) -> None:
    """Download url to path if it doesn't already exist (the cache check).
    Tries the direct URL first, then a Wayback Machine snapshot."""
    if path.exists():
        print(f"{label}: already downloaded ({path})")
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        resp = requests.get(url, timeout=120, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200 and resp.headers.get("Content-Type", "").startswith(
            ("application/pdf", "application/octet-stream")
        ) or (resp.status_code == 200 and resp.content[:4] == b"%PDF"):
            path.write_bytes(resp.content)
            print(f"{label}: downloaded directly ({len(resp.content):,} bytes)")
            return
        print(f"{label}: direct download blocked (HTTP {resp.status_code}) — trying Wayback Machine")
    except requests.RequestException as exc:
        print(f"{label}: direct download failed ({exc}) — trying Wayback Machine")

    wayback_url = f"https://web.archive.org/web/2id_/{url}"
    resp = requests.get(wayback_url, timeout=180, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    if resp.content[:4] != b"%PDF":
        raise RuntimeError(f"{label}: Wayback Machine did not return a PDF for {url}")
    path.write_bytes(resp.content)
    print(f"{label}: downloaded via Wayback Machine ({len(resp.content):,} bytes)")
