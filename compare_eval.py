"""
compare_eval.py — compare two evaluation runs.

Run it:  python compare_eval.py baseline no_rerank
"""

import json
import sys

from config import PROJECT_ROOT

RESULTS_DIR = PROJECT_ROOT / "eval_results"


def load(tag: str) -> dict:
    return {r["id"]: r for r in json.loads((RESULTS_DIR / f"{tag}.json").read_text())}


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_eval.py <tag_a> <tag_b>")
        return
    tag_a, tag_b = sys.argv[1], sys.argv[2]
    a, b = load(tag_a), load(tag_b)

    acc_a = sum(r["correct"] for r in a.values()) / len(a) * 100
    acc_b = sum(r["correct"] for r in b.values()) / len(b) * 100
    print(f"{tag_a}: {acc_a:.0f}% correct  ({len(a)} questions)")
    print(f"{tag_b}: {acc_b:.0f}% correct  ({len(b)} questions)")
    print(f"Difference: {acc_b - acc_a:+.0f} percentage points\n")

    print("Questions where the two runs DISAGREED:")
    diffs = [qid for qid in a if qid in b and a[qid]["correct"] != b[qid]["correct"]]
    if not diffs:
        print("  (none)")
    for qid in diffs:
        print(f"  - {qid}: {tag_a}={a[qid]['correct']}  {tag_b}={b[qid]['correct']}")


if __name__ == "__main__":
    main()