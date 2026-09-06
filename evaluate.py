"""
evaluate.py — the evaluation harness.

Runs every question in eval_questions.py through the LIVE agent, grades
each answer, and prints + saves a scored summary.

This is what turns "I think the reranker helps" into a real number:

    python evaluate.py --tag baseline
    # ... now edit config.py: RERANK_ENABLED = False ...
    python evaluate.py --tag no_rerank
    python compare_eval.py baseline no_rerank

Cost note: ~32 questions x 1-2 model calls each, on gpt-4o-mini, is a few
cents per full run — check LangSmith afterward if you want the exact figure.
"""

import argparse
import json
import time

from agent import ask
from config import PROJECT_ROOT
from eval_questions import EVAL_QUESTIONS

RESULTS_DIR = PROJECT_ROOT / "eval_results"
REFUSAL_MARKERS = ("can't find", "cannot find", "not in the available")


def run_one(case: dict) -> dict:
    """Run one test case through the live agent and grade it."""
    out = {}
    answer = "".join(ask(case["question"], out=out))  # drain the streaming generator

    result = {
        "id": case["id"],
        "question": case["question"],
        "answer": answer,
        "n_tool_calls": len(out.get("tool_calls", [])),
        "cited_correctly": None,
    }
    tool_names = [call["name"] for call in out.get("tool_calls", [])]
    expected_tool = case.get("expect_tool")
    result["called_expected_tool"] = (
        expected_tool in tool_names if expected_tool else None
    )

    if case.get("expect_refusal"):
        refused = any(marker in answer.lower() for marker in REFUSAL_MARKERS)
        result["correct"] = refused and result["called_expected_tool"] is not False
        return result

    if case.get("expect_both_sources"):
        sources_seen = {c["source"] for c in out.get("citations", [])}
        has_gao = any("GAO" in s for s in sources_seen)
        has_wb = any("Weapon System" in s for s in sources_seen)
        result["correct"] = (
            has_gao and has_wb and result["called_expected_tool"] is not False
        )
        return result

    # Default: every expected keyword must appear (case-insensitive).
    answer_lower = answer.lower()
    missing = [kw for kw in case["expect_keywords"] if kw.lower() not in answer_lower]
    result["correct"] = (
        len(missing) == 0 and result["called_expected_tool"] is not False
    )
    result["missing_keywords"] = missing

    if "expect_page" in case:
        pages_cited = {
            p for c in out.get("citations", [])
            for p in range(c["page_start"], c["page_end"] + 1)
        }
        result["cited_correctly"] = case["expect_page"] in pages_cited
        result["correct"] = result["correct"] and result["cited_correctly"]

    return result


def main():
    parser = argparse.ArgumentParser(description="Run the evaluation harness")
    parser.add_argument("--tag", required=True, help="label for this run, e.g. 'baseline' or 'no_rerank'")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N questions (quick testing)")
    args = parser.parse_args()

    cases = EVAL_QUESTIONS[: args.limit] if args.limit else EVAL_QUESTIONS
    print(f"Running {len(cases)} questions (tag: {args.tag})...\n")

    results = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['id']}...", end=" ", flush=True)
        t0 = time.time()
        r = run_one(case)
        r["seconds"] = round(time.time() - t0, 1)
        results.append(r)
        print(f"{'PASS' if r['correct'] else 'FAIL'} ({r['seconds']}s)")

    n_correct = sum(r["correct"] for r in results)
    citation_checked = [r for r in results if r["cited_correctly"] is not None]
    n_cited = sum(r["cited_correctly"] for r in citation_checked)

    print(f"\n=== Summary (tag: {args.tag}) ===")
    print(f"Correct answers:   {n_correct}/{len(results)}  ({100*n_correct/len(results):.0f}%)")
    if citation_checked:
        print(f"Correct citations: {n_cited}/{len(citation_checked)}  ({100*n_cited/len(citation_checked):.0f}%)")

    failures = [r for r in results if not r["correct"]]
    if failures:
        print(f"\nFailed ({len(failures)}):")
        for r in failures:
            print(f"  - {r['id']}: missing {r.get('missing_keywords')}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{args.tag}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
