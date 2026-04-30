"""
End-to-end evaluator for structural questions (router path).

Calls answer() for each structural question and scores the generated answer
against expected_answer by checking how many key entities from the expected
answer appear in the generated answer.

Usage:
    python evaluate/eval_router.py
    python evaluate/eval_router.py --questions path/to/other.yaml
    python evaluate/eval_router.py --threshold 0.5   # pass if >=50% of expected terms found
    python evaluate/eval_router.py --verbose          # show full answers on failure
"""

import argparse
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.rag import answer

DEFAULT_QUESTIONS = Path(__file__).parent / "test_questions.yaml"
DEFAULT_THRESHOLD = 0.5


def _key_terms(text: str) -> list[str]:
    """Extract meaningful terms from expected_answer for matching."""
    # Split on commas, semicolons, numbered list markers, parentheses
    parts = re.split(r"[,;()\n]|\d+\.", text)
    terms = []
    for part in parts:
        term = part.strip().strip(".")
        # Keep terms that are at least 3 chars and not pure stopwords
        if len(term) >= 3 and term.lower() not in {"the", "and", "for", "via", "not", "yes", "no"}:
            terms.append(term)
    return terms


def _score(generated: str, expected: str) -> tuple[float, list[str], list[str]]:
    """
    Return (score, found_terms, missing_terms).
    Score = fraction of expected key terms found in the generated answer (case-insensitive).
    """
    gen_lower = generated.lower()
    terms = _key_terms(expected)
    if not terms:
        return 1.0, [], []

    found = [t for t in terms if t.lower() in gen_lower]
    missing = [t for t in terms if t.lower() not in gen_lower]
    return len(found) / len(terms), found, missing


def evaluate(questions_path: Path, threshold: float, verbose: bool) -> None:
    data = yaml.safe_load(questions_path.read_text())
    structural = [
        q for q in data["questions"]
        if q.get("category") == "structural"
    ]

    if not structural:
        print("No structural questions found in the questions file.")
        return

    passed = 0
    failed = 0

    print(f"\nEvaluating {len(structural)} structural questions  (threshold={threshold:.0%})\n")
    print(f"{'─' * 70}")

    for entry in structural:
        question = entry["question"]
        expected = entry.get("expected_answer", "")
        notes = entry.get("notes", "")

        result = answer(question)
        generated = result.get("answer", "")

        score, found, missing = _score(generated, expected)
        hit = score >= threshold
        status = "PASS" if hit else "FAIL"

        if hit:
            passed += 1
        else:
            failed += 1

        print(f"[{status}] {question}")
        print(f"       score={score:.0%} ({len(found)}/{len(found)+len(missing)} key terms found)")
        if notes:
            print(f"       {notes}")

        if not hit and verbose:
            print(f"       Expected answer: {expected}")
            print(f"       Missing terms: {missing}")
            print(f"       Generated answer:\n         {generated[:500]}{'...' if len(generated) > 500 else ''}")

        print()

    total = passed + failed
    overall = passed / total * 100 if total else 0
    print(f"{'─' * 70}")
    print(f"Result: {passed}/{total} passed  ({overall:.0f}%)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Fraction of expected key terms that must appear (default 0.5)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show full answers and missing terms on failure")
    args = parser.parse_args()
    evaluate(args.questions, args.threshold, args.verbose)
