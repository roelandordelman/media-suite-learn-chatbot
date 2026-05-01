"""
Evaluate retrieval quality against test_questions.yaml.

For each question, runs the retrieval pipeline and checks whether any
expected URL appears in the returned chunks. Reports pass/fail per question
and an overall score.

Usage:
    python3 evaluate/eval_retrieval.py
    python3 evaluate/eval_retrieval.py --questions path/to/other.yaml
    python3 evaluate/eval_retrieval.py --top-k 8
    python3 evaluate/eval_retrieval.py --verbose   # show retrieved URLs on failure
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.rag import TOP_K, _expand_query, _get_collection, _retrieve

DEFAULT_QUESTIONS = Path(__file__).parent / "test_questions.yaml"


def evaluate(questions_path: Path, top_k: int, verbose: bool) -> None:
    data = yaml.safe_load(questions_path.read_text())
    questions = [q for q in data["questions"] if q.get("category") != "structural"]

    collection = _get_collection()

    passed = 0
    failed = 0
    pending = 0

    print(f"\nEvaluating narrative questions  (top_k={top_k})\n")
    print(f"{'─' * 70}")

    for entry in questions:
        if entry.get("category") == "structural":
            continue

        question = entry["question"]
        notes    = entry.get("notes", "")

        if not entry.get("annotated", True) or not entry.get("expected_urls"):
            queries = _expand_query(question)
            docs, metas, distances = _retrieve(queries, collection, top_k)
            retrieved_urls = sorted({m["url"] for m in metas})
            print(f"[PENDING] {question}")
            if notes:
                print(f"          {notes}")
            print(f"          Retrieved:")
            for u in retrieved_urls:
                print(f"            {u}")
            print()
            pending += 1
            continue

        expected = set(entry["expected_urls"])
        queries = _expand_query(question)
        docs, metas, distances = _retrieve(queries, collection, top_k)
        retrieved_urls = {m["url"] for m in metas}

        hit = bool(expected & retrieved_urls)
        status = "PASS" if hit else "FAIL"
        if hit:
            passed += 1
        else:
            failed += 1

        print(f"[{status}] {question}")
        if notes:
            print(f"       {notes}")

        if not hit and verbose:
            print(f"       Expected one of:")
            for u in sorted(expected):
                print(f"         {u}")
            print(f"       Got:")
            for u in sorted(retrieved_urls):
                print(f"         {u}")

        print()

    total = passed + failed
    score = passed / total * 100 if total else 0
    print(f"{'─' * 70}")
    print(f"Result: {passed}/{total} passed  ({score:.0f}%)  |  {pending} pending annotation\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--verbose", action="store_true", help="Show retrieved vs expected URLs on failure")
    args = parser.parse_args()
    evaluate(args.questions, args.top_k, args.verbose)
