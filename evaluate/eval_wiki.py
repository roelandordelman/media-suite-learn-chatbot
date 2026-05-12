"""
End-to-end evaluator for wiki path questions.

Calls answer() for each wiki question and checks:
  1. Key terms — same fraction-threshold scoring as eval_router.py
  2. Attribution — if attribution_check: true, answer must contain "Beeld & Geluid Wiki"
  3. No attribution — if no_attribution_check: true, answer must NOT contain "Beeld & Geluid Wiki"

Requires the wiki REST API (port 8002) and Ollama to be running.

Usage:
    python evaluate/eval_wiki.py
    python evaluate/eval_wiki.py --threshold 0.5
    python evaluate/eval_wiki.py --verbose    # show full answers on failure
    python evaluate/eval_wiki.py --debug      # show wiki_sources count + SPARQL info
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.rag import answer

DEFAULT_QUESTIONS = Path(__file__).parent / "test_questions.yaml"
DEFAULT_THRESHOLD = 0.5
ATTRIBUTION_MARKER = "Beeld & Geluid Wiki"


def _score(generated: str, terms: list[str]) -> tuple[float, list[str], list[str]]:
    if not terms:
        return 1.0, [], []
    gen_lower = generated.lower()
    found = [t for t in terms if t.lower() in gen_lower]
    missing = [t for t in terms if t.lower() not in gen_lower]
    return len(found) / len(terms), found, missing


def evaluate(questions_path: Path, threshold: float, verbose: bool, debug: bool) -> None:
    data = yaml.safe_load(questions_path.read_text())
    wiki_qs = [q for q in data["questions"] if q.get("category") == "wiki"]

    if not wiki_qs:
        print("No wiki questions found in the questions file.")
        return

    passed = failed = pending = 0

    print(f"\nEvaluating {len(wiki_qs)} wiki questions  (threshold={threshold:.0%})\n")
    print(f"{'─' * 70}")

    for entry in wiki_qs:
        question = entry["question"]
        notes = entry.get("notes", "")

        if not entry.get("annotated", True):
            result = answer(question, debug=True)
            generated = result.get("answer", "")
            dbg = result.get("_debug", {})
            print(f"[PENDING] {question}")
            if notes:
                print(f"          {notes}")
            print(f"          wiki_sources={dbg.get('wiki_sources', 0)}")
            print(f"          Answer: {generated[:300]}{'…' if len(generated) > 300 else ''}")
            print()
            pending += 1
            continue

        result = answer(question, debug=True)
        generated = result.get("answer", "")
        dbg = result.get("_debug", {})
        wiki_sources = dbg.get("wiki_sources", 0)

        # Key term scoring
        raw_terms = entry.get("expected_terms")
        expected = entry.get("expected_answer", "")
        if raw_terms:
            terms = raw_terms
        else:
            terms = [t.strip() for t in expected.split(",") if t.strip()]

        score, found, missing = _score(generated, terms)
        terms_ok = score >= threshold or not terms

        # Attribution checks
        has_attribution = ATTRIBUTION_MARKER in generated
        attribution_check = entry.get("attribution_check", False)
        no_attribution_check = entry.get("no_attribution_check", False)

        attribution_ok = True
        attribution_note = ""
        if attribution_check and not has_attribution:
            attribution_ok = False
            attribution_note = f"MISSING attribution ({ATTRIBUTION_MARKER!r} not in answer)"
        if no_attribution_check and has_attribution:
            attribution_ok = False
            attribution_note = f"UNEXPECTED attribution ({ATTRIBUTION_MARKER!r} found in answer)"

        hit = terms_ok and attribution_ok
        status = "PASS" if hit else "FAIL"

        if hit:
            passed += 1
        else:
            failed += 1

        print(f"[{status}] {question}")
        if terms:
            print(f"       terms={score:.0%} ({len(found)}/{len(found)+len(missing)} found)  wiki_sources={wiki_sources}")
        else:
            print(f"       (no terms)  wiki_sources={wiki_sources}")
        if attribution_note:
            print(f"       {attribution_note}")
        if notes:
            print(f"       {notes}")

        if debug:
            print(f"       SPARQL queries: {dbg.get('sparql_queries', []) or '(none)'}")
            print(f"       crag_triggered: {dbg.get('crag_triggered', False)}")

        if not hit and verbose:
            if missing:
                print(f"       Missing terms: {missing}")
            print(f"       Generated answer:\n         {generated[:500]}{'...' if len(generated) > 500 else ''}")

        print()

    total = passed + failed
    overall = passed / total * 100 if total else 0
    print(f"{'─' * 70}")
    print(f"Result: {passed}/{total} passed  ({overall:.0f}%)  |  {pending} pending annotation\n")
    if total and failed:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--verbose", action="store_true",
                        help="Show full answers and missing terms on failure")
    parser.add_argument("--debug", action="store_true",
                        help="Show SPARQL queries selected and crag_triggered per question")
    args = parser.parse_args()
    evaluate(args.questions, args.threshold, args.verbose, args.debug)
