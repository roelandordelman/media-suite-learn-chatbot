"""
Show expanded queries and top-k retrieved chunks for a question.
Useful for diagnosing retrieval quality without invoking the full LLM answer.

Usage:
    python3 query_debug.py "your question here"
    python3 query_debug.py "your question here" --top-k 10
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from api.rag import _expand_query, _retrieve, _get_collection


def debug_query(question: str, top_k: int = 5) -> None:
    print(f"\nOriginal query: {question}")

    queries = _expand_query(question)
    print(f"\nExpanded queries ({len(queries)} total):")
    for i, q in enumerate(queries):
        prefix = "  [original]" if i == 0 else f"  [variant {i}]"
        print(f"{prefix} {q}")

    collection = _get_collection()
    docs, metas, distances = _retrieve(queries, collection, top_k)

    print(f"\nTop {top_k} retrieved chunks (merged across all queries):\n")
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
        print(f"{'─' * 60}")
        print(f"#{i}  score: {1 - dist:.3f}  |  type: {meta['content_type']}")
        print(f"    title:   {meta['title']}")
        print(f"    section: {meta['section']}")
        print(f"    url:     {meta['url']}")
        print(f"    text:    {doc[:300]}{'…' if len(doc) > 300 else ''}")
    print(f"{'─' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    debug_query(args.question, args.top_k)
