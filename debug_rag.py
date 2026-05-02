"""
Run the full RAG pipeline for a question and print every intermediate step:
  1. Expanded query variants
  2. Retrieved + deduplicated chunks with similarity scores
  3. CRAG check — if best distance > threshold, reformulate and retry
  4. Relevance threshold check
  5. The exact context string passed to the LLM
  6. The generated answer

Usage:
    python3 debug_rag.py "your question here"
    python3 debug_rag.py "your question here" --no-generate
    python3 debug_rag.py "your question here" --top-k 8
"""

import argparse
import sys
from pathlib import Path

import ollama

sys.path.insert(0, str(Path(__file__).parent))

from api.rag import (
    CRAG_RETRIEVAL_THRESHOLD,
    GENERATE_MODEL,
    MAX_DISTANCE,
    NO_ANSWER_RESPONSE,
    TOP_K,
    _SYSTEM_PROMPT_BASE,
    _USER_PROMPT_TEMPLATE,
    _expand_query,
    _get_collection,
    _reformulate_query,
    _retrieve,
)

SEP = "─" * 70


def run(question: str, top_k: int, generate: bool) -> None:
    print(f"\n{SEP}")
    print(f"QUESTION: {question}")
    print(SEP)

    # 1. Query expansion
    queries = _expand_query(question)
    print(f"\n[1] QUERY EXPANSION  ({len(queries)} variants)")
    for i, q in enumerate(queries):
        label = "original" if i == 0 else f"variant {i}"
        print(f"    [{label}] {q}")

    # 2. Retrieval + deduplication
    collection = _get_collection()
    docs, metas, distances = _retrieve(queries, collection, top_k)

    print(f"\n[2] RETRIEVED CHUNKS  (top {len(docs)}, deduplicated by URL)")
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
        score = 1 - dist
        print(f"\n  #{i}  score={score:.4f}  dist={dist:.4f}")
        print(f"      type:    {meta['content_type']}")
        print(f"      title:   {meta['title']}")
        print(f"      section: {meta['section']}")
        print(f"      url:     {meta['url']}")
        print(f"      text:\n")
        for line in doc.splitlines():
            print(f"        {line}")

    # 3. CRAG — fire if best distance exceeds threshold (narrative only)
    print(f"\n[3] CRAG CHECK  (threshold={CRAG_RETRIEVAL_THRESHOLD})")
    if not distances or distances[0] > CRAG_RETRIEVAL_THRESHOLD:
        print(f"    TRIGGERED — best distance {distances[0] if distances else 'n/a':.4f} exceeds threshold.")
        reformulated = _reformulate_query(question)
        if reformulated:
            print(f"    Reformulated: {reformulated}")
            r_docs, r_metas, r_dists = _retrieve([reformulated], collection, top_k)
            url_map: dict = {m["url"]: (d, m, di) for d, m, di in zip(docs, metas, distances)}
            for d, m, dist in zip(r_docs, r_metas, r_dists):
                url = m["url"]
                if url not in url_map or dist < url_map[url][2]:
                    url_map[url] = (d, m, dist)
            merged = sorted(url_map.values(), key=lambda x: x[2])[:top_k]
            docs      = [x[0] for x in merged]
            metas     = [x[1] for x in merged]
            distances = [x[2] for x in merged]
            print(f"    After merge: {len(docs)} chunks, best distance {distances[0]:.4f}")
        else:
            print("    Reformulation returned nothing useful.")
    else:
        print(f"    OK — best distance {distances[0]:.4f} is below threshold.")

    # 4. Relevance threshold check
    print(f"\n[4] RELEVANCE CHECK  (MAX_DISTANCE={MAX_DISTANCE})")
    if not distances or distances[0] > MAX_DISTANCE:
        print(f"    FAIL — best distance {distances[0]:.4f} exceeds threshold.")
        print(f"    Would return: {NO_ANSWER_RESPONSE}")
        return
    print(f"    PASS — best distance {distances[0]:.4f} is within threshold.")

    # 5. Context string
    context = "\n\n---\n\n".join(
        f"[{m['content_type']}] {m['title']}{(' — ' + m['section']) if m.get('section') else ''}\nURL: {m['url']}\n\n{doc}"
        for doc, m in zip(docs, metas)
    )
    print(f"\n[5] CONTEXT STRING PASSED TO LLM")
    print(f"    ({len(context)} chars, {len(docs)} chunks)")
    print()
    for line in context.splitlines():
        print(f"    {line}")

    if not generate:
        print(f"\n{SEP}")
        print("(skipping generation — run without --no-generate to see the answer)")
        print(SEP)
        return

    # 6. Generation
    print(f"\n[6] GENERATING ANSWER  (model={GENERATE_MODEL})")
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT_BASE},
        {"role": "user", "content": _USER_PROMPT_TEMPLATE.format(context=context, question=question)},
    ]
    response = ollama.chat(model=GENERATE_MODEL, messages=messages)
    answer_text = response["message"]["content"]

    print(f"\n{SEP}")
    print("ANSWER:")
    print(SEP)
    print(answer_text)
    print(SEP)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--no-generate", action="store_true", help="Skip LLM generation, only show retrieved chunks")
    args = parser.parse_args()
    run(args.question, args.top_k, generate=not args.no_generate)
