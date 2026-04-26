"""
RAG pipeline: expand query → embed → retrieve chunks → generate grounded answer.

Query expansion generates alternative phrasings of the user's question before
retrieval, so vocabulary mismatches (e.g. "work with" vs "access") don't cause
relevant chunks to be missed.
"""

import json
from pathlib import Path

import ollama
import chromadb
import yaml

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

EMBED_MODEL = "nomic-embed-text"
GENERATE_MODEL = "mistral"
TOP_K = 5

# Chunks with L2 distance above this are considered too dissimilar to be useful.
# Lower = stricter. Run query_debug.py to see typical scores for your queries
# and tune this value accordingly.
MAX_DISTANCE = 1.0

NO_ANSWER_RESPONSE = (
    "I don't have information about that in the Media Suite documentation. "
    "You can browse the full documentation at https://mediasuite.clariah.nl/documentation."
)

SYSTEM_PROMPT = """You are a helpful assistant for researchers using the CLARIAH Media Suite.

Rules you must follow without exception:
- Answer ONLY using information explicitly present in the context chunks below.
- If the context does not contain a clear answer, respond with exactly: "I don't have information about that in the Media Suite documentation."
- Do NOT use any knowledge from your training data about Media Suite or related tools.
- Do NOT speculate, infer, or fill gaps with general knowledge.
- Always include the source URLs from the context chunks in your answer."""

EXPANSION_PROMPT = """Generate 3 alternative phrasings of the following question to improve document retrieval.
Use different vocabulary and sentence structure, but keep the same meaning.
Output only the 3 phrasings, one per line, no numbering or explanation.

Question: {question}"""

# JSON-encoded list fields that ChromaDB stores as strings
_JSON_FIELDS = ("tags", "categories", "tools_mentioned", "collections_mentioned")


def _load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())["knowledge_base"]


def _decode_meta(meta: dict) -> dict:
    for field in _JSON_FIELDS:
        if field in meta:
            meta[field] = json.loads(meta[field])
    return meta


def _get_collection() -> chromadb.Collection:
    cfg = _load_config()
    client = chromadb.HttpClient(host=cfg["chroma_host"], port=cfg["chroma_port"])
    return client.get_collection(cfg["collection_name"])


def _expand_query(question: str) -> list[str]:
    """Return the original question plus LLM-generated alternative phrasings."""
    response = ollama.chat(
        model=GENERATE_MODEL,
        messages=[{"role": "user", "content": EXPANSION_PROMPT.format(question=question)}],
    )
    alternatives = [
        line.strip()
        for line in response["message"]["content"].splitlines()
        if line.strip()
    ]
    # Always include the original so it anchors retrieval
    return [question] + alternatives[:3]


def _retrieve(queries: list[str], collection: chromadb.Collection, top_k: int) -> tuple[list, list, list]:
    """Embed all queries, retrieve top_k for each, merge by best distance."""
    embeddings = ollama.embed(model=EMBED_MODEL, input=queries)["embeddings"]

    # Map chunk id → (doc, meta, distance), keeping the best (lowest) distance
    best: dict[str, tuple] = {}
    for embedding in embeddings:
        results = collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            meta = _decode_meta(meta)
            chunk_id = meta["url"] + "|" + meta["section"]
            if chunk_id not in best or dist < best[chunk_id][2]:
                best[chunk_id] = (doc, meta, dist)

    # Sort by distance (ascending = most similar first) and return top_k
    ranked = sorted(best.values(), key=lambda x: x[2])[:top_k]
    docs  = [r[0] for r in ranked]
    metas = [r[1] for r in ranked]
    dists = [r[2] for r in ranked]
    return docs, metas, dists


def answer(question: str, history: list[dict] = None, top_k: int = TOP_K) -> dict:
    """Return {"answer": str, "sources": [{"title": str, "url": str}]}"""
    collection = _get_collection()

    queries = _expand_query(question)
    docs, metas, distances = _retrieve(queries, collection, top_k)

    # Bail out before calling the LLM if nothing is close enough
    if not distances or distances[0] > MAX_DISTANCE:
        return {"answer": NO_ANSWER_RESPONSE, "sources": []}

    context = "\n\n---\n\n".join(
        f"[{m['content_type']}] {m['title']} — {m['section']}\nURL: {m['url']}\n\n{doc}"
        for doc, m in zip(docs, metas)
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *(history or []),
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]

    response = ollama.chat(model=GENERATE_MODEL, messages=messages)
    answer_text = response["message"]["content"]

    # Don't return sources if the LLM couldn't answer from the context
    if "I don't have information about that" in answer_text:
        return {"answer": answer_text, "sources": []}

    seen = set()
    unique_sources = []
    for m in metas:
        if m.get("url") and m["url"] not in seen:
            seen.add(m["url"])
            unique_sources.append({"title": m["title"], "url": m["url"]})

    return {"answer": answer_text, "sources": unique_sources}
