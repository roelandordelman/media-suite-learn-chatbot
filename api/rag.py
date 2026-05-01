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
GENERATE_MODEL = "llama3.1:8b"
TOP_K = 5

# Chunks with L2 distance above this are considered too dissimilar to be useful.
# Lower = stricter. Run query_debug.py to see typical scores for your queries
# and tune this value accordingly.
MAX_DISTANCE = 1.0

# Chunks whose body (text after the first context-prefix line) is shorter than this
# are header/stub chunks with no real content and are skipped during retrieval.
MIN_BODY_CHARS = 150

NO_ANSWER_RESPONSE = (
    "I don't have information about that in the Media Suite documentation. "
    "You can browse the full documentation at https://mediasuite.clariah.nl/documentation."
)

_SYSTEM_PROMPT_BASE = "You are a Media Suite documentation assistant. Answer using ONLY the provided context, which may include documentation text and knowledge graph facts. Do not use outside knowledge."

_USER_PROMPT_TEMPLATE = """\
CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer using ONLY the CONTEXT above (documentation text and/or knowledge graph facts).
- Knowledge graph facts (marked [Knowledge graph facts]) are authoritative — use them directly to answer structural questions about tools, collections, workflows, and access rights.
- If the context does not contain a clear answer, write ONLY this line: "I don't have information about that in the Media Suite documentation."
- Do not speculate or add information from outside the context.
- End your answer with the source URLs from the context chunks you used (if any).

ANSWER:"""

EXPANSION_PROMPT = """Generate 3 search queries to retrieve documentation that answers the following question.
Mix natural-language phrasings with keyword-style queries (e.g. "Media Suite definition overview").
Output only the 3 queries, one per line, no numbering or explanation.

Question: {question}"""

# JSON-encoded list fields that ChromaDB stores as strings
_JSON_FIELDS = ("tags", "categories", "tools_mentioned", "collections_mentioned")


def _load_config() -> dict:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    return cfg["knowledge_base"], cfg.get("knowledge_graph", {})


def _decode_meta(meta: dict) -> dict:
    for field in _JSON_FIELDS:
        if field in meta:
            meta[field] = json.loads(meta[field])
    return meta


def _get_collection() -> chromadb.Collection:
    cfg, _ = _load_config()
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


def _deduplicate_by_url(
    docs: list[str], metadatas: list[dict], distances: list[float]
) -> tuple[list[str], list[dict], list[float]]:
    """Keep only the highest-scoring (lowest-distance) chunk per source URL."""
    seen: dict[str, tuple[str, dict, float]] = {}
    for doc, meta, dist in zip(docs, metadatas, distances):
        url = meta["url"]
        if url not in seen or dist < seen[url][2]:
            seen[url] = (doc, meta, dist)
    deduped = sorted(seen.values(), key=lambda x: x[2])
    docs_out, metas_out, dists_out = zip(*deduped) if deduped else ([], [], [])
    return list(docs_out), list(metas_out), list(dists_out)


# Content types that get reserved result slots so they can't be crowded out by
# tutorial volume. FAQ and Help answer "what is" questions; How-to Guide answers
# "how do I" questions — both are authoritative and tend to have lower semantic
# similarity scores than tutorial content despite being more relevant.
_PRIORITY_TYPES = {"FAQ", "Help", "How-to Guide"}
PRIORITY_SLOTS = 2  # of top_k results reserved for priority-type chunks


def _retrieve(queries: list[str], collection: chromadb.Collection, top_k: int) -> tuple[list, list, list]:
    """Embed all queries, return top_k results with PRIORITY_SLOTS reserved for FAQ/Help/How-to."""
    embeddings = ollama.embed(model=EMBED_MODEL, input=queries)["embeddings"]

    def _collect(results, store):
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            body = "\n".join(doc.splitlines()[1:]).strip()
            if len(body) < MIN_BODY_CHARS:
                continue
            meta = _decode_meta(meta)
            # Dedup by title+section — collapses tool-tutorial/subject-tutorial duplicates
            chunk_id = meta.get("title", "") + "|" + meta.get("section", "")
            if chunk_id not in store or dist < store[chunk_id][2]:
                store[chunk_id] = (doc, meta, dist)

    # Semantic search across all query variants
    semantic: dict[str, tuple] = {}
    for embedding in embeddings:
        results = collection.query(
            query_embeddings=[embedding],
            n_results=top_k * 6,
            include=["documents", "metadatas", "distances"],
        )
        _collect(results, semantic)

    # Priority pull: best FAQ/Help/How-to chunks for the original query
    priority: dict[str, tuple] = {}
    priority_results = collection.query(
        query_embeddings=[embeddings[0]],
        n_results=PRIORITY_SLOTS * 4,
        where={"content_type": {"$in": list(_PRIORITY_TYPES)}},
        include=["documents", "metadatas", "distances"],
    )
    _collect(priority_results, priority)

    # Build final list: top PRIORITY_SLOTS from priority pool +
    # top (top_k - PRIORITY_SLOTS) from semantic pool (excluding already-included URLs)
    priority_ranked = sorted(priority.values(), key=lambda x: x[2])[:PRIORITY_SLOTS]
    priority_urls = {r[1]["url"] for r in priority_ranked}

    semantic_ranked = [
        r for r in sorted(semantic.values(), key=lambda x: x[2])
        if r[1]["url"] not in priority_urls
    ][: top_k - PRIORITY_SLOTS]

    combined = priority_ranked + semantic_ranked
    docs  = [r[0] for r in combined]
    metas = [r[1] for r in combined]
    dists = [r[2] for r in combined]

    docs, metas, dists = _deduplicate_by_url(docs, metas, dists)
    return docs, metas, dists


def answer(question: str, history: list[dict] = None, top_k: int = TOP_K, debug: bool = False) -> dict:
    """
    Return {"answer": str, "sources": [{"title": str, "url": str}]}.
    When debug=True, also include "_debug": {"route", "sparql_queries", "sparql_context", "entity_uris"}.
    """
    from api.router import classify_question, sparql_query_structural, retrieve_by_entity_uris

    _, kg_cfg = _load_config()
    collection = _get_collection()

    route = "narrative"
    kg_enabled = bool(kg_cfg.get("fuseki_url"))
    if kg_enabled:
        route = classify_question(question, GENERATE_MODEL)

    sparql_context = ""
    sparql_selections = []
    entity_uris = []
    docs, metas, distances = [], [], []

    if route == "structural" and kg_enabled:
        sparql_context, entity_uris, sparql_selections = sparql_query_structural(question, kg_cfg, GENERATE_MODEL)
        if entity_uris:
            docs, metas, distances = retrieve_by_entity_uris(
                question, entity_uris, collection, EMBED_MODEL, top_k
            )
        # Fall back to vector search only if SPARQL also returned nothing
        if not sparql_context:
            route = "narrative"

    if route == "narrative":
        queries = _expand_query(question)
        docs, metas, distances = _retrieve(queries, collection, top_k)

    # For structural: SPARQL context is primary; chunks are supplemental.
    # For narrative: only chunk context, apply relevance threshold.
    if route == "narrative":
        if not distances or distances[0] > MAX_DISTANCE:
            return {"answer": NO_ANSWER_RESPONSE, "sources": []}

    # Build context: SPARQL facts first, then chunk text
    context_parts = []
    if sparql_context:
        context_parts.append(f"[Knowledge graph facts]\n{sparql_context}")
    if docs:
        chunk_text = "\n\n---\n\n".join(
            f"[{m['content_type']}] {m['title']}{(' — ' + m['section']) if m.get('section') else ''}\nURL: {m['url']}\n\n{doc}"
            for doc, m in zip(docs, metas)
        )
        context_parts.append(chunk_text)
    context = "\n\n---\n\n".join(context_parts)

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT_BASE},
        *(history or []),
        {"role": "user", "content": _USER_PROMPT_TEMPLATE.format(context=context, question=question)},
    ]

    response = ollama.chat(model=GENERATE_MODEL, messages=messages)
    answer_text = response["message"]["content"]

    # Don't return sources if the LLM couldn't answer from the context
    if "I don't have information about that" in answer_text:
        result = {"answer": answer_text, "sources": []}
        if debug:
            result["_debug"] = _build_debug(route, sparql_selections, sparql_context, entity_uris)
        return result

    seen = set()
    unique_sources = []
    for m in metas:
        if m.get("url") and m["url"] not in seen:
            seen.add(m["url"])
            unique_sources.append({"title": m["title"], "url": m["url"]})

    result = {"answer": answer_text, "sources": unique_sources}
    if debug:
        result["_debug"] = _build_debug(route, sparql_selections, sparql_context, entity_uris)
    return result


def _build_debug(route: str, selections: list, sparql_context: str, entity_uris: list) -> dict:
    query_labels = [
        name if not params else f"{name}({', '.join(f'{k}=…{v[-20:]}' for k, v in params.items())})"
        for name, params in selections
    ]
    return {
        "route": route,
        "sparql_queries": query_labels,
        "sparql_context_preview": sparql_context[:400] if sparql_context else "(empty)",
        "entity_uris": entity_uris,
    }
