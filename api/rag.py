"""
RAG pipeline: expand query → embed → retrieve chunks → generate grounded answer.

Query expansion generates alternative phrasings of the user's question before
retrieval, so vocabulary mismatches (e.g. "work with" vs "access") don't cause
relevant chunks to be missed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import ollama
import chromadb
import yaml

logger = logging.getLogger(__name__)
_qlog = logging.getLogger("questions")

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

# Defaults — overridden at runtime by the [rag] section in config.yaml.
# These also serve as the values used by debug_rag.py / query_debug.py.
EMBED_MODEL = "nomic-embed-text"
GENERATE_MODEL = "llama3.1:8b"
TOP_K = 5
MAX_DISTANCE = 1.0
MIN_BODY_CHARS = 150
CRAG_RETRIEVAL_THRESHOLD = 0.75
PRIORITY_SLOTS = 2

NO_ANSWER_RESPONSE = (
    "I don't have information about that in the Media Suite documentation. "
    "You can browse the full documentation at https://mediasuite.clariah.nl/documentation."
)

_SYSTEM_PROMPT_BASE = (
    "You are a Media Suite assistant for CLARIAH researchers. "
    "Answer using ONLY the provided context, which may include: "
    "(1) Media Suite documentation text, "
    "(2) knowledge graph facts about tools, collections, and workflows, "
    "(3) wiki background information about persons, productions, and topics "
    "from Dutch media history (Beeld & Geluid Wiki). "
    "Do not use outside knowledge."
)

_USER_PROMPT_TEMPLATE = """\
CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer using ONLY the CONTEXT above.
- Begin your answer by naming the source(s) you are drawing from:
    • For [Gestructureerde data] or [Beeld & Geluid Wiki — achtergrond] blocks: start with "According to the Beeld & Geluid Wiki, …"
    • For [Knowledge graph facts] blocks: start with "According to the Media Suite knowledge graph, …"
    • For documentation chunks (no special tag): start with "According to the Media Suite documentation, …"
    • If multiple sources contribute, mention each briefly: "Based on the Beeld & Geluid Wiki and the Media Suite documentation, …"
- Knowledge graph facts (marked [Knowledge graph facts]) are authoritative for questions about Media Suite tools, collections, and workflows.
- Structured data blocks (marked [Gestructureerde data]) are complete database query results from the Beeld & Geluid Wiki. Present these as complete lists. Do NOT summarize, select examples, or add any information that is not in the list itself.
- Wiki background (marked [Beeld & Geluid Wiki — achtergrond]) provides context about Dutch media history persons and productions. Note that wiki content may be outdated.
- If the context does not contain a clear answer, write ONLY this line: "I don't have information about that in the Media Suite documentation."
- Do not speculate or add information from outside the context.
- End your answer with the source URLs from the context chunks you used (if any).

ANSWER:"""

EXPANSION_PROMPT = """Generate 3 search queries to retrieve documentation that answers the following question.
Mix natural-language phrasings with keyword-style queries (e.g. "Media Suite definition overview").
Output only the 3 queries, one per line, no numbering or explanation.

Question: {question}"""

REFORMULATION_PROMPT = """The following question produced poor search results against documentation.
Rephrase it using different vocabulary, synonyms, or a more specific/general form that might match documentation language better.
Output only the rephrased question, nothing else.

Original question: {question}"""

STANDALONE_REWRITE_PROMPT = """Given the conversation history below and a follow-up question, rewrite the follow-up as a fully self-contained question that can be understood without the conversation history.
Keep it concise. Output only the rewritten question, nothing else. If the question is already self-contained, output it unchanged.

Conversation history:
{history_text}

Follow-up question: {question}"""

# If the best (lowest) narrative L2 distance is above this and the structural path
# returned nothing, CRAG fires: reformulate the question and retry retrieval once.
# Overridable via config.yaml [rag] section.

# Pronouns and demonstratives that suggest a question refers back to prior context.
# Used as a fast pre-filter before calling the LLM for standalone rewrite.
_FOLLOWUP_SIGNALS = frozenset([
    "it", "its", "they", "them", "their", "theirs",
    "this", "that", "these", "those",
    "the tool", "the collection", "the workflow", "the service",
    "the first", "the second", "the third", "the last",
    "which one", "that one", "the one",
])

# JSON-encoded list fields that ChromaDB stores as strings
_JSON_FIELDS = ("tags", "categories", "tools_mentioned", "collections_mentioned")


def _load_config() -> tuple[dict, dict, dict, dict]:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    return (
        cfg["knowledge_base"],
        cfg.get("knowledge_graph", {}),
        cfg.get("rag", {}),
        cfg.get("wiki_api", {}),
    )


def _decode_meta(meta: dict) -> dict:
    for field in _JSON_FIELDS:
        if field in meta:
            meta[field] = json.loads(meta[field])
    return meta


def _get_collection() -> chromadb.Collection:
    cfg, *_ = _load_config()
    client = chromadb.HttpClient(host=cfg["chroma_host"], port=cfg["chroma_port"])
    return client.get_collection(cfg["collection_name"])


def _expand_query(question: str, model: str = GENERATE_MODEL) -> list[str]:
    """Return the original question plus LLM-generated alternative phrasings."""
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": EXPANSION_PROMPT.format(question=question)}],
    )
    alternatives = [
        line.strip()
        for line in response["message"]["content"].splitlines()
        if line.strip()
    ]
    # Always include the original so it anchors retrieval
    return [question] + alternatives[:3]


def _reformulate_query(question: str, model: str = GENERATE_MODEL) -> str | None:
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": REFORMULATION_PROMPT.format(question=question)}],
        )
        reformulated = response["message"]["content"].strip()
        return reformulated if reformulated and reformulated.lower() != question.lower() else None
    except Exception:
        logger.warning("Query reformulation failed", exc_info=True)
        return None


def _rewrite_as_standalone(question: str, history: list[dict], model: str = GENERATE_MODEL) -> str:
    """
    If the question looks like a follow-up, rewrite it as a self-contained query.
    Returns the rewritten question, or the original if rewrite is not needed or fails.
    Only called when history is non-empty.
    """
    q_lower = question.lower()
    # Fast path: skip LLM call if no follow-up signals are present
    if not any(signal in q_lower for signal in _FOLLOWUP_SIGNALS):
        return question

    # Build a compact history snippet (last 3 turns max, role + content)
    recent = history[-6:]  # up to 3 user+assistant pairs
    history_text = "\n".join(
        f"{m['role'].capitalize()}: {m['content'][:300]}"
        for m in recent
        if m.get("role") in ("user", "assistant") and m.get("content")
    )
    if not history_text:
        return question

    try:
        response = ollama.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": STANDALONE_REWRITE_PROMPT.format(
                    history_text=history_text, question=question
                ),
            }],
        )
        rewritten = response["message"]["content"].strip()
        return rewritten if rewritten else question
    except Exception:
        logger.warning("Standalone rewrite failed", exc_info=True)
        return question


def _deduplicate(
    docs: list[str], metadatas: list[dict], distances: list[float]
) -> tuple[list[str], list[dict], list[float]]:
    """Keep only the highest-scoring (lowest-distance) chunk per (url, section) pair."""
    seen: dict[tuple, tuple] = {}
    for doc, meta, dist in zip(docs, metadatas, distances):
        key = (meta["url"], meta.get("section", ""))
        if key not in seen or dist < seen[key][2]:
            seen[key] = (doc, meta, dist)
    deduped = sorted(seen.values(), key=lambda x: x[2])
    if not deduped:
        return [], [], []
    docs_out, metas_out, dists_out = zip(*deduped)
    return list(docs_out), list(metas_out), list(dists_out)


# Content types that get reserved result slots so they can't be crowded out by
# tutorial volume. FAQ and Help answer "what is" questions; How-to Guide answers
# "how do I" questions — both are authoritative and tend to have lower semantic
# similarity scores than tutorial content despite being more relevant.
_PRIORITY_TYPES = {"FAQ", "Help", "How-to Guide"}


def _retrieve(
    queries: list[str],
    collection: chromadb.Collection,
    top_k: int,
    embed_model: str = EMBED_MODEL,
    min_body_chars: int = MIN_BODY_CHARS,
    priority_slots: int = PRIORITY_SLOTS,
) -> tuple[list, list, list]:
    """Embed all queries, return top_k results with priority_slots reserved for FAQ/Help/How-to."""
    embeddings = ollama.embed(model=embed_model, input=queries)["embeddings"]

    def _collect(results, store):
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            body = "\n".join(doc.splitlines()[1:]).strip()
            if len(body) < min_body_chars:
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
        n_results=priority_slots * 4,
        where={"content_type": {"$in": list(_PRIORITY_TYPES)}},
        include=["documents", "metadatas", "distances"],
    )
    _collect(priority_results, priority)

    # Build final list: top priority_slots from priority pool +
    # top (top_k - priority_slots) from semantic pool (excluding already-included URLs)
    priority_ranked = sorted(priority.values(), key=lambda x: x[2])[:priority_slots]
    priority_urls = {r[1]["url"] for r in priority_ranked}

    semantic_ranked = [
        r for r in sorted(semantic.values(), key=lambda x: x[2])
        if r[1]["url"] not in priority_urls
    ][: top_k - priority_slots]

    combined = priority_ranked + semantic_ranked
    docs  = [r[0] for r in combined]
    metas = [r[1] for r in combined]
    dists = [r[2] for r in combined]

    return _deduplicate(docs, metas, dists)


def answer(question: str, history: list[dict] = None, top_k: int = TOP_K, debug: bool = False) -> dict:
    """
    Return {"answer": str, "sources": [{"title": str, "url": str}]}.
    When debug=True, also include "_debug": {"sparql_queries", "sparql_context_preview", "entity_uris"}.

    Three retrieval paths always run:
    - Structural: embedding similarity selects SPARQL queries → Fuseki returns facts.
                  Returns empty when no queries exceed the similarity threshold.
    - Narrative:  LLM expands the question → embed → ChromaDB semantic search.
    - Wiki:       semantic search against the Beeld & Geluid Wiki (via wiki REST API).
                  Returns empty when the wiki API is down or no relevant results found.
    The LLM generates an answer from whatever context all three paths returned.
    """
    from api.router import sparql_query_structural, retrieve_by_entity_uris

    kb_cfg, kg_cfg, rag_cfg, wiki_cfg = _load_config()
    collection = _get_collection()

    # Runtime overrides from config.yaml [rag] section; fall back to module defaults
    embed_model    = rag_cfg.get("embed_model", EMBED_MODEL)
    generate_model = rag_cfg.get("generate_model", GENERATE_MODEL)
    effective_top_k    = rag_cfg.get("top_k", top_k)
    max_distance       = rag_cfg.get("max_distance", MAX_DISTANCE)
    min_body_chars     = rag_cfg.get("min_body_chars", MIN_BODY_CHARS)
    crag_threshold        = rag_cfg.get("crag_retrieval_threshold", CRAG_RETRIEVAL_THRESHOLD)
    priority_slots        = rag_cfg.get("priority_slots", PRIORITY_SLOTS)
    query_index_threshold = rag_cfg.get("query_index_threshold", 0.60)

    # History-aware rewrite: if this looks like a follow-up, resolve references
    # before retrieval so embeddings match documentation vocabulary.
    # Generation always uses the original question so the answer reads naturally.
    retrieval_question = (
        _rewrite_as_standalone(question, history, generate_model) if history else question
    )

    # Structural path — always attempt; no LLM involved in routing
    sparql_context = ""
    sparql_selections = []
    entity_uris = []
    if kg_cfg.get("fuseki_url"):
        sparql_context, entity_uris, sparql_selections = sparql_query_structural(
            retrieval_question, kg_cfg, embed_model, threshold=query_index_threshold
        )

    # Wiki path — dual-path retrieval via wiki agent /ask endpoint (optional)
    wiki_context = ""
    wiki_sources: list[dict] = []
    if wiki_cfg.get("url"):
        from api.wiki_client import retrieve_wiki
        wiki_response = retrieve_wiki(
            wiki_cfg["url"],
            retrieval_question,
            top_k=wiki_cfg.get("top_k", 3),
            min_score=wiki_cfg.get("min_score", 0.70),
        )
        wiki_context = wiki_response.get("context", "")
        wiki_sources = wiki_response.get("sources", [])

    # Narrative path — always run
    queries = _expand_query(retrieval_question, generate_model)
    docs, metas, distances = _retrieve(
        queries, collection, effective_top_k, embed_model, min_body_chars, priority_slots
    )

    # CRAG: if structural returned nothing and narrative retrieval is weak, reformulate once
    crag_triggered = False
    if not sparql_context and (not distances or distances[0] > crag_threshold):
        reformulated = _reformulate_query(retrieval_question, generate_model)
        if reformulated:
            crag_triggered = True
            r_docs, r_metas, r_dists = _retrieve(
                [reformulated], collection, effective_top_k, embed_model, min_body_chars, priority_slots
            )
            # Merge: best-scoring chunk per (url, section) pair wins
            url_map: dict[tuple, tuple] = {
                (m["url"], m.get("section", "")): (d, m, di)
                for d, m, di in zip(docs, metas, distances)
            }
            for d, m, dist in zip(r_docs, r_metas, r_dists):
                key = (m["url"], m.get("section", ""))
                if key not in url_map or dist < url_map[key][2]:
                    url_map[key] = (d, m, dist)
            merged = sorted(url_map.values(), key=lambda x: x[2])[:effective_top_k]
            docs      = [x[0] for x in merged]
            metas     = [x[1] for x in merged]
            distances = [x[2] for x in merged]

    # If structural found entity URIs, merge entity-specific chunks into results
    if entity_uris:
        entity_docs, entity_metas, entity_dists = retrieve_by_entity_uris(
            question, entity_uris, collection, embed_model, effective_top_k
        )
        existing_keys = {(m["url"], m.get("section", "")) for m in metas}
        for d, m, dist in zip(entity_docs, entity_metas, entity_dists):
            if (m["url"], m.get("section", "")) not in existing_keys:
                docs.append(d)
                metas.append(m)
                distances.append(dist)

    # Bail out only if all three paths returned nothing useful
    if not sparql_context and not wiki_context and (not distances or distances[0] > max_distance):
        _qlog.info("Q: %r | sparql:%s | wiki:%s | crag:%s | no-answer:threshold", question, bool(sparql_context), bool(wiki_context), crag_triggered)
        return {"answer": NO_ANSWER_RESPONSE, "sources": []}

    # Build context: SPARQL facts first, then wiki background, then chunk text
    context_parts = []
    if sparql_context:
        context_parts.append(f"[Knowledge graph facts]\n{sparql_context}")
    if wiki_context:
        context_parts.append(wiki_context)
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

    response = ollama.chat(model=generate_model, messages=messages)
    answer_text = response["message"]["content"]

    # Don't return sources if the LLM couldn't answer from the context
    if "I don't have information about that" in answer_text:
        _qlog.info("Q: %r | sparql:%s | wiki:%s | crag:%s | no-answer:llm", question, bool(sparql_context), bool(wiki_context), crag_triggered)
        result = {"answer": answer_text, "sources": []}
        if debug:
            result["_debug"] = _build_debug(
                sparql_selections, sparql_context, entity_uris, crag_triggered,
                retrieval_question if retrieval_question != question else None,
                wiki_sources,
            )
        return result

    seen = set()
    unique_sources = []
    for m in metas:
        if m.get("url") and m["url"] not in seen:
            seen.add(m["url"])
            unique_sources.append({"title": m["title"], "url": m["url"]})
    for r in wiki_sources:
        if r.get("url") and r["url"] not in seen:
            seen.add(r["url"])
            unique_sources.append({"title": r.get("title", ""), "url": r["url"]})

    _qlog.info("Q: %r | sparql:%s | wiki:%s | crag:%s | sources:%d", question, bool(sparql_context), bool(wiki_context), crag_triggered, len(unique_sources))
    result = {"answer": answer_text, "sources": unique_sources}
    if debug:
        result["_debug"] = _build_debug(
            sparql_selections, sparql_context, entity_uris, crag_triggered,
            retrieval_question if retrieval_question != question else None,
            wiki_sources,
        )
    return result


def _build_debug(
    selections: list,
    sparql_context: str,
    entity_uris: list,
    crag_triggered: bool = False,
    rewritten_query: str | None = None,
    wiki_sources: list | None = None,
) -> dict:
    query_labels = [
        name if not params else f"{name}({', '.join(f'{k}=…{v[-20:]}' for k, v in params.items())})"
        for name, params in selections
    ]
    result = {
        "sparql_queries": query_labels,
        "sparql_context_preview": sparql_context[:400] if sparql_context else "(empty)",
        "entity_uris": entity_uris,
        "crag_triggered": crag_triggered,
        "wiki_sources": len(wiki_sources) if wiki_sources else 0,
    }
    if rewritten_query:
        result["rewritten_query"] = rewritten_query
    return result
