# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

An "Ask Media Suite" RAG chatbot for researchers using the CLARIAH Media Suite. Researchers ask questions in natural language and get answers grounded in the official Help, How-to, FAQ, Tutorial and Glossary content, with direct links back to the relevant pages on mediasuite.clariah.nl.

The chatbot will be embedded on the Media Suite Community site at https://roelandordelman.github.io/media-suite-community/

## This repo is the application layer only

All ingestion, embedding, and vector store infrastructure lives in the separate [mediasuite-knowledge-base](https://github.com/roelandordelman/mediasuite-knowledge-base) repository (local path: `/Users/roeland.ordelman/Projects/mediasuite-knowledge-base`). This repo contains only:

- `api/` — FastAPI backend: RAG pipeline, retrieval router, SPARQL query library
- `widget/` — embeddable vanilla JS chat widget
- `evaluate/` — retrieval and router evaluation scripts
- `config.yaml` — all connection config: ChromaDB, Fuseki, entity/tool/collection mappings

Do not add ingestion or embedding code here.

## Stack

- **Generation + query expansion**: llama3.1:8b via Ollama (local)
- **Embeddings + SPARQL routing**: nomic-embed-text via Ollama (local)
- **Vector store**: ChromaDB HTTP server (port 8001), built in mediasuite-knowledge-base
- **Knowledge graph**: Apache Jena Fuseki (port 3030, dataset `mediasuite`), built in mediasuite-knowledge-base
- **Backend**: FastAPI + uvicorn
- **Frontend**: vanilla JS widget, no framework

## Project structure

```
api/
  main.py            — FastAPI app (POST /ask endpoint, conversation history)
  rag.py             — Full RAG pipeline: both paths always run → generate
  router.py          — Structural path: SPARQL query execution + result formatting
  query_index.py     — Embedding-based SPARQL query selector (QueryIndex singleton)
  sparql_queries.py  — Named SPARQL query library (10 templates) + run_query()
widget/
  chatbot.js         — Floating chat widget, maintains conversation history
  chatbot.html       — Standalone test page
evaluate/
  test_questions.yaml    — All eval questions (narrative + structural, with category field)
  eval_retrieval.py      — Tests vector-search retrieval for narrative questions (checks URLs)
  eval_router.py         — Tests full answer() pipeline for structural questions (checks key terms)
config.yaml          — ChromaDB + Fuseki connection, tool_entities, collection_entities mappings
debug_rag.py         — CLI: full pipeline debug (queries, chunks, context, answer)
query_debug.py       — CLI: retrieval-only debug (chunks + scores, no generation)
```

## Retrieval architecture

Both paths always run for every question. The LLM is only used for query expansion and answer generation — not for routing.

**Structural path** ("what tools exist for X?", "which collections are open?", "what workflows use Y?"):
1. `QueryIndex.select()` — embeds the question, computes cosine similarity against pre-embedded trigger questions per named query, selects queries above threshold (0.60). Deterministic.
2. For parametric queries, fills URI slots by embedding similarity against known entity names (tool names, collection names, workflow names, tadirah activity labels)
3. SPARQL runs against Fuseki; rows are formatted as `[Knowledge graph facts]` context
4. Entity URIs from results also filter ChromaDB for supporting documentation chunks
5. Returns empty when no queries exceed threshold (question is purely narrative)

**Narrative path** ("how do I annotate?", "what is the Media Suite?"):
1. `_expand_query()` — llama3.1:8b generates 3 alternative phrasings
2. `_retrieve()` — embed all variants, semantic search in ChromaDB (top_k×6 candidates)
3. Priority slots: FAQ/Help/How-to chunks get 2 reserved result slots so tutorial volume can't crowd them out
4. Dedup by title+section, then by URL
5. **CRAG gate** — if structural returned nothing and best narrative distance > `CRAG_RETRIEVAL_THRESHOLD` (0.75), `_reformulate_query()` asks the LLM to rephrase with different vocabulary, then retries retrieval once; results are merged (best-score-per-URL wins)

The LLM generates an answer from whatever context both paths returned. If only narrative context is present (structural returned nothing), the narrative relevance threshold still applies.

## Chunk schema (ChromaDB metadata fields)

Every chunk has these metadata fields:
- `url` — always present; must be cited in answers (direct link back to source page)
- `content_type` — distinguishes `How-to Guide` / `FAQ` / `Tool Tutorial` / `Subject Tutorial` / `Help / Documentation` / `Glossary` / etc.; used for priority-slot retrieval
- `entity_uri` — `ms:` URI of the primary tool or collection the chunk is about (e.g. `https://mediasuite.clariah.nl/vocab#AnnotationTool`); set on how-to pages and collection pages; empty string on general content. Used by the structural path to filter by entity.
- `title`, `section` — for dedup and context formatting
- `tags`, `categories`, `tools_mentioned`, `collections_mentioned` — JSON-encoded lists stored as strings; decoded by `_decode_meta()`

## Config structure

`config.yaml` has three top-level sections:
- `knowledge_base` — ChromaDB host/port/collection
- `knowledge_graph` — Fuseki URL/dataset/credentials + `tool_entities` (tool name → entity URI + tadirah activities) + `collection_entities` (collection name → entity URI)

The `tool_entities` and `collection_entities` maps are sourced from `mediasuite-knowledge-base/config.yaml` and must be kept in sync manually when the knowledge base adds tools or collections.

## Evaluation

```bash
python evaluate/eval_retrieval.py              # narrative questions, checks URL presence in top-5
python evaluate/eval_retrieval.py --verbose    # show retrieved vs expected URLs on failure
python evaluate/eval_router.py                 # structural questions, scores key terms in generated answer
python evaluate/eval_router.py --verbose       # show full answers on failure
```

Baseline: 14/14 narrative (100%). Structural: 26/26 (100%) with embedding-based routing; was 3-5/10 with LLM-based routing. Occasional LLM non-determinism in answer generation (~1 failure per run at 50% threshold); routing itself is deterministic.

`eval_retrieval.py` skips questions with `category: structural`. `eval_router.py` only runs `category: structural` questions.

## Key conventions

- Always preserve the `url` field from chunks in answers so researchers get direct links.
- Answers must be grounded strictly in retrieved chunks and/or SPARQL facts, not general knowledge.
- Keep the widget embeddable via a single `<script>` tag.
- All connection config lives in `config.yaml` only — never hardcode hosts, ports, or URIs in `api/`.
- The named graph URI `https://mediasuite.clariah.nl/graph` must be included in every SPARQL query (`FROM <...>`) — data is not in the default graph.
