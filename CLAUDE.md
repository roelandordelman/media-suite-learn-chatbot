# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## What this project is

"Ask Media Suite" — a RAG chatbot for researchers using the [CLARIAH Media Suite](https://mediasuite.clariah.nl). Researchers ask questions in natural language; the chatbot answers using the official Media Suite documentation, the Media Suite knowledge graph, and the Beeld & Geluid Wiki. Answers include direct links to source pages.

The widget is embedded on the [Media Suite Community site](https://roelandordelman.github.io/media-suite-community/).

## This repo is the application layer only

Infrastructure lives in two separate repos:

- **[mediasuite-knowledge-base](https://github.com/roelandordelman/mediasuite-knowledge-base)** (local: `/Users/roeland.ordelman/Projects/mediasuite-knowledge-base`) — ChromaDB documentation index, Apache Jena Fuseki knowledge graph, ingestion and embedding pipelines.
- **[mediasuite-wiki-agent](https://github.com/roelandordelman/mediasuite-wiki-agent)** (local: `/Users/roeland.ordelman/Projects/mediasuite-wiki-agent`) — Beeld & Geluid Wiki harvest, Milvus index, RDF graph (wiki dataset in the same Fuseki instance), REST API on port 8002.

Do not add ingestion, embedding, or harvest code here.

## Stack

| Layer | Technology |
|---|---|
| Generation + query expansion | llama3.1:8b via Ollama (local) |
| Embeddings (docs + routing) | nomic-embed-text via Ollama (local) |
| Embeddings (wiki) | multilingual-e5-large-instruct (inside wiki agent) |
| Vector store (docs) | ChromaDB HTTP server — port 8001 |
| Vector store (wiki) | Milvus Lite — inside wiki agent |
| Knowledge graph | Apache Jena Fuseki — port 3030 |
| Wiki retrieval | mediasuite-wiki-agent REST API — port 8002 |
| Backend | FastAPI + uvicorn |
| Frontend | Vanilla JS widget, no framework |

## Project structure

```
api/
  main.py            — FastAPI app (POST /ask, conversation history)
  rag.py             — RAG pipeline: all three paths run in parallel → generate
  router.py          — Structural path: SPARQL execution + result formatting
  query_index.py     — QueryIndex singleton: trigger embeddings, named-entity detection
  sparql_queries.py  — Named SPARQL query catalogue (11 templates) + run_query()
  wiki_client.py     — Wiki path: HTTP client for the wiki-agent REST API
widget/
  chatbot.js         — Floating chat widget (single <script> embed)
  chatbot.html       — Standalone test page
evaluate/
  test_questions.yaml    — Eval questions with category, expected_urls/key_terms, annotated flag
  eval_retrieval.py      — Narrative eval: URL presence in top-k retrieved chunks
  eval_router.py         — Structural eval: key-term scoring on generated answers
config.yaml          — All connection config + entity/tool/collection mappings
debug_rag.py         — Full pipeline debug CLI (queries, chunks, context, answer)
query_debug.py       — Retrieval-only debug CLI (chunks + scores, no generation)
logs/questions.log   — Rotating log of questions asked (written by rag.py)
```

## Three-path retrieval architecture

All three paths run for every question. The LLM is used only for query expansion and answer generation — never for routing.

### 1. Structural path

Answers precise questions about Media Suite tools, collections, and workflows.

1. `QueryIndex.select()` — embeds the question, cosine similarity against pre-embedded trigger questions per named query; selects queries above threshold (0.60). Deterministic.
2. For parametric queries, fills URI slots by embedding similarity against known entity names from `config.yaml` (`tool_entities`, `collection_entities`).
3. SPARQL runs against Fuseki (`mediasuite` dataset); rows formatted as `[Knowledge graph facts]` context.
4. Entity URIs also filter ChromaDB for supporting documentation chunks.

All named queries live in `api/sparql_queries.py`. Every query must include `FROM <https://mediasuite.clariah.nl/graph>` — data is not in the default graph.

### 2. Narrative path

Answers how-to and explanatory questions.

1. `_expand_query()` — llama3.1:8b generates 3 alternative phrasings.
2. `_retrieve()` — embeds all variants, semantic search in ChromaDB (top_k × 6 candidates).
3. Priority slots: FAQ/Help/How-to chunks get 2 reserved slots so tutorial volume can't crowd them out.
4. Dedup by title+section, then by URL.
5. **CRAG gate** — if structural returned nothing and best narrative L2 distance > `CRAG_RETRIEVAL_THRESHOLD` (0.75), `_reformulate_query()` asks the LLM to rephrase with different vocabulary and retries once; results merged best-score-per-URL. `crag_triggered` flag in debug output.

### 3. Wiki path

Provides biographical, production, and genre background from the Beeld & Geluid Wiki for questions about Dutch media history persons or topics.

- `wiki_client.py` — POSTs to `http://localhost:8002/ask`; the wiki agent runs its own dual-path retrieval (keyword-routed SPARQL + Milvus semantic search) and returns a merged context block.
- Results are filtered at similarity threshold 0.70 inside the wiki agent before being returned.
- If the wiki API is not running, this path is silently skipped — the chatbot degrades gracefully.

### Before retrieval (follow-up questions)

When conversation history is present and the question contains follow-up signals (pronouns, "the tool", "the first", etc.), `_rewrite_as_standalone()` asks the LLM to rewrite the question as a self-contained query using the last 3 turns. The rewritten `retrieval_question` is used for embedding and routing; the original `question` is used for generation so the answer reads naturally.

## Answer generation and source attribution

Context blocks are labelled so the LLM knows which source to attribute:

- `[Knowledge graph facts]` — structural path results
- `[Gestructureerde data]` — wiki SPARQL results (structured lists)
- `[Beeld & Geluid Wiki — achtergrond]` — wiki semantic results (biographical/historical background)
- Unlabelled chunks — narrative path ChromaDB results

The system prompt (`_USER_PROMPT_TEMPLATE` in `rag.py`) instructs the LLM to:
- Open answers by naming the source: "According to the Beeld & Geluid Wiki, …" / "According to the Media Suite knowledge graph, …" / "According to the Media Suite documentation, …"
- Present `[Gestructureerde data]` blocks as complete lists without summarising or selecting examples
- Treat `[Knowledge graph facts]` as authoritative for tool/collection/workflow questions
- End answers with source URLs from the context chunks used
- If context contains no clear answer, respond only with: "I don't have information about that in the Media Suite documentation."

## Chunk schema (ChromaDB metadata fields)

- `url` — always present; must appear in answers as direct source links
- `content_type` — `How-to Guide` / `FAQ` / `Tool Tutorial` / `Subject Tutorial` / `Help / Documentation` / `Glossary` / etc.; used for priority-slot retrieval
- `entity_uri` — `ms:` URI of the primary tool or collection (e.g. `https://mediasuite.clariah.nl/vocab#AnnotationTool`); used by structural path for entity-filtered ChromaDB lookups; empty string on general content
- `title`, `section` — for dedup and context formatting
- `tags`, `categories`, `tools_mentioned`, `collections_mentioned` — JSON-encoded lists stored as strings; decoded by `_decode_meta()`

## Config structure

`config.yaml` top-level sections:
- `knowledge_base` — ChromaDB host/port/collection
- `knowledge_graph` — Fuseki URL/dataset + `tool_entities` (tool name → entity URI + tadirah activities) + `collection_entities` (collection name → entity URI)
- `wiki_api` — URL (default `http://localhost:8002`), `top_k`, `min_score`; remove the key to disable the wiki path entirely

`tool_entities` and `collection_entities` are sourced from `mediasuite-knowledge-base/config.yaml` and must be kept in sync manually when the KB adds tools or collections.

## Evaluation

```bash
python evaluate/eval_retrieval.py              # narrative: URL presence in top-k
python evaluate/eval_retrieval.py --verbose    # show retrieved vs expected URLs on failure
python evaluate/eval_router.py                 # structural: key-term scoring in generated answer
python evaluate/eval_router.py --debug         # show route, SPARQL queries, context per question
python evaluate/eval_router.py --verbose       # show full answers and missing terms on failure
python evaluate/eval_wiki.py                   # wiki path: attribution + key-term scoring
python evaluate/eval_wiki.py --verbose         # show full answers and missing terms on failure
python evaluate/eval_wiki.py --debug           # show SPARQL queries selected and crag_triggered
```

Baselines: 14/14 narrative (100%), 25–26/26 structural (routing is deterministic; ~1 failure per run is LLM non-determinism in generation, not a routing problem), 7/7 wiki (100%). Questions with `annotated: false` are shown as `[PENDING]` with the chatbot's actual output.

`eval_retrieval.py` skips questions with `category: structural`. `eval_router.py` only runs `category: structural` questions.

## Debugging

```bash
python debug_rag.py "your question here"
python debug_rag.py "your question here" --no-generate   # retrieval only
python query_debug.py "your question here" --top-k 10
```

`debug_rag.py` shows: expanded query variants, SPARQL queries selected, retrieved chunks with scores, wiki path result, the exact context string passed to the LLM, and the generated answer.

## Key conventions

- All connection config in `config.yaml` only — never hardcode hosts, ports, or URIs in `api/`.
- Always preserve `url` metadata in answers so researchers get direct source links.
- Answers must be grounded strictly in retrieved context, not general knowledge.
- Keep the widget embeddable via a single `<script>` tag with a `data-api-url` attribute.
- Every SPARQL query must include `FROM <https://mediasuite.clariah.nl/graph>` — data is not in the default graph.
- The wiki path is optional infrastructure — code must handle its absence gracefully.

## Roadmap
The shared project roadmap is at `../mediasuite-knowledge-base/docs/roadmap.md`.
Before starting significant work, check current priorities there.
