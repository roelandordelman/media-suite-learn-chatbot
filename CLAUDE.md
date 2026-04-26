# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

An "Ask Media Suite" RAG chatbot for researchers using the CLARIAH Media Suite. Researchers ask questions in natural language and get answers grounded in the actual Help, How-to, FAQ, Tutorial and Glossary content, with direct links back to the relevant pages on mediasuite.clariah.nl.

The chatbot will be embedded on the Media Suite Community site at https://roelandordelman.github.io/media-suite-community/

## Stack

- **Embeddings**: nomic-embed-text via Ollama (local)
- **Generation**: mistral via Ollama (local)
- **Vector store**: ChromaDB (local, stored in `embed/chroma_db/`)
- **Backend**: FastAPI + uvicorn
- **Frontend**: vanilla JS widget, no framework

## Project structure

```
ingest/     — ingestion script + knowledge base JSON
embed/      — build_index.py + chroma_db/ vector store
api/        — FastAPI app (main.py, rag.py)
widget/     — chat widget (chatbot.js, chatbot.html)
```

## Key conventions

- Always preserve the `url` field from chunks in answers so researchers get direct links back to the relevant Media Suite pages.
- Answers must be grounded strictly in retrieved chunks, not general knowledge.
- Keep the widget embeddable via a single `<script>` tag.

## Ingestion pipeline

Content source: `https://github.com/beeldengeluid/mediasuite-website` (public Jekyll/Siteleaf repo), already ingested into `ingest/mediasuite_knowledge_base.json` — 10,719 deduplicated chunks with `title`, `section`, `content_type`, `url`, and `text` fields.

To regenerate the knowledge base:

```bash
# Clone the source documentation repo
git clone https://github.com/beeldengeluid/mediasuite-website ./mediasuite-website

# Install dependencies
pip install python-frontmatter

# Run ingestion
python ingest/ingest_mediasuite.py --repo ./mediasuite-website --output ./ingest/mediasuite_knowledge_base.json
```

The ingestion pipeline in `ingest/ingest_mediasuite.py`:

1. **Collection config** (`COLLECTIONS` dict) — maps each Jekyll `_collection` directory to metadata: `content_type`, `url_prefix`, and an `include` flag.
2. **Markdown cleaning** — strips Markdown syntax while preserving semantic content.
3. **Smart chunking** — overlapping chunks (~800 chars target, 150 char overlap) split at paragraph/sentence boundaries.
4. **Chunk assembly** — each chunk gets a context prefix with page title and section heading so it is self-contained for retrieval.

Key constants at the top of the ingestion script:

```python
CHUNK_TARGET_CHARS = 800
CHUNK_OVERLAP_CHARS = 150
```

`mediasuite_knowledge_base.json` and all `embed/` artifacts (`chroma_db/`, `*.index`, `*.pkl`) are gitignored.
