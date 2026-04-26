# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

An "Ask Media Suite" RAG chatbot for researchers using the CLARIAH Media Suite. Researchers ask questions in natural language and get answers grounded in the actual Help, How-to, FAQ, Tutorial and Glossary content, with direct links back to the relevant pages on mediasuite.clariah.nl.

The chatbot will be embedded on the Media Suite Community site at https://roelandordelman.github.io/media-suite-community/

## This repo is the application layer only

All ingestion, embedding, and vector store infrastructure lives in the separate [mediasuite-knowledge-base](https://github.com/roelandordelman/mediasuite-knowledge-base) repository. This repo contains only:

- `api/` — FastAPI backend that queries the vector store
- `widget/` — embeddable vanilla JS chat widget
- `config.yaml` — points to the ChromaDB path in mediasuite-knowledge-base

Do not add ingestion or embedding code here.

## Stack

- **Generation**: mistral via Ollama (local)
- **Vector store**: ChromaDB, built and maintained in mediasuite-knowledge-base
- **Backend**: FastAPI + uvicorn
- **Frontend**: vanilla JS widget, no framework

## Project structure

```
api/        — FastAPI app (main.py, rag.py)
widget/     — chat widget (chatbot.js, chatbot.html)
config.yaml — knowledge base path configuration
```

## Key conventions

- Always preserve the `url` field from chunks in answers so researchers get direct links back to the relevant Media Suite pages.
- Answers must be grounded strictly in retrieved chunks, not general knowledge.
- Keep the widget embeddable via a single `<script>` tag.
- The ChromaDB path is defined only in `config.yaml` — never hardcode it in `api/rag.py`.
