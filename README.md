# Ask Media Suite

A RAG chatbot for researchers using the [CLARIAH Media Suite](https://mediasuite.clariah.nl). Ask questions in natural language and get answers grounded in the official Help, How-to, FAQ, Tutorial and Glossary content, with direct links back to the relevant pages.

The widget is intended to be embedded on the [Media Suite Community site](https://roelandordelman.github.io/media-suite-community/).

## Stack

| Layer | Technology |
|---|---|
| Embeddings | nomic-embed-text via Ollama (local) |
| Generation | mistral via Ollama (local) |
| Vector store | ChromaDB (local) |
| Backend | FastAPI + uvicorn |
| Frontend | Vanilla JS widget |

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Install Ollama and pull models**

Download from [ollama.com/download](https://ollama.com/download), then:
```bash
ollama pull nomic-embed-text
ollama pull mistral
```

**3. Build the vector index**
```bash
python3 embed/build_index.py
```

This embeds all 10,719 chunks from `ingest/mediasuite_knowledge_base.json` into ChromaDB. Takes a few minutes on first run; subsequent runs skip already-indexed chunks.

**4. Start the API**
```bash
uvicorn api.main:app --reload
```

The API is available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

**5. Test the widget**

Open `widget/chatbot.html` in a browser.

## Usage

**Ask a question via curl:**
```bash
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Who can access the Media Suite?"}'
```

**Embed the widget on any page:**
```html
<script src="chatbot.js" data-api-url="https://your-api-url"></script>
```

## Project structure

```
ingest/     — ingestion script + knowledge base JSON (10,719 chunks)
embed/      — build_index.py, query_debug.py, chroma_db/ vector store
api/        — FastAPI app (main.py, rag.py)
widget/     — chat widget (chatbot.js, chatbot.html)
```

## Debugging retrieval

```bash
python3 embed/query_debug.py "your question here"
python3 embed/query_debug.py "your question here" --top-k 10
```

Shows expanded query variants, retrieved chunks with similarity scores, content type, and source URLs — useful for diagnosing why a question isn't finding the right content.

## Regenerating the knowledge base

The knowledge base is built from [beeldengeluid/mediasuite-website](https://github.com/beeldengeluid/mediasuite-website) (the Jekyll source of the Media Suite documentation):

```bash
git clone https://github.com/beeldengeluid/mediasuite-website ./mediasuite-website
python3 ingest/ingest_mediasuite.py --repo ./mediasuite-website --output ./ingest/mediasuite_knowledge_base.json
```
