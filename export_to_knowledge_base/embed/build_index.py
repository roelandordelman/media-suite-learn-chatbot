"""
Load mediasuite_knowledge_base.json and build a ChromaDB vector index
using nomic-embed-text via Ollama.

Usage:
    python embed/build_index.py
"""

import json
import argparse
from pathlib import Path

import ollama
import chromadb

KNOWLEDGE_BASE = Path(__file__).parent.parent / "ingest" / "mediasuite_knowledge_base.json"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "mediasuite"
EMBED_MODEL = "nomic-embed-text"
BATCH_SIZE = 64


def build_index(knowledge_base: Path, chroma_dir: Path) -> None:
    print(f"Loading chunks from {knowledge_base} …")
    chunks = json.loads(knowledge_base.read_text())
    print(f"  {len(chunks):,} chunks loaded")

    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(COLLECTION_NAME)

    existing_ids = set(collection.get(include=[])["ids"])
    new_chunks = [c for c in chunks if c["id"] not in existing_ids]
    print(f"  {len(new_chunks):,} new chunks to embed (skipping {len(existing_ids):,} already indexed)")

    for i in range(0, len(new_chunks), BATCH_SIZE):
        batch = new_chunks[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        response = ollama.embed(model=EMBED_MODEL, input=texts)
        embeddings = response["embeddings"]

        collection.add(
            ids=[c["id"] for c in batch],
            embeddings=embeddings,
            documents=[c["text"] for c in batch],
            metadatas=[
                {
                    "title": c.get("title", ""),
                    "section": c.get("section", ""),
                    "content_type": c.get("content_type", ""),
                    "url": c.get("url", ""),
                }
                for c in batch
            ],
        )
        print(f"  Indexed batch {i // BATCH_SIZE + 1} / {-(-len(new_chunks) // BATCH_SIZE)}")

    print(f"Done. Index saved to {chroma_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--knowledge-base", type=Path, default=KNOWLEDGE_BASE)
    parser.add_argument("--chroma-dir", type=Path, default=CHROMA_DIR)
    args = parser.parse_args()
    build_index(args.knowledge_base, args.chroma_dir)
