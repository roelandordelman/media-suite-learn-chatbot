"""
Media Suite Knowledge Base Ingestion
=====================================
Converts beeldengeluid/mediasuite-website (Jekyll/Markdown) into a chunked
JSON file ready for embedding + RAG.

Usage:
    python ingest_mediasuite.py \
        --repo ./mediasuite-website \
        --output ./mediasuite_knowledge_base.json

Output: a JSON file with a list of chunk objects, each containing:
    {
        "id":           "unique chunk identifier",
        "title":        "page title from front matter",
        "section":      "heading the chunk falls under",
        "collection":   "source collection (_howtos, _faq, etc.)",
        "content_type": "human-readable type (How-to, FAQ, Tutorial, ...)",
        "url":          "live URL on mediasuite.clariah.nl",
        "tags":         ["any", "front matter", "tags"],
        "author":       "author if present",
        "categories":   ["subject categories if present"],
        "text":         "the chunk text",
        "char_count":   250
    }

Requirements:
    pip install python-frontmatter
"""

import argparse
import json
import re
import sys
from pathlib import Path

import frontmatter  # python-frontmatter

# ---------------------------------------------------------------------------
# Collection configuration
# ---------------------------------------------------------------------------
# Maps each Jekyll collection folder to:
#   content_type  — label shown in chatbot answers
#   url_prefix    — how the path maps to a live URL
#   include       — True = include in knowledge base

COLLECTIONS = {
    "_help": {
        "content_type": "Help / Documentation",
        "url_prefix": "https://mediasuite.clariah.nl/documentation",
        "include": True,
    },
    "_howtos": {
        "content_type": "How-to Guide",
        "url_prefix": "https://mediasuite.clariah.nl/documentation/howtos",
        "include": True,
    },
    "_faq": {
        "content_type": "FAQ",
        "url_prefix": "https://mediasuite.clariah.nl/documentation/faq",
        "include": True,
    },
    "_glossary": {
        "content_type": "Glossary",
        "url_prefix": "https://mediasuite.clariah.nl/documentation/glossary",
        "include": True,
    },
    "_learn_main": {
        "content_type": "Learn (General)",
        "url_prefix": "https://mediasuite.clariah.nl/learn",
        "include": True,
    },
    "_learn_tutorials_tool": {
        "content_type": "Tool Tutorial",
        "url_prefix": "https://mediasuite.clariah.nl/learn/tool-tutorials",
        "include": True,
    },
    "_learn_tutorials_subject": {
        "content_type": "Subject Tutorial",
        "url_prefix": "https://mediasuite.clariah.nl/learn/subject-tutorials",
        "include": True,
    },
    "_learn_tool_criticism": {
        "content_type": "Tool Criticism",
        "url_prefix": "https://mediasuite.clariah.nl/learn/tool-criticism",
        "include": True,
    },
    "_learn_example_projects": {
        "content_type": "Example Project",
        "url_prefix": "https://mediasuite.clariah.nl/learn/example-projects",
        "include": True,
    },
    "_labo-help": {
        "content_type": "Labo Help",
        "url_prefix": "https://mediasuite.clariah.nl/labo/documentation",
        "include": True,
    },
}

# Chunking settings
CHUNK_TARGET_CHARS = 800    # aim for chunks around this size
CHUNK_OVERLAP_CHARS = 150   # overlap between consecutive chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slug_from_filename(filename: str) -> str:
    """Convert filename (without extension) to URL slug."""
    return filename.replace(".markdown", "").replace(".md", "")


def build_url(collection_key: str, slug: str) -> str:
    prefix = COLLECTIONS[collection_key]["url_prefix"]
    return f"{prefix}/{slug}"


def clean_markdown(text: str) -> str:
    """Strip Markdown syntax for cleaner text chunks."""
    # Remove image tags (they add nothing to text)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    # Convert links to just their display text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove bold/italic markers
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,3}([^_]+)_{1,3}', r'\1', text)
    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def split_into_sections(body: str) -> list[tuple[str, str]]:
    """
    Split markdown body into (heading, text) pairs.
    Returns a list of (section_heading, section_text).
    The first section may have an empty heading (content before first heading).
    """
    # Match h2 and h3 headings
    pattern = re.compile(r'^(#{2,3})\s+(.+)$', re.MULTILINE)
    sections = []
    last_end = 0
    current_heading = ""

    for match in pattern.finditer(body):
        # Content before this heading
        chunk_text = body[last_end:match.start()].strip()
        if chunk_text:
            sections.append((current_heading, chunk_text))
        current_heading = match.group(2).strip()
        last_end = match.end()

    # Remaining content after last heading
    remaining = body[last_end:].strip()
    if remaining:
        sections.append((current_heading, remaining))

    return sections


def chunk_text(text: str, target: int = CHUNK_TARGET_CHARS,
               overlap: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    """
    Split text into overlapping chunks of ~target characters,
    trying to break on paragraph or sentence boundaries.
    """
    if len(text) <= target:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + target, len(text))

        if end < len(text):
            # Try to find a paragraph break near end
            para_break = text.rfind('\n\n', start, end)
            if para_break > start + target // 2:
                end = para_break
            else:
                # Try sentence break
                sent_break = max(
                    text.rfind('. ', start, end),
                    text.rfind('.\n', start, end),
                )
                if sent_break > start + target // 2:
                    end = sent_break + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start forward with overlap
        start = max(start + 1, end - overlap)

    return chunks


# ---------------------------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------------------------

def ingest_file(filepath: Path, collection_key: str) -> list[dict]:
    """Parse a single markdown file and return a list of chunk dicts."""
    try:
        post = frontmatter.load(filepath)
    except Exception as e:
        print(f"  WARNING: could not parse {filepath}: {e}", file=sys.stderr)
        return []

    slug = slug_from_filename(filepath.name)
    url = build_url(collection_key, slug)
    conf = COLLECTIONS[collection_key]

    # Metadata from front matter
    title = str(post.get("title", slug))
    author = str(post.get("author", ""))
    tags = post.get("tags", [])
    if isinstance(tags, str):
        tags = [tags]
    categories = post.get("categories", [])
    if isinstance(categories, str):
        categories = [categories]
    introduction = str(post.get("introduction", ""))

    body = post.content or ""

    # Prepend introduction if present (often only in front matter)
    if introduction and introduction not in body:
        body = introduction + "\n\n" + body

    body_clean = clean_markdown(body)

    if not body_clean.strip():
        return []  # Skip empty files

    # Split into sections, then chunk each section
    sections = split_into_sections(body_clean)
    if not sections:
        sections = [("", body_clean)]

    records = []
    chunk_idx = 0

    for section_heading, section_text in sections:
        # Add title + section heading as context prefix for each chunk
        context_prefix = title
        if section_heading:
            context_prefix += f" — {section_heading}"

        text_chunks = chunk_text(section_text)

        for chunk_text_item in text_chunks:
            # Prepend context so each chunk is self-contained
            full_text = f"[{context_prefix}]\n{chunk_text_item}"

            record = {
                "id": f"{collection_key}/{slug}/{chunk_idx}",
                "title": title,
                "section": section_heading,
                "collection": collection_key,
                "content_type": conf["content_type"],
                "url": url,
                "tags": tags,
                "author": author,
                "categories": categories,
                "text": full_text,
                "char_count": len(full_text),
            }
            records.append(record)
            chunk_idx += 1

    return records


def ingest_repo(repo_path: Path) -> list[dict]:
    all_chunks = []
    stats = {}

    for collection_key, conf in COLLECTIONS.items():
        if not conf["include"]:
            continue

        coll_dir = repo_path / collection_key
        if not coll_dir.exists():
            print(f"  Skipping {collection_key} (not found)", file=sys.stderr)
            continue

        files = list(coll_dir.glob("*.markdown")) + list(coll_dir.glob("*.md"))
        file_chunks = []

        for f in sorted(files):
            chunks = ingest_file(f, collection_key)
            file_chunks.extend(chunks)
            if chunks:
                print(f"  {collection_key}/{f.name}: {len(chunks)} chunks")

        all_chunks.extend(file_chunks)
        stats[collection_key] = {"files": len(files), "chunks": len(file_chunks)}

    return all_chunks, stats


def main():
    parser = argparse.ArgumentParser(description="Ingest mediasuite-website into RAG-ready JSON")
    parser.add_argument("--repo", default="./mediasuite-website",
                        help="Path to cloned mediasuite-website repo")
    parser.add_argument("--output", default="./mediasuite_knowledge_base.json",
                        help="Output JSON file path")
    args = parser.parse_args()

    repo_path = Path(args.repo)
    if not repo_path.exists():
        print(f"ERROR: repo not found at {repo_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Ingesting from: {repo_path.resolve()}")
    print("-" * 60)

    chunks, stats = ingest_repo(repo_path)

    print("-" * 60)
    print(f"\nSummary:")
    total_files = sum(s["files"] for s in stats.values())
    total_chunks = sum(s["chunks"] for s in stats.values())
    for coll, s in stats.items():
        print(f"  {coll:35s} {s['files']:3d} files → {s['chunks']:4d} chunks")
    print(f"  {'TOTAL':35s} {total_files:3d} files → {total_chunks:4d} chunks")

    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"\nWritten to: {output_path.resolve()}")
    print(f"File size:  {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
