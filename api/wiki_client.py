"""
HTTP client for the Beeld & Geluid Wiki REST API.

Wraps the /search endpoint exposed by mediasuite-wiki-agent/api/serve.py.
Returns empty results when the wiki API is unreachable so the chatbot degrades
gracefully when the wiki service is down.

Only results whose cosine similarity score meets the min_score threshold are
returned — low-scoring hits indicate no relevant wiki article was found and
should not be added to the answer context.
"""

from __future__ import annotations

import logging

import requests

logger = logging.getLogger(__name__)


def retrieve_wiki(
    base_url: str,
    query: str,
    limit: int = 3,
    min_score: float = 0.70,
) -> list[dict]:
    """Call wiki /search and return results above min_score."""
    try:
        resp = requests.post(
            f"{base_url.rstrip('/')}/search",
            json={"query": query, "limit": limit},
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json()
        return [r for r in results if r.get("score", 0) >= min_score]
    except requests.RequestException:
        logger.warning("Wiki API unavailable at %s — skipping wiki retrieval", base_url)
        return []
    except Exception:
        logger.warning("Wiki retrieval failed", exc_info=True)
        return []


def format_wiki_context(results: list[dict]) -> str:
    """Format wiki search results as a context block for the answer prompt."""
    if not results:
        return ""
    lines = ["[Achtergrond — Beeld & Geluid Wiki (wiki.beeldengeluid.nl)]"]
    for r in results:
        title = r.get("title", "?")
        url = r.get("url", "")
        excerpt = r.get("excerpt", "")
        last_edited = r.get("last_edited", "")
        stale = r.get("staleness_warning", "")
        lines.append(f"\n## {title}")
        if excerpt:
            lines.append(excerpt)
        if stale:
            lines.append(f"Let op: {stale}")
        if url:
            date = last_edited[:10] if last_edited else "?"
            lines.append(f"Bron: {url} (bijgewerkt: {date})")
    return "\n".join(lines)
