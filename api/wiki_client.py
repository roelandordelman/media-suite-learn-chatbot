"""
HTTP client for the Beeld & Geluid Wiki REST API.

Calls the /ask endpoint, which runs dual-path retrieval (SPARQL + semantic)
inside the wiki agent and returns a single merged context block. The chatbot
passes this context to its LLM alongside documentation and knowledge graph
context — no wiki-specific routing logic lives here.

Returns empty context when the wiki API is unreachable, so the chatbot
degrades gracefully when the wiki service is down.
"""

from __future__ import annotations

import logging

import requests

logger = logging.getLogger(__name__)


def retrieve_wiki(
    base_url: str,
    question: str,
    top_k: int = 3,
    min_score: float = 0.70,
) -> dict:
    """
    Call wiki /ask and return {"context": str, "sources": list, "found": bool}.
    Returns empty result on any failure.
    """
    empty = {"context": "", "sources": [], "found": False}
    try:
        resp = requests.post(
            f"{base_url.rstrip('/')}/ask",
            json={"question": question, "top_k": top_k, "min_score": min_score},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        logger.warning("Wiki API unavailable at %s — skipping wiki retrieval", base_url)
        return empty
    except Exception:
        logger.warning("Wiki retrieval failed", exc_info=True)
        return empty
