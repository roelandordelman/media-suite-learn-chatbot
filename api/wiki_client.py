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
import re

import requests

logger = logging.getLogger(__name__)

# ── Wiki relevance signals ────────────────────────────────────────────────────
# Mirrors the detection logic in mediasuite-wiki-agent/api/wiki_router.py.
# Keep in sync when new signals are added to the router.

_YEAR_RE = re.compile(r'\b(1[5-9]\d{2}|20[0-9]{2})\b')

_DECADE_TERMS = {
    # Dutch
    "twintig", "dertig", "veertig", "vijftig", "zestig",
    "zeventig", "tachtig", "negentig",
    # English
    "1920s", "1930s", "1940s", "1950s", "1960s", "1970s",
    "1980s", "1990s", "2000s",
    "20s", "30s", "40s", "50s", "60s", "70s", "80s", "90s",
}

_BIOGRAPHICAL_PHRASES = {
    "wie was", "wie is", "wie zijn", "wie waren",
    "who was", "who is", "who are", "who were",
    "vertel over", "vertel me over",
    "tell me about", "what do you know about",
    "wat weet je over",
    "actief", "active", "werkzaam", "werkten",
}

_FUNCTION_TERMS = {
    "presentator", "presentatrice", "presenter", "host",
    "acteur", "actrice", "actor",
    "regisseur", "director",
    "journalist", "verslaggever",
    "zanger", "zangeres", "singer",
    "schrijver", "writer",
    "cabaretier", "comedian",
    "producer",
}

_BROADCASTER_SIGNALS = {
    "omroep", "omroepen", "broadcaster", "broadcasters", "broadcasting",
    "zenders", "stations",
}

_KNOWN_BROADCASTERS = {
    "vpro", "vara", "nos", "avro", "kro", "ncrv", "tros", "eo",
    "veronica", "bnn", "nps", "rvu", "teleac", "max", "wnl",
    "human", "at5", "rtl", "sbs", "net5", "nts",
}

_EXPLICIT_WIKI_TERMS = {
    "beeld en geluid", "beeld & geluid", "beeldengeluid",
    "dutch media", "nederlandse media", "nederlandse televisie",
    "nederlandse film", "dutch television", "dutch film",
}


def _is_wiki_relevant(question: str) -> bool:
    """
    Return True if the question is likely about Dutch media history.

    Checks the same signals as the wiki router so the wiki API is only
    called for questions it can plausibly answer. Questions with no
    matching signals skip the HTTP call entirely.
    """
    q = question.lower()

    if _YEAR_RE.search(q):
        return True
    if any(t in q for t in _DECADE_TERMS):
        return True
    if any(p in q for p in _BIOGRAPHICAL_PHRASES):
        return True
    if any(f in q for f in _FUNCTION_TERMS):
        return True
    if any(s in q for s in _BROADCASTER_SIGNALS):
        return True
    if any(b in q for b in _KNOWN_BROADCASTERS):
        return True
    if any(t in q for t in _EXPLICIT_WIKI_TERMS):
        return True

    return False


def retrieve_wiki(
    base_url: str,
    question: str,
    top_k: int = 3,
    min_score: float = 0.70,
) -> dict:
    """
    Call wiki /ask and return {"context": str, "sources": list, "found": bool}.
    Returns empty result on any failure or when the question is not wiki-relevant.
    """
    empty = {"context": "", "sources": [], "found": False}

    if not _is_wiki_relevant(question):
        logger.debug("Skipping wiki — no Dutch media signals in question: %r", question)
        return empty

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
