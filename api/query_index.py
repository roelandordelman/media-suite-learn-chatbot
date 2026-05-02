"""
Embedding-based SPARQL query index.

Maps user questions to named SPARQL query templates using cosine similarity
against pre-computed trigger question embeddings. Fully deterministic — no
LLM involved in query selection or parameter filling.

The index is built once on first use (lazy initialisation) and cached for
the lifetime of the process. Build time: ~5s (embedding ~100 trigger questions
plus entity names). Subsequent query selections take one embed call (~50ms).

See README — Design decisions for the rationale.
"""
from __future__ import annotations

import numpy as np
import ollama

from api.sparql_queries import GRAPH, QUERIES, run_query, tadirah_uri

# ---------------------------------------------------------------------------
# Trigger questions per named query
# Representative questions each query answers. Diversity in phrasing matters
# more than quantity — aim for the full range of ways a researcher might ask.
# ---------------------------------------------------------------------------

QUERY_TRIGGERS: dict[str, list[str]] = {
    "all_tools": [
        "What tools does the Media Suite have?",
        "List all Media Suite tools",
        "What tools are available in the Media Suite?",
        "Give me an overview of Media Suite tools",
        "Which tools can I use in the Media Suite?",
    ],
    "tools_with_status": [
        "Which tools are experimental or not yet released?",
        "Which tools are still in beta or might change?",
        "Which tools are outdated or unstable?",
        "Which tools have known limitations I should be aware of?",
        "Are there tools I should be cautious about relying on for long-term research?",
        "Which tools are planned but not yet available?",
    ],
    "tools_by_activity": [
        "What tools support annotation?",
        "Which tools can I use for searching collections?",
        "What tools are available for browsing?",
        "Which tools support data visualisation?",
        "What tools exist for automatic speech recognition?",
        "Which tools support comparing search results?",
        "What tools are there for segmenting media into fragments?",
        "Which tools help with building a corpus?",
    ],
    "services_by_tool": [
        "What backend service powers the Similarity Tool?",
        "Which infrastructure services does the Media Suite use?",
        "What service does the FactRank tool use?",
        "Which backend services are deployed by Media Suite tools?",
        "What powers the computer vision features?",
    ],
    "collections_by_access": [
        "Which collections have access restrictions?",
        "What are the access rights for Media Suite collections?",
        "Which collections have a Creative Commons license?",
        "Which collections have a CC0 license?",
        "Which collections require institutional affiliation?",
        "Which collections are restricted to university researchers?",
        "Can I use Radio Oranje data commercially?",
        "Which collections can I reuse freely in my research publications?",
        "What license does the Sound and Vision collection have?",
    ],
    "open_collections": [
        "Which collections can I access without logging in?",
        "What collections are publicly available without authentication?",
        "What can I access without a SURFconext account?",
        "Which collections are open access?",
        "What can I do in the Media Suite without a university login?",
        "Which collections are freely accessible without an account?",
        "What can I do in the Media Suite without a login?",
    ],
    "restricted_collections": [
        "Which collections can I use only as a researcher working at a university?",
        "Which collections require institutional affiliation or a SURFconext login?",
        "Which collections have restricted access and require special permission?",
        "Which collections are not publicly available without login?",
        "Which collections require a university account to access?",
        "Which collections are restricted to researchers with institutional access?",
    ],
    "workflows_by_tool": [
        "Which workflows use the Search Tool?",
        "What workflows involve the Annotation Tool?",
        "Which research workflows use the Workspace?",
        "What workflows is the Similarity Tool part of?",
        "Which workflows use the Compare Tool as an instrument?",
    ],
    "all_workflows": [
        "What research workflows does the Media Suite support?",
        "Which workflows are aspirational or not yet supported?",
        "What future workflows are planned?",
        "Which workflows are not yet possible?",
        "What research workflows exist?",
        "Which workflows are fully supported?",
        "What workflows are partially supported?",
        "Can I request enrichment of Media Suite content on demand?",
        "Is researcher-triggered enrichment possible?",
        "Are there workflows for enriching collections on demand?",
    ],
    "workflow_steps": [
        "What are the steps of the SANE workflow in order?",
        "Walk me through the gender representation workflow",
        "What steps are involved in the ASR transcript research workflow?",
        "How does a Media Suite workflow work step by step?",
        "What steps do I need to follow to use SANE?",
        "What is the order of steps in the cross-collection comparison workflow?",
        "Can you describe the steps involved in a specific workflow?",
    ],
    "entity_description": [
        "What is the Annotation Tool?",
        "Describe the Similarity Tool and what it does",
        "What does the FactRank Tool do?",
        "Tell me about the Radio Oranje collection",
        "What is the Compare Tool used for?",
        "What was the Similarity Tool built for?",
        "What problem does the FactRank tool solve?",
        "What is the Fragment Cutter?",
        "Can I use Radio Oranje without restrictions?",
        "Is Radio Oranje openly available?",
    ],
}

# ---------------------------------------------------------------------------
# Workflow aliases — short labels for workflows whose graph schema:name is too
# long or descriptive to embed close to concise user questions.
# Key: URI local name. Value: one or more aliases to embed as additional labels.
# ---------------------------------------------------------------------------

WORKFLOW_ALIASES: dict[str, list[str]] = {
    "RestrictedDataSANEWorkflow": [
        "how to use SANE step-by-step: restricted data research workflow",
        "steps involved in using SANE for research: corpus selection, export, analyse, deposit",
    ],
    "SANEAccessSubWorkflow": [
        "administrative approval process for SANE: data access request, legal agreement, account setup",
    ],
    "ASRTranscriptResearchWorkflow": [
        "ASR automatic speech recognition transcript research workflow",
    ],
    "GenderWorkflow": [
        "gender representation media analysis workflow",
    ],
    "CrossCollectionComparativeWorkflow": [
        "cross-collection comparison workflow",
    ],
}


# ---------------------------------------------------------------------------
# Activity labels → tadirah URI(s)
# Multiple URIs: query runs once per URI (e.g. annotation → audio + visual).
# Labels are used as embedding targets — descriptive phrases work better than
# bare URI local names.
# ---------------------------------------------------------------------------

ACTIVITY_LABELS: dict[str, list[str]] = {
    "searching and querying collections": [tadirah_uri("searching")],
    "annotation of audio recordings and spoken content": [tadirah_uri("audioAnnotation")],
    "annotation of images and visual content in video": [tadirah_uri("visualAnnotation")],
    "annotation of media content, both audio and visual": [
        tadirah_uri("audioAnnotation"),
        tadirah_uri("visualAnnotation"),
    ],
    "segmenting media into temporal or spatial fragments": [tadirah_uri("segmenting")],
    "browsing and exploring collections": [tadirah_uri("browsing")],
    "comparing search results or queries across collections": [tadirah_uri("comparing")],
    "collecting items into a research corpus": [tadirah_uri("collecting")],
    "automatic speech recognition and transcription": [tadirah_uri("speechRecognizing")],
    "data visualisation and statistical analysis": [tadirah_uri("dataVisualization")],
    "linked open data and knowledge graph exploration": [tadirah_uri("linkedOpenData")],
    "sampling items from search results": [tadirah_uri("Sampling")],
}


# ---------------------------------------------------------------------------
# QueryIndex
# ---------------------------------------------------------------------------

class QueryIndex:
    """
    Embeds trigger questions and entity names at build time.
    Selects SPARQL queries and fills URI parameters at query time using
    cosine similarity — no LLM involved.
    """

    def __init__(self) -> None:
        self._trigger_embs: dict[str, np.ndarray] = {}
        # key → list of (label, uri_or_list, embedding)
        self._entities: dict[str, list] = {}
        self._built = False

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, kg_cfg: dict, embed_model: str) -> None:
        """
        Embed all trigger questions and entity names.
        Called once on first use; safe to call again to rebuild after config change.
        """
        fuseki = kg_cfg.get("fuseki_url", "").rstrip("/")
        dataset = kg_cfg.get("dataset", "")
        endpoint = f"{fuseki}/{dataset}/sparql" if fuseki and dataset else ""
        user, pw = kg_cfg.get("admin_user"), kg_cfg.get("admin_password")
        auth = (user, pw) if user and pw else None

        # Trigger questions
        for qname, triggers in QUERY_TRIGGERS.items():
            result = ollama.embed(model=embed_model, input=triggers)
            self._trigger_embs[qname] = np.array(result["embeddings"])

        # Tool names
        self._embed_pairs(
            "tool",
            [(name, info["entity_uri"]) for name, info in kg_cfg.get("tool_entities", {}).items()],
            embed_model,
        )

        # Collection names
        self._embed_pairs(
            "collection",
            list(kg_cfg.get("collection_entities", {}).items()),
            embed_model,
        )

        # Workflow names — fetch live from Fuseki so they stay in sync with the graph
        if endpoint:
            try:
                rows = run_query(endpoint, QUERIES["all_workflows"].format(graph=GRAPH), auth=auth)
                pairs = [(r["name"], r["uri"]) for r in rows if "name" in r and "uri" in r]
                # Add short human-friendly aliases for workflows whose graph name is
                # too long/descriptive to embed close to concise user questions
                for name, uri in list(pairs):
                    local = uri.split("#")[-1] if "#" in uri else uri.split("/")[-1]
                    for alias in WORKFLOW_ALIASES.get(local, []):
                        pairs.append((alias, uri))
                self._embed_pairs("workflow", pairs, embed_model)
            except Exception:
                pass

        # Activity labels
        labels = list(ACTIVITY_LABELS.keys())
        result = ollama.embed(model=embed_model, input=labels)
        self._entities["activity"] = [
            (label, ACTIVITY_LABELS[label], np.array(emb))
            for label, emb in zip(labels, result["embeddings"])
        ]

        self._built = True

    def _embed_pairs(self, key: str, pairs: list, embed_model: str) -> None:
        if not pairs:
            return
        labels = [p[0] for p in pairs]
        uris = [p[1] for p in pairs]
        result = ollama.embed(model=embed_model, input=labels)
        self._entities[key] = [
            (label, uri, np.array(emb))
            for label, uri, emb in zip(labels, uris, result["embeddings"])
        ]

    # ------------------------------------------------------------------
    # Select
    # ------------------------------------------------------------------

    def select(
        self, question: str, embed_model: str, threshold: float = 0.60
    ) -> list[tuple]:
        """
        Return list of (query_name, params) for queries whose best trigger
        similarity exceeds threshold. Capped at 4 queries per question.
        """
        if not self._built:
            raise RuntimeError("QueryIndex not built — call build() first.")

        q_emb = np.array(ollama.embed(model=embed_model, input=[question])["embeddings"][0])
        q_norm = float(np.linalg.norm(q_emb))

        candidates: list[tuple] = []
        for qname, trigger_embs in self._trigger_embs.items():
            norms = np.linalg.norm(trigger_embs, axis=1)
            sims = (trigger_embs @ q_emb) / np.maximum(norms * q_norm, 1e-10)
            max_sim = float(np.nanmax(sims))
            if max_sim >= threshold:
                candidates.append((qname, max_sim))

        candidates.sort(key=lambda x: x[1], reverse=True)

        selections: list[tuple] = []
        for qname, _ in candidates[:4]:
            for params in self._fill_params(qname, q_emb, q_norm):
                selections.append((qname, params))

        return selections

    # ------------------------------------------------------------------
    # Parameter filling
    # ------------------------------------------------------------------

    def _fill_params(self, qname: str, q_emb: np.ndarray, q_norm: float) -> list[dict]:
        """Return list of param dicts for this query (multiple for multi-URI activities)."""
        if qname in (
            "all_tools", "tools_with_status", "services_by_tool",
            "open_collections", "restricted_collections",
            "collections_by_access", "all_workflows",
        ):
            return [{}]

        if qname == "tools_by_activity":
            acts = self._entities.get("activity", [])
            if not acts:
                return []
            _, uris, _ = self._closest(q_emb, q_norm, acts)
            return [{"activity_uri": uri} for uri in (uris if isinstance(uris, list) else [uris])]

        if qname == "workflow_steps":
            wfs = self._entities.get("workflow", [])
            if not wfs:
                return []
            _, uri, _ = self._closest(q_emb, q_norm, wfs)
            return [{"workflow_uri": uri}]

        if qname == "workflows_by_tool":
            tools = self._entities.get("tool", [])
            if not tools:
                return []
            _, uri, _ = self._closest(q_emb, q_norm, tools)
            return [{"tool_uri": uri}]

        if qname == "entity_description":
            entities = self._entities.get("tool", []) + self._entities.get("collection", [])
            if not entities:
                return []
            _, uri, _ = self._closest(q_emb, q_norm, entities)
            return [{"entity_uri": uri}]

        if qname == "activities_by_tool":
            tools = self._entities.get("tool", [])
            if not tools:
                return []
            _, uri, _ = self._closest(q_emb, q_norm, tools)
            return [{"tool_uri": uri}]

        return [{}]

    def _closest(self, q_emb: np.ndarray, q_norm: float, entities: list) -> tuple:
        """Return (label, uri, sim) for the entity most similar to q_emb."""
        best_sim = -1.0
        best: tuple = ("", "", -1.0)
        for label, uri, emb in entities:
            sim = float(np.dot(emb, q_emb) / max(float(np.linalg.norm(emb)) * q_norm, 1e-10))
            if sim > best_sim:
                best_sim = sim
                best = (label, uri, sim)
        return best


# ---------------------------------------------------------------------------
# Module-level singleton — built once on first use
# ---------------------------------------------------------------------------

_index: QueryIndex = QueryIndex()


def get_index() -> QueryIndex:
    return _index
