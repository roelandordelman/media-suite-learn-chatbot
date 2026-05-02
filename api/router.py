"""
Retrieval router: selects named SPARQL queries using embedding similarity
and retrieves structured facts from the knowledge graph.

The structural path maps the user question to pre-written SPARQL query templates
using cosine similarity against trigger questions — no LLM involved in routing.

See api/query_index.py for the trigger question catalogue and QueryIndex class.
"""

import json

import ollama
import chromadb

from api.sparql_queries import GRAPH, QUERIES, run_query


_JSON_META_FIELDS = ("tags", "categories", "tools_mentioned", "collections_mentioned")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sparql_query_structural(
    question: str, kg_cfg: dict, embed_model: str
) -> tuple[str, list[str], list[tuple]]:
    """
    Select and run named SPARQL queries using embedding similarity.

    Builds the QueryIndex on first call (lazy), then uses cosine similarity
    against pre-embedded trigger questions to select queries — deterministic,
    no LLM involved.

    Returns (sparql_context, entity_uris, selections):
      sparql_context — formatted text of SPARQL results
      entity_uris    — ms: entity URIs from results (for ChromaDB entity filtering)
      selections     — list of (query_name, params) selected (for debug)

    Returns ("", [], []) when no queries exceed the similarity threshold.
    """
    from api.query_index import get_index

    index = get_index()
    if not index._built:
        index.build(kg_cfg, embed_model)

    selections = index.select(question, embed_model)
    if not selections:
        return "", [], []

    endpoint = _sparql_endpoint(kg_cfg)
    auth = _sparql_auth(kg_cfg)

    context_parts: list[str] = []
    entity_uris: set[str] = set()

    for query_name, params in selections:
        template = QUERIES.get(query_name)
        if not template:
            continue
        try:
            query = template.format(graph=GRAPH, **params)
            rows = run_query(endpoint, query, auth=auth)
            if rows:
                context_parts.append(_format_rows(query_name, rows, params))
                for row in rows:
                    for val in row.values():
                        if isinstance(val, str) and val.startswith("https://mediasuite.clariah.nl/vocab#"):
                            entity_uris.add(val)
                # entity_description subject URI is in params, not returned as a value
                if query_name == "entity_description" and "entity_uri" in params:
                    entity_uris.add(params["entity_uri"])
        except Exception:
            continue

    return "\n\n".join(context_parts), list(entity_uris), selections


def retrieve_by_entity_uris(
    question: str,
    entity_uris: list[str],
    collection: chromadb.Collection,
    embed_model: str,
    top_k: int,
) -> tuple[list, list, list]:
    """
    Retrieve ChromaDB chunks whose entity_uri is in entity_uris, ranked by
    semantic similarity. Returns empty if no matching chunks found.
    """
    if not entity_uris:
        return [], [], []

    embedding = ollama.embed(model=embed_model, input=[question])["embeddings"][0]

    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k * 4,
        where={"entity_uri": {"$in": entity_uris}},
        include=["documents", "metadatas", "distances"],
    )

    raw_docs = results["documents"][0]
    raw_metas = results["metadatas"][0]
    raw_dists = results["distances"][0]

    if not raw_docs:
        return [], [], []

    ranked = sorted(zip(raw_docs, raw_metas, raw_dists), key=lambda x: x[2])[:top_k]
    docs, metas, distances = zip(*ranked)

    decoded_metas = []
    for meta in metas:
        meta = dict(meta)
        for field in _JSON_META_FIELDS:
            if field in meta and isinstance(meta[field], str):
                try:
                    meta[field] = json.loads(meta[field])
                except Exception:
                    pass
        decoded_metas.append(meta)

    return list(docs), decoded_metas, list(distances)


# ---------------------------------------------------------------------------
# Result formatters — one per query type
# ---------------------------------------------------------------------------

def _format_rows(query_name: str, rows: list[dict], params: dict) -> str:
    formatters = {
        "all_tools":            _fmt_all_tools,
        "tools_with_status":    _fmt_tools_with_status,
        "tools_by_activity":    _fmt_tools_by_activity,
        "all_workflows":        _fmt_all_workflows,
        "workflows_by_tool":    _fmt_workflows_by_tool,
        "workflow_steps":       _fmt_workflow_steps,
        "open_collections":          _fmt_open_collections,
        "restricted_collections":    _fmt_restricted_collections,
        "collections_by_access":     _fmt_collections_by_access,
        "entity_description":   _fmt_entity_description,
        "services_by_tool":     _fmt_services_by_tool,
        "activities_by_tool":   _fmt_activities_by_tool,
    }
    fn = formatters.get(query_name, _fmt_generic)
    return fn(rows, params)


def _short(uri: str) -> str:
    return uri.split("#")[-1] if "#" in uri else uri.split("/")[-1]


def _fmt_all_tools(rows, params):
    # Group by tool URI, collect unique activities
    tools: dict[str, dict] = {}
    for r in rows:
        uri = r.get("uri", "")
        if uri not in tools:
            tools[uri] = {"label": r.get("label", ""), "description": r.get("description", ""), "activities": []}
        act = r.get("activity", "")
        if act:
            tools[uri]["activities"].append(_short(act))

    lines = ["# Media Suite tools"]
    for info in sorted(tools.values(), key=lambda x: x["label"]):
        line = f"- {info['label']}"
        if info["description"]:
            line += f": {info['description']}"
        if info["activities"]:
            line += f" (activities: {', '.join(sorted(set(info['activities'])))})"
        lines.append(line)
    return "\n".join(lines)


def _fmt_tools_with_status(rows, params):
    lines = ["# Media Suite tools with status"]
    for r in rows:
        label = r.get("label", "")
        desc = r.get("description", "")
        status = r.get("status", "")
        line = f"- {label}"
        if status:
            line += f" [{status}]"
        if desc:
            line += f": {desc}"
        lines.append(line)
    return "\n".join(lines)


def _fmt_tools_by_activity(rows, params):
    activity = _short(params.get("activity_uri", ""))
    lines = [f"# Tools supporting {activity}"]
    for r in rows:
        label = r.get("label", "")
        desc = r.get("description", "")
        status = r.get("status", "")
        line = f"- {label}"
        if desc:
            line += f": {desc}"
        if status:
            line += f" [status: {status}]"
        lines.append(line)
    return "\n".join(lines)


def _fmt_all_workflows(rows, params):
    lines = ["# Research workflows"]
    for r in rows:
        name = r.get("name", "")
        status = r.get("status", "")
        desc = r.get("description", "")
        line = f"- {name}"
        if status:
            line += f" ({status})"
        if desc:
            line += f": {desc[:200]}"
        lines.append(line)
    return "\n".join(lines)


def _fmt_workflows_by_tool(rows, params):
    tool = _short(params.get("tool_uri", ""))
    lines = [f"# Workflows using {tool}"]
    seen = set()
    for r in rows:
        name = r.get("wfName", "")
        status = r.get("status", "")
        if name not in seen:
            seen.add(name)
            line = f"- {name}"
            if status:
                line += f" ({status})"
            lines.append(line)
    return "\n".join(lines)


def _fmt_workflow_steps(rows, params):
    wf = _short(params.get("workflow_uri", ""))
    lines = [f"# Steps of {wf}"]
    for r in sorted(rows, key=lambda x: int(x.get("position", 0))):
        pos = r.get("position", "?")
        name = r.get("stepName", "")
        instrument = _short(r.get("instrument", ""))
        optional = r.get("optional", "")
        line = f"{pos}. {name}"
        if instrument:
            line += f" (tool: {instrument})"
        if optional == "true":
            line += " [optional]"
        lines.append(line)
    return "\n".join(lines)


def _fmt_open_collections(rows, params):
    lines = ["# Open-access collections (no login required)"]
    for r in rows:
        label = r.get("label", "")
        license_uri = r.get("license", "")
        line = f"- {label}"
        if license_uri:
            if "zero" in license_uri:
                line += " (CC0)"
            elif "publicdomain/mark" in license_uri:
                line += " (Public Domain)"
            else:
                line += f" (license: {license_uri})"
        lines.append(line)
    return "\n".join(lines)


def _fmt_restricted_collections(rows, params):
    lines = ["# Collections requiring institutional access (login required)"]
    for r in rows:
        label = r.get("label", "")
        conditions = r.get("conditions", "")
        line = f"- {label}"
        if conditions:
            line += f": {conditions}"
        lines.append(line)
    return "\n".join(lines)


def _fmt_collections_by_access(rows, params):
    lines = ["# Collections and access rights"]
    for r in rows:
        label = r.get("label", "")
        access = _short(r.get("accessRights", ""))
        conditions = r.get("conditions", "")
        line = f"- {label}: {access}"
        if conditions:
            line += f" — {conditions}"
        lines.append(line)
    return "\n".join(lines)


def _fmt_entity_description(rows, params):
    entity = _short(params.get("entity_uri", ""))
    if not rows:
        return f"# {entity}\n(no information found)"
    r = rows[0]
    label = r.get("label", entity)
    lines = [f"# {label}"]
    desc = r.get("description", "")
    if desc:
        lines.append(desc)
    url = r.get("url", "")
    if url:
        lines.append(f"URL: {url}")
    activities = [_short(row.get("activity", "")) for row in rows if row.get("activity")]
    if activities:
        lines.append(f"Activities: {', '.join(sorted(set(activities)))}")
    status = r.get("status", "")
    if status:
        lines.append(f"Status/notes: {status}")
    return "\n".join(lines)


def _fmt_services_by_tool(rows, params):
    # Coalesce rows with same serviceUri: one service may have multiple altLabels
    # (e.g. "Visual Similarity Service" + altLabel "VisXP")
    services: dict = {}  # serviceUri → {toolLabel, labels: set, desc}
    for r in rows:
        uri = r.get("serviceUri", r.get("serviceLabel", ""))
        if uri not in services:
            services[uri] = {
                "tool": r.get("toolLabel", ""),
                "labels": set(),
                "desc": r.get("serviceDesc", ""),
            }
        services[uri]["labels"].add(r.get("serviceLabel", ""))
        alt = r.get("altLabel", "")
        if alt:
            services[uri]["labels"].add(alt)

    lines = ["# Backend services by tool"]
    for info in services.values():
        combined_name = " / ".join(sorted(info["labels"]))
        line = f"- {info['tool']} → {combined_name}"
        if info["desc"]:
            line += f": {info['desc'][:200]}"
        lines.append(line)
    return "\n".join(lines)


def _fmt_activities_by_tool(rows, params):
    tool = _short(params.get("tool_uri", ""))
    lines = [f"# Activities supported by {tool}"]
    for r in rows:
        activity = _short(r.get("activity", ""))
        pref = r.get("prefLabel", "")
        line = f"- {pref or activity}"
        lines.append(line)
    return "\n".join(lines)


def _fmt_generic(rows, params):
    if not rows:
        return ""
    lines = []
    for r in rows:
        lines.append(", ".join(f"{k}: {v}" for k, v in r.items()))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sparql_endpoint(kg_cfg: dict) -> str:
    return f"{kg_cfg['fuseki_url'].rstrip('/')}/{kg_cfg['dataset']}/sparql"


def _sparql_auth(kg_cfg: dict):
    user = kg_cfg.get("admin_user")
    pw = kg_cfg.get("admin_password")
    return (user, pw) if user and pw else None
