from typing import Any, Dict, List
from haystack.tools import Tool
from etsi_mec_agent.search import search_specs


def search_local_etsi_specs(query: str, diagrams_only: bool = False) -> str:
    """
    Search indexed local ETSI MEC specifications using Dense + ColBERT late-interaction in Qdrant.
    Returns matched markdown clauses, tables, and architectural diagram file paths.
    """
    hits = search_specs(query_text=query, top_k=3, diagrams_only=diagrams_only)
    if not hits:
        return f"No local ETSI specification passages found matching query: '{query}'."

    output_lines = [f"Found {len(hits)} matching local section(s):"]
    for i, h in enumerate(hits, 1):
        payload = h.meta or {}
        doc = payload.get("filename") or payload.get("doc_id")
        page = payload.get("page")
        heading = payload.get("heading", "")
        text = h.content or ""
        diagrams = payload.get("diagram_paths", [])

        output_lines.append(f"\n[Result {i}] Document: {doc} | Page: {page}")
        if heading:
            output_lines.append(f"Section: {heading}")
        if diagrams:
            output_lines.append(f"Diagram(s): {', '.join(diagrams)}")
        output_lines.append(f"Content excerpt:\n{text[:600]}...")

    return "\n".join(output_lines)


search_local_specs_tool = Tool(
    name="search_local_etsi_specs",
    description="Search locally ingested ETSI MEC specifications, markdown tables, and architectural diagrams using Qdrant hybrid search.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language technical query or clause name.",
            },
            "diagrams_only": {
                "type": "boolean",
                "description": "Whether to only return chunks that contain architecture diagrams/charts.",
            },
        },
        "required": ["query"],
    },
    function=search_local_etsi_specs,
)
