"""Cosine search with cached corpus embeddings and actual parent boundaries."""
import asyncio
import json

from mcp_aas.tools.base import BaseTool, ToolResult


class AASSearchProperty(BaseTool):
    name: str = "aas_search_property"
    description: str = "Search parsed AAS elements by cosine similarity. Returns scores, paths, parent boundaries and siblings. Run aas_parse first."
    parameters: dict = {"type": "object", "properties": {
        "prop_name": {"type": "string", "description": "Property name or semantic query."},
        "aas_idShort": {"type": "string", "description": "Exact parsed AAS idShort."},
        "top_k": {"type": "integer", "default": 5, "description": "Maximum results."},
        "min_score": {"type": "number", "default": 0.0, "description": "Minimum cosine similarity."},
        "model_type": {"type": "string", "description": "Optional AAS modelType filter, applied before ranking."},
    }, "required": ["prop_name", "aas_idShort"]}

    async def execute(self, prop_name, aas_idShort, top_k=5, min_score=0., model_type=None, **kwargs):
        from mcp_aas.semantic.runtime import load_index, embed
        from mcp_aas.semantic.index import search, Filter
        try:
            if not 1 <= top_k <= 100 or not -1 <= min_score <= 1:
                raise ValueError("Require 1 <= top_k <= 100 and -1 <= min_score <= 1")
            index, paths, _ = await load_index(aas_idShort)
            if not index.nodes:
                return ToolResult(output="No submodel elements found in the parsed AAS")
            vector = (await asyncio.to_thread(embed, [prop_name]))[0]
            hits = search(index, vector, top_k=top_k, min_score=min_score,
                          where=Filter(model_type=model_type) if model_type else None)
            results = [{**paths[h.node.path], "description": h.node.definition or h.node.description,
                        "similarity": h.score, "parent_boundary": h.parent.path if h.parent else h.node.submodel_id_short,
                        "siblings": h.siblings, "also_in": h.also_in} for h in hits]
            return ToolResult(output=json.dumps({"results": results}, ensure_ascii=False, indent=2))
        except Exception as exc:
            return ToolResult(error=f"aas_search_property failed: {exc}")
