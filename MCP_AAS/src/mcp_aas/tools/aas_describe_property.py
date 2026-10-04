from mcp_aas.tools.base import BaseTool, ToolResult
from mcp_aas.aas_utils.basyx_client import BasyxApiClient
import os
from typing import Optional
from mcp_aas.resource_manager import DEFAULT_AAS_ENDPOINT


class AASDescribeProperty(BaseTool):
    name: str = "aas_describe_property"
    description: str = (
        "Retrieves metadata (such as idShort, description, value type, and current value) "
        "for properties in an AAS by idShort name. If multiple properties share the same idShort, "
        "all matches are returned with their semantic paths for disambiguation."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "endpoint": {
                "type": "string",
                "description": f"The base URL of the AAS server. Optional. (default: {DEFAULT_AAS_ENDPOINT})"
            },
            "prop_idShort": {
                "type": "string",
                "description": "The idShort (name) of the property to describe, e.g. 'MaxClampingForce'."
            },
            "aas_idShort": {
                "type": "string",
                "description": "The idShort of the AAS that the property belongs to."
            }
        },
        "required": ["prop_idShort", "aas_idShort"]
    }

    def get_description_text(self, desc_dict, language='en'):
        if not desc_dict:
            return 'None'
        if isinstance(desc_dict, list):
            entries = [entry for entry in desc_dict if isinstance(entry, dict) and entry.get("text")]
            return next((entry["text"] for entry in entries if entry.get("language") == language),
                        entries[0]["text"] if entries else "None")
        try:
            # Try to get the preferred language
            if language in desc_dict:
                return desc_dict[language]
            # Fallback: return the first available language's text
            return next(iter(desc_dict.values()), 'None')
        except Exception:
            return 'None'

    async def execute(self, prop_idShort: str, aas_idShort: str, endpoint: Optional[str] = None, **kwargs) -> ToolResult:
        endpoint = endpoint or DEFAULT_AAS_ENDPOINT
        try:
            import networkx as nx
            import json

            from mcp_aas.resource_manager import TEMP_DIR
            # Step 1: Load graph
            from mcp_aas.semantic.runtime import cache_file
            graph_path = cache_file(aas_idShort, "_graph.json")

            if not graph_path.exists():
                return ToolResult(output="No cached Graph JSON file found. Should execute first the function aas_parse(endpoint, id).")

            with open(graph_path, 'r', encoding='utf-8') as f:
                G = nx.node_link_graph(json.load(f))

            # Step 2: Find ALL nodes matching the idShort
            matches = [
                (node_id, data) for node_id, data in G.nodes(data=True)
                if data.get("idShort") == prop_idShort
            ]

            if not matches:
                return ToolResult(error=f"No property with idShort '{prop_idShort}' found in AAS '{aas_idShort}' graph.")

            # Step 3: Describe each match via BaSyx API
            client = BasyxApiClient(endpoint)
            result_blocks = []

            for node_id, node_data in matches:
                api_path = node_data.get("API_path")
                if not api_path:
                    result_blocks.append(
                        f"[{node_id}] API path not available."
                    )
                    continue

                try:
                    result = await client.get(api_path)

                    id_short = result.get("idShort", "<unknown>")
                    model_type = result.get("modelType", "<unknown>")
                    description_text = self.get_description_text(result.get("description", []))

                    if model_type == "Property":
                        value_type = result.get("valueType", "<unknown>")
                        value = result.get("value", "<no value>")
                        block = (
                            f"Property '{id_short}' (semantic_path: {node_id}):\n"
                            f"  - Type: {model_type}\n"
                            f"  - Value Type: {value_type}\n"
                            f"  - Value: {value}\n"
                            f"  - Description: {description_text}"
                        )
                    else:
                        block = (
                            f"Element '{id_short}' (semantic_path: {node_id}):\n"
                            f"  - Type: {model_type}\n"
                            f"  - Description: {description_text}\n"
                            f"  - Raw JSON:\n{json.dumps(result, indent=2)}"
                        )
                    result_blocks.append(block)

                except Exception as e:
                    result_blocks.append(
                        f"[{node_id}] API call failed: {e}"
                    )

            header = f"Found {len(matches)} match(es) for idShort '{prop_idShort}':\n\n"
            return ToolResult(output=header + "\n\n".join(result_blocks))

        except Exception as e:
            return ToolResult(error=f"describe_aas_property failed: {str(e)}")
