from mcp_aas.tools.base import BaseTool, ToolResult
from mcp_aas.aas_utils.basyx_client import BasyxApiClient
import os
from typing import Optional
from mcp_aas.resource_manager import DEFAULT_AAS_ENDPOINT


class AASReadProperty(BaseTool):
    name: str = "aas_read_property"
    description: str = (
        "Reads the current value of properties in an AAS by idShort name. "
        "If multiple properties share the same idShort, all matches are returned "
        "with their values and semantic paths for disambiguation."
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
                "description": "The idShort (name) of the property to read, e.g. 'MaxClampingForce'."
            },
            "aas_idShort": {
                "type": "string",
                "description": "The idShort of the AAS that the property belongs to."
            }
        },
        "required": ["prop_idShort", "aas_idShort"]
    }

    async def execute(self, prop_idShort: str, aas_idShort: str, endpoint: Optional[str] = None, **kwargs) -> ToolResult:
        endpoint = endpoint or DEFAULT_AAS_ENDPOINT
        try:
            import networkx as nx
            import json

            from mcp_aas.resource_manager import TEMP_DIR
            # Step 1: Load graph
            graph_path = None
            for file in os.listdir(TEMP_DIR):
                if file.endswith("_graph.json") and aas_idShort in file:
                    graph_path = os.path.join(TEMP_DIR, file)
                    break

            if not graph_path:
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

            # Step 3: Read value for each match via BaSyx API
            client = BasyxApiClient(endpoint)
            result_lines = []

            for node_id, node_data in matches:
                api_path = node_data.get("API_path")
                id_short = node_data.get("idShort")

                if not api_path:
                    result_lines.append(f"  '{id_short}' (path: {node_id}): API path not available")
                    continue

                try:
                    value_api = api_path + "/$value"
                    result = await client.get(value_api)
                    result_lines.append(f"  '{id_short}' (path: {node_id}): {result}")
                except Exception as e:
                    result_lines.append(f"  '{id_short}' (path: {node_id}): read failed - {e}")

            header = f"Found {len(matches)} match(es) for idShort '{prop_idShort}':\n"
            return ToolResult(output=header + "\n".join(result_lines))

        except Exception as e:
            return ToolResult(error=f"read_aas_property failed: {str(e)}")
