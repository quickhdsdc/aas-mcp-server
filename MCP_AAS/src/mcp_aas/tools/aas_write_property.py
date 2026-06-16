from mcp_aas.tools.base import BaseTool, ToolResult
from mcp_aas.aas_utils.basyx_client import BasyxApiClient
import os
import json
from typing import Optional
from mcp_aas.resource_manager import DEFAULT_AAS_ENDPOINT


class AASWriteProperty(BaseTool):
    name: str = "aas_write_property"
    description: str = (
        "Overwrites property values in an AAS. Supports single write via prop_idShort/value "
        "or batch writes via updates=[{prop_idShort, value}, ...]. If multiple properties "
        "share the same idShort, all matches are updated."
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
                "description": "The idShort (name) of the property to write to, e.g. 'MaxClampingForce'."
            },
            "value": {
                "description": "The new value to assign to the property."
            },
            "aas_idShort": {
                "type": "string",
                "description": "The idShort of the AAS that the property belongs to."
            },
            "updates": {
                "type": "array",
                "description": "Optional batch updates. Each item must include prop_idShort and value.",
                "items": {
                    "type": "object",
                    "properties": {
                        "prop_idShort": {
                            "type": "string",
                            "description": "The idShort (name) of the property to write to."
                        },
                        "value": {
                            "description": "The new value to assign to the property."
                        }
                    },
                    "required": ["prop_idShort", "value"]
                }
            }
        },
        "required": ["aas_idShort"]
    }

    async def execute(self, prop_idShort: Optional[str] = None, value=None,
                      aas_idShort: str = "", endpoint: Optional[str] = None,
                      updates: Optional[list] = None, **kwargs) -> ToolResult:
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

            if updates is None:
                if prop_idShort is None or value is None:
                    return ToolResult(error="Provide either (prop_idShort, value) or updates=[{prop_idShort, value}, ...].")
                updates = [{"prop_idShort": prop_idShort, "value": value}]
            elif not isinstance(updates, list) or len(updates) == 0:
                return ToolResult(error="Parameter 'updates' must be a non-empty list.")

            # Step 2: Write value(s) via BaSyx API
            client = BasyxApiClient(endpoint, headers={"accept": "application/json", "Content-Type": "application/json"})
            result_lines: list[str] = []

            for item in updates:
                item_id_short = item.get("prop_idShort") if isinstance(item, dict) else None
                item_value = item.get("value") if isinstance(item, dict) else None
                if not item_id_short:
                    result_lines.append("  [invalid] missing prop_idShort in one updates item")
                    continue
                if item_value is None:
                    result_lines.append(f"  [invalid] idShort '{item_id_short}' missing value")
                    continue

                matches = [
                    (node_id, data) for node_id, data in G.nodes(data=True)
                    if data.get("idShort") == item_id_short
                ]
                if not matches:
                    result_lines.append(f"  '{item_id_short}': no matching property found")
                    continue

                result_lines.append(f"Updated {len(matches)} match(es) for idShort '{item_id_short}':")
                for node_id, node_data in matches:
                    api_path = node_data.get("API_path")
                    id_short = node_data.get("idShort")

                    if not api_path:
                        result_lines.append(f"  '{id_short}' (path: {node_id}): API path not available")
                        continue

                    try:
                        value_api = api_path + "/$value"
                        await client.patch(value_api, data=json.dumps(str(item_value)))
                        result_lines.append(f"  '{id_short}' (path: {node_id}): updated to {item_value}")
                    except Exception as e:
                        result_lines.append(f"  '{id_short}' (path: {node_id}): write failed - {e}")

            return ToolResult(output="\n".join(result_lines))

        except Exception as e:
            return ToolResult(error=f"write_aas_property failed: {str(e)}")
