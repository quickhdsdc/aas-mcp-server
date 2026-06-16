from mcp_aas.tools.base import BaseTool, ToolResult
from mcp_aas.aas_utils.basyx_client import BasyxApiClient
from mcp_aas.resource_manager import DEFAULT_AAS_ENDPOINT
import base64
import json

class AASExplore(BaseTool):
    name: str = "aas_explore"
    description: str = (
        "Given an endpoint of an AAS server, fetch the metainformation of AAS instances "
        "and their submodels hosted on the AAS server. "
        "Can filter by asset_kind: 'Type' for blueprints only, 'Instance' for instances only, "
        "or 'both'. If user did not specify the asset type, default to 'Instance'."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "endpoint": {
                "type": "string",
                "description": f"The base URL of the AAS server. Optional. (default: {DEFAULT_AAS_ENDPOINT})",
            },
            "asset_kind": {
                "type": "string",
                "description": "Filter by assetKind: 'Type' for blueprint AAS only, 'Instance' for instantiated AAS only, 'both' for all (default: 'Instance').",
                "enum": ["Type", "Instance", "both"],
                "default": "Instance",
            },
        },
        "required": [],
    }

    async def execute(self, endpoint: str = None, asset_kind: str = "Instance", **kwargs) -> ToolResult:
        endpoint = endpoint or DEFAULT_AAS_ENDPOINT
        try:
            client = BasyxApiClient(endpoint)
            shells = await client.get_shells()

            if not isinstance(shells, list):
                return ToolResult(error="Shells response is not a list.")

            results = []
            for shell in shells:
                aas_id = shell.get("id")
                id_short = shell.get("idShort")
                if not aas_id:
                    continue

                # Filter by assetKind. Use `or {}` not `.get(default={})`: BaSyx
                # returns "assetInformation": null sometimes, and dict.get
                # treats a None value as present (returns None, not the default).
                shell_asset_kind = (shell.get("assetInformation") or {}).get("assetKind", "Instance")
                if asset_kind != "both" and shell_asset_kind != asset_kind:
                    continue

                b64_id = base64.urlsafe_b64encode(aas_id.encode()).decode()
                # Get submodel references (returns list of ModelReference)
                submodel_refs = await client.get(f"/shells/{b64_id}/submodel-refs")
                submodel_refs = submodel_refs.get('result', [])
                submodel_infos = []
                for ref in submodel_refs:
                    try:
                        submodel_id = ref["keys"][0]["value"]  # Full URL
                        submodel_infos.append({
                            "id": submodel_id,
                        })
                    except Exception as e:
                        continue

                results.append({
                    "aas_id": aas_id,
                    "aas_idShort": id_short,
                    "assetKind": shell_asset_kind,
                    "submodels": submodel_infos,
                })

            return ToolResult(output=json.dumps(results, indent=2))

        except Exception as e:
            return ToolResult(error=f"AASExplore failed: {str(e)}")
