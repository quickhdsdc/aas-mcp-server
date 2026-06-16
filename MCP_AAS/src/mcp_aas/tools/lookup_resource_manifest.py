from mcp_aas.tools.base import BaseTool, ToolResult
from mcp_aas.resource_manager import get_resource_manifest
import json


class LookupResourceManifest(BaseTool):
    name: str = "lookup_resource_manifest"
    description: str = (
        "Look up the AAS Resource Manifest — Only for an overview, Optionally filter by asset idShort or blueprint type."
        " For more details like AAS/SM id and properties, please use aas_explore and aas_parse"
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "asset_id": {
                "type": "string",
                "description": "Optional. Filter by asset idShort (e.g. 'Metal3DPrinter_M1'). "
                               "Leave empty, null, or 'None' to return all assets.",
            },
            "blueprint": {
                "type": "string",
                "description": "Optional. Filter by blueprint type (e.g. 'Metal3DPrinter'). "
                               "Leave empty, null, or 'None' to return all blueprints.",
            },
            "asset_kind": {
                "type": "string",
                "description": "Optional. Filter by 'Type' (blueprint), 'Instance', or 'both' (default: 'Instance').",
                "enum": ["Type", "Instance", "both"],
                "default": "Instance",
            },
        },
        "required": [],
    }

    @staticmethod
    def _normalize_filter(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            v = value.strip()
            if v.lower() in {"", "none", "null", "all", "any", "*"}:
                return ""
            return v
        return str(value).strip()

    @staticmethod
    def _normalize_asset_kind(value: object, default: str = "Instance") -> str:
        v = LookupResourceManifest._normalize_filter(value)
        if not v:
            return default
        return v if v in {"Type", "Instance", "both"} else default

    async def execute(self, asset_id: str = "", blueprint: str = "", asset_kind: str = "Instance", **kwargs) -> ToolResult:
        asset_id = self._normalize_filter(asset_id)
        blueprint = self._normalize_filter(blueprint)
        asset_kind = self._normalize_asset_kind(asset_kind, default="Instance")
        manifest = get_resource_manifest()

        if not manifest:
            return ToolResult(
                output="Resource Manifest is empty. No AAS instances found on the server. "
                       "Ensure assets are instantiated on the BaSyx server."
            )

        # Filter
        filtered = {}
        for key, value in manifest.items():
            if asset_id and key != asset_id:
                continue
            if blueprint and value.get("blueprint", "") != blueprint:
                continue
            if asset_kind != "both" and value.get("asset_kind", "Instance") != asset_kind:
                continue
            filtered[key] = value

        if not filtered:
            return ToolResult(
                output=f"No assets found matching asset_id='{asset_id}', blueprint='{blueprint}'. "
                       f"Available assets: {list(manifest.keys())}"
            )

        return ToolResult(output=json.dumps(filtered, indent=2, ensure_ascii=False))
