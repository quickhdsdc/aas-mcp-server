from mcp_aas.tools.base import BaseTool, ToolResult
from mcp_aas.resource_manager import get_service_manifest
import json


class LookupServiceManifest(BaseTool):
    name: str = "lookup_service_manifest"
    description: str = (
        "Look up available delegated services (Operations) from the AAS Service Manifest. "
        "Returns all services grouped by asset, including their sm_id, sme_path, "
        "input/output variable schemas, and delegation endpoints. "
        "Optionally filter by asset idShort or blueprint type. For more details, please use aas_describe_property"
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
                "description": "Optional. Filter by 'Type' (blueprint), 'Instance', or 'both' (default: 'both').",
                "enum": ["Type", "Instance", "both"],
                "default": "both",
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
    def _normalize_asset_kind(value: object, default: str = "both") -> str:
        v = LookupServiceManifest._normalize_filter(value)
        if not v:
            return default
        return v if v in {"Type", "Instance", "both"} else default

    async def execute(self, asset_id: str = "", blueprint: str = "", asset_kind: str = "both", **kwargs) -> ToolResult:
        asset_id = self._normalize_filter(asset_id)
        blueprint = self._normalize_filter(blueprint)
        asset_kind = self._normalize_asset_kind(asset_kind, default="both")
        manifest = get_service_manifest()

        if not manifest:
            return ToolResult(
                output="Service Manifest is empty. No AAS instances with delegated services found. "
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
                output=f"No services found matching asset_id='{asset_id}', blueprint='{blueprint}'. "
                       f"Available assets: {list(manifest.keys())}"
            )

        return ToolResult(output=json.dumps(filtered, indent=2, ensure_ascii=False))
