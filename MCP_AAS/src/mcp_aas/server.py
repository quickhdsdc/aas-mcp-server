
import sys
import io
import os
import logging

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stderr)])
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import argparse
import asyncio
import atexit
import json
from contextlib import asynccontextmanager
from inspect import Parameter, Signature
from typing import Any, AsyncIterator, Dict, Optional

from mcp.server.fastmcp import FastMCP

from mcp_aas import settings
from mcp_aas.logger import logger
from mcp_aas.tools.base import BaseTool

# Core BaSyx tools (no LLM / embedding dependency).
from mcp_aas.tools.aas_explore import AASExplore
from mcp_aas.tools.aas_read_property import AASReadProperty
from mcp_aas.tools.aas_write_property import AASWriteProperty
from mcp_aas.tools.aas_describe_property import AASDescribeProperty
from mcp_aas.tools.aas_parse import AASParse
from mcp_aas.tools.aas_from_smt import AASfromSMT
from mcp_aas.tools.aas_populate_inputs import AASPopulateInputs
from mcp_aas.tools.aas_operation_delegation import AASOperationDelegation

# Manifest tools (need the background crawler to have data).
from mcp_aas.tools.lookup_resource_manifest import LookupResourceManifest
from mcp_aas.tools.lookup_service_manifest import LookupServiceManifest

# Keep stdout clean — the stdio transport sends JSON-RPC over stdout, so all
# human-readable logging must go to stderr (configured above + in logger.py).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def _semantic_available() -> bool:
    try:
        import faiss  # noqa: F401
        import langchain_openai  # noqa: F401
        import numpy  # noqa: F401
        return True
    except Exception:
        return False


class MCPServer:

    def __init__(
        self,
        name: str = "AAS-Tools",
        host: Optional[str] = None,
        port: Optional[int] = None,
        manifest_poll_interval: Optional[float] = None,
    ):
        self.host = host or settings.MCP_HOST
        self.port = port or settings.MCP_PORT
        self.poll_interval = (
            settings.MANIFEST_POLL_INTERVAL
            if manifest_poll_interval is None
            else manifest_poll_interval
        )
        self.server = FastMCP(name, host=self.host, port=self.port, lifespan=self._lifespan)
        self.tools: Dict[str, BaseTool] = {}

        # --- Core AAS tools (always available) ---
        self.tools["aas_explore"] = AASExplore()
        self.tools["aas_read_property"] = AASReadProperty()
        self.tools["aas_write_property"] = AASWriteProperty()
        self.tools["aas_describe_property"] = AASDescribeProperty()
        self.tools["aas_parse"] = AASParse()
        self.tools["aas_from_smt"] = AASfromSMT()
        self.tools["aas_populate_inputs"] = AASPopulateInputs()
        self.tools["aas_operation_delegation"] = AASOperationDelegation()

        # --- Manifest tools ---
        self.tools["lookup_resource_manifest"] = LookupResourceManifest()
        self.tools["lookup_service_manifest"] = LookupServiceManifest()

        # --- Semantic tools (only if the [semantic] extra is installed) ---
        if _semantic_available():
            from mcp_aas.tools.aas_search_property import AASSearchProperty
            from mcp_aas.tools.aas_match_inputs import AASMatchInputs

            self.tools["aas_search_property"] = AASSearchProperty()
            self.tools["aas_match_inputs"] = AASMatchInputs()
        else:
            logger.warning(
                "Semantic tools (aas_search_property, aas_match_inputs) disabled: "
                "install the optional extra with `pip install mcp-aas[semantic]` "
                "(faiss-cpu, numpy, langchain-openai, openai) to enable them."
            )

    @asynccontextmanager
    async def _lifespan(self, _server: FastMCP) -> AsyncIterator[None]:
        poll_task: Optional[asyncio.Task] = None
        if self.poll_interval and self.poll_interval > 0:
            from mcp_aas import resource_manager

            logger.info(
                f"Starting manifest crawler: every {self.poll_interval}s against "
                f"{settings.DEFAULT_AAS_ENDPOINT}"
            )
            poll_task = asyncio.create_task(
                resource_manager.poll_manifests(
                    self.poll_interval, settings.DEFAULT_AAS_ENDPOINT
                )
            )
        else:
            logger.info(
                "Manifest crawler disabled (MCP_AAS_MANIFEST_POLL_INTERVAL=0). "
                "lookup_*_manifest tools will report an empty manifest until enabled."
            )
        try:
            yield
        finally:
            if poll_task is not None:
                poll_task.cancel()
                try:
                    await poll_task
                except asyncio.CancelledError:
                    pass

    def register_tool(self, tool: BaseTool, method_name: Optional[str] = None) -> None:
        tool_name = method_name or tool.name
        tool_param = tool.to_param()
        tool_function = tool_param["function"]

        async def tool_method(**kwargs):
            logger.info(f"Executing {tool_name}: {kwargs}")
            result = await tool.execute(**kwargs)
            logger.info(f"Result of {tool_name}: {result}")
            if hasattr(result, "model_dump"):
                return json.dumps(result.model_dump())
            elif isinstance(result, dict):
                return json.dumps(result)
            return result

        tool_method.__name__ = tool_name
        tool_method.__doc__ = self._build_docstring(tool_function)
        tool_method.__signature__ = self._build_signature(tool_function)

        param_props = tool_function.get("parameters", {}).get("properties", {})
        required_params = tool_function.get("parameters", {}).get("required", [])
        tool_method._parameter_schema = {
            param_name: {
                "description": param_details.get("description", ""),
                "type": param_details.get("type", "any"),
                "required": param_name in required_params,
            }
            for param_name, param_details in param_props.items()
        }

        self.server.tool()(tool_method)
        logger.info(f"Registered tool: {tool_name}")

    def _build_docstring(self, tool_function: dict) -> str:
        description = tool_function.get("description", "")
        param_props = tool_function.get("parameters", {}).get("properties", {})
        required_params = tool_function.get("parameters", {}).get("required", [])

        docstring = description
        if param_props:
            docstring += "\n\nParameters:\n"
            for param_name, param_details in param_props.items():
                required_str = "(required)" if param_name in required_params else "(optional)"
                param_type = param_details.get("type", "any")
                param_desc = param_details.get("description", "")
                docstring += f"    {param_name} ({param_type}) {required_str}: {param_desc}\n"
        return docstring

    def _build_signature(self, tool_function: dict) -> Signature:
        param_props = tool_function.get("parameters", {}).get("properties", {})
        required_params = tool_function.get("parameters", {}).get("required", [])

        parameters = []
        for param_name, param_details in param_props.items():
            param_type = param_details.get("type", "")
            # Optional params must carry the schema's own default, not None —
            # FastMCP fills omitted arguments from this signature, so hardcoding
            # None here silently overrode defaults like asset_kind="Instance".
            default = (
                Parameter.empty
                if param_name in required_params
                else param_details.get("default", None)
            )

            annotation = Any
            if param_type == "string":
                annotation = str
            elif param_type == "integer":
                annotation = int
            elif param_type == "number":
                annotation = float
            elif param_type == "boolean":
                annotation = bool
            elif param_type == "object":
                annotation = dict
            elif param_type == "array":
                annotation = list

            parameters.append(
                Parameter(
                    name=param_name,
                    kind=Parameter.KEYWORD_ONLY,
                    default=default,
                    annotation=annotation,
                )
            )
        return Signature(parameters=parameters)

    def register_all_tools(self) -> None:
        for tool in self.tools.values():
            self.register_tool(tool)

    def run(self, transport: str = "stdio") -> None:
        self.register_all_tools()
        logger.info(
            f"Starting AAS MCP server ({transport}) with {len(self.tools)} tools: "
            f"{sorted(self.tools)}"
        )
        self.server.run(transport=transport)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone AAS MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Communication method: stdio (default) or sse",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"Port for the sse transport (default: MCP_PORT or {settings.MCP_PORT})",
    )
    parser.add_argument(
        "--manifest-poll",
        type=float,
        default=None,
        help="Manifest crawler interval in seconds; 0 disables "
        "(default: MCP_AAS_MANIFEST_POLL_INTERVAL)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = MCPServer(port=args.port, manifest_poll_interval=args.manifest_poll)
    server.run(transport=args.transport)


if __name__ == "__main__":
    main()
