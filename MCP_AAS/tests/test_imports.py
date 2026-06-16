"""Import smoke tests — confirm every module imports and every core tool
instantiates with no lingering `app.*` dependency."""

import importlib
import pkgutil

import pytest

import mcp_aas


def test_no_app_imports_remain():
    """No vendored module may still import from the original `app.*` package."""
    import mcp_aas.tools  # noqa: F401

    pkg_paths = list(mcp_aas.__path__)
    offenders = []
    for _finder, name, _ispkg in pkgutil.walk_packages(pkg_paths, prefix="mcp_aas."):
        try:
            mod = importlib.import_module(name)
        except Exception:
            # Optional-dep modules may fail to import without [semantic]; that's
            # covered by the lazy-import design and tested separately.
            continue
        src = getattr(mod, "__file__", None)
        if not src:
            continue
        with open(src, "r", encoding="utf-8") as f:
            text = f.read()
        if "from app." in text or "import app." in text:
            offenders.append(name)
    assert not offenders, f"modules still reference app.*: {offenders}"


def test_core_modules_import():
    import mcp_aas.settings  # noqa: F401
    import mcp_aas.config  # noqa: F401
    import mcp_aas.logger  # noqa: F401
    import mcp_aas.resource_manager  # noqa: F401
    import mcp_aas.aas_utils.basyx_client  # noqa: F401
    import mcp_aas.aas_utils.aas_loader  # noqa: F401


CORE_TOOLS = [
    ("mcp_aas.tools.aas_explore", "AASExplore"),
    ("mcp_aas.tools.aas_read_property", "AASReadProperty"),
    ("mcp_aas.tools.aas_write_property", "AASWriteProperty"),
    ("mcp_aas.tools.aas_describe_property", "AASDescribeProperty"),
    ("mcp_aas.tools.aas_parse", "AASParse"),
    ("mcp_aas.tools.aas_from_smt", "AASfromSMT"),
    ("mcp_aas.tools.aas_populate_inputs", "AASPopulateInputs"),
    ("mcp_aas.tools.aas_operation_delegation", "AASOperationDelegation"),
    ("mcp_aas.tools.lookup_resource_manifest", "LookupResourceManifest"),
    ("mcp_aas.tools.lookup_service_manifest", "LookupServiceManifest"),
]


@pytest.mark.parametrize("module_name, class_name", CORE_TOOLS)
def test_core_tool_instantiates(module_name, class_name):
    mod = importlib.import_module(module_name)
    tool = getattr(mod, class_name)()
    assert tool.name
    assert tool.description
    # Every tool must expose a function-call schema.
    param = tool.to_param()
    assert param["function"]["name"] == tool.name


def test_semantic_tools_import_classes():
    """The semantic tool *classes* must import even without the extra installed
    (heavy deps are lazy). Instantiation must also succeed."""
    from mcp_aas.tools.aas_search_property import AASSearchProperty
    from mcp_aas.tools.aas_match_inputs import AASMatchInputs

    assert AASSearchProperty().name == "aas_search_property"
    assert AASMatchInputs().name == "aas_match_inputs"


def test_server_builds():
    """The server constructs and reports a sane tool count (>= 10 core+manifest)."""
    from mcp_aas.server import MCPServer

    srv = MCPServer(manifest_poll_interval=0)
    assert len(srv.tools) >= 10
    assert "aas_explore" in srv.tools
    assert "lookup_service_manifest" in srv.tools
