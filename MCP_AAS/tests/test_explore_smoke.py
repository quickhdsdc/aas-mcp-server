"""Optional live smoke test against a real AAS server.

Skipped unless MCP_AAS_TEST_ENDPOINT is set, e.g.:

    MCP_AAS_TEST_ENDPOINT=http://localhost:8081 uv run pytest tests/test_explore_smoke.py
"""

import json
import os

import pytest

ENDPOINT = os.environ.get("MCP_AAS_TEST_ENDPOINT")

pytestmark = pytest.mark.skipif(
    not ENDPOINT, reason="set MCP_AAS_TEST_ENDPOINT to run the live AAS smoke test"
)


async def test_aas_explore_live():
    from mcp_aas.tools.aas_explore import AASExplore

    result = await AASExplore().execute(endpoint=ENDPOINT, asset_kind="both")
    assert result.error is None, f"aas_explore errored: {result.error}"
    # Output is a JSON array of shells (possibly empty on a fresh server).
    parsed = json.loads(result.output)
    assert isinstance(parsed, list)
