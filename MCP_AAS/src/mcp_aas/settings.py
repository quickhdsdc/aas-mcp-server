
import os

# ---------------------------------------------------------------------------
# AAS server target
# ---------------------------------------------------------------------------
# Base URL of the BaSyx (or other AAS-API-compliant) repository the tools talk
# to. Callers may also override per-call via a tool's ``endpoint`` argument.
DEFAULT_AAS_ENDPOINT = os.getenv("AAS_SERVER_ENDPOINT", "http://localhost:8081")

# Host used when rewriting delegated-operation endpoints (aas_operation_delegation).
DEFAULT_INTERNAL_HOST = os.getenv("INTERNAL_SERVICE_HOST", "127.0.0.1")

# ---------------------------------------------------------------------------
# Work directories
# ---------------------------------------------------------------------------
DATA_DIR = os.path.abspath(os.getenv("MCP_AAS_DATA_DIR", os.path.join(os.getcwd(), ".mcp_aas")))

# Cache for parsed AAS graphs (aas_parse writes <idShort>_graph.json here; the
# semantic tools read them back).
TEMP_DIR = os.path.abspath(os.getenv("MCP_AAS_TEMP_DIR", os.path.join(DATA_DIR, "temp")))

# Extracted-entity / attachment cache (input_* tools, aas_match_inputs).
ATTACHMENTS_DIR = os.path.abspath(
    os.getenv("MCP_AAS_ATTACHMENTS_DIR", os.path.join(DATA_DIR, "attachments"))
)

# Persisted Service/Resource manifest JSONs written by the crawler.
MANIFEST_DIR = os.path.abspath(
    os.getenv("MCP_AAS_MANIFEST_DIR", os.path.join(DATA_DIR, "manifests"))
)

RECORDS_DIR = os.path.abspath(os.getenv("MCP_AAS_RECORDS_DIR", os.path.join(DATA_DIR, "records")))

# Where the logger writes its rotating logfile.
LOG_DIR = os.path.abspath(os.getenv("MCP_AAS_LOG_DIR", os.path.join(DATA_DIR, "logs")))

# ---------------------------------------------------------------------------
# Manifest crawler
# ---------------------------------------------------------------------------
# Seconds between manifest refresh probes. 0 (default) disables the background
# crawler — the two lookup_*_manifest tools then return a "manifest empty" hint.
MANIFEST_POLL_INTERVAL = float(os.getenv("MCP_AAS_MANIFEST_POLL_INTERVAL", "0") or 0)

# ---------------------------------------------------------------------------
# MCP server bind (only used by the sse / streamable-http transports)
# ---------------------------------------------------------------------------
MCP_HOST = os.getenv("MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.getenv("MCP_PORT", "8000"))

# ---------------------------------------------------------------------------
# Create the work directories up-front. Several tools list TEMP_DIR to find a
# cached graph before anything has ever written to it (aas_read_property,
# aas_write_property, aas_describe_property, aas_search_property), so on a
# fresh install those calls would fail with FileNotFoundError instead of the
# intended "run aas_parse first" hint.
# ---------------------------------------------------------------------------
for _work_dir in (TEMP_DIR, ATTACHMENTS_DIR, MANIFEST_DIR, RECORDS_DIR):
    try:
        os.makedirs(_work_dir, exist_ok=True)
    except Exception:
        # A read-only filesystem must not stop the server from starting.
        pass
