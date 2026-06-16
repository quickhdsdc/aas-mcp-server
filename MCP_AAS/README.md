# mcp-aas — Asset Administration Shell MCP server

A standalone [Model Context Protocol](https://modelcontextprotocol.io) server that
exposes a set of **Asset Administration Shell (AAS)** tools so any MCP-capable
agent system (Claude Desktop, Claude Code, custom Agent SDK apps, …) can explore,
read, write, parse, and operate on AAS instances hosted on a
[BaSyx](https://www.eclipse.org/basyx/) (or any AAS-API-compliant) server.

It is a self-contained extraction of the AAS tool-set from the EmbodiedTwin
agent — no `app.*` dependencies, installable on its own.

## Tools

**Core (always available, BaSyx REST only):**

| Tool | Purpose |
|------|---------|
| `aas_explore` | List AAS shells + their submodels on a server (filter by assetKind) |
| `aas_read_property` | Read a property value from a submodel |
| `aas_write_property` | Write/patch a property value |
| `aas_describe_property` | Describe a property (type, semantics, metadata) |
| `aas_parse` | Parse an AAS into a cached knowledge graph (`<idShort>_graph.json`) |
| `aas_from_smt` | Instantiate a new AAS from a Submodel Template |
| `aas_populate_inputs` | Populate an AAS from matched input entities |
| `aas_operation_delegation` | Invoke a delegated Operation on a submodel |
| `lookup_resource_manifest` | Overview of crawled AAS instances (needs crawler, see below) |
| `lookup_service_manifest` | Overview of delegated services/operations (needs crawler) |

**Semantic (optional `[semantic]` extra — Azure/OpenAI embeddings + faiss):**

| Tool | Purpose |
|------|---------|
| `aas_search_property` | Semantic property search within a parsed AAS graph |
| `aas_match_inputs` | Match extracted input entities to AAS entities (embeddings + LLM ranking) |

## Install

```bash
# Core install
uv sync                  # or: pip install .

# With the semantic tools
uv sync --extra semantic # or: pip install ".[semantic]"
```

## Run

```bash
mcp-aas                              # stdio (default) — for local agent embedding
mcp-aas --transport sse --port 8077  # networked SSE/HTTP
```

Point it at your AAS server via the environment (see below). Inspect it with the
MCP Inspector:

```bash
npx @modelcontextprotocol/inspector uv run mcp-aas
```

### Use from Claude Desktop / Claude Code

Add to your MCP client config (e.g. `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "aas": {
      "command": "uv",
      "args": ["run", "mcp-aas"],
      "env": {
        "AAS_SERVER_ENDPOINT": "http://localhost:8081"
      }
    }
  }
}
```

## Configuration (environment)

All configuration is environment-driven — no hardcoded hosts or ports.

| Variable | Purpose | Default |
|----------|---------|---------|
| `AAS_SERVER_ENDPOINT` | Target BaSyx/AAS server base URL | `http://localhost:8081` |
| `INTERNAL_SERVICE_HOST` | Host used to rewrite delegated-operation endpoints | `127.0.0.1` |
| `AAS_SSL_VERIFY` | Verify TLS certs on AAS calls (`true`/`false`) | `false` |
| `MCP_AAS_DATA_DIR` | Root for cache/log dirs | `<cwd>/.mcp_aas` |
| `MCP_AAS_TEMP_DIR` | Parsed-graph cache | `<DATA_DIR>/temp` |
| `MCP_AAS_ATTACHMENTS_DIR` | Extracted-entity / attachment cache | `<DATA_DIR>/attachments` |
| `MCP_AAS_MANIFEST_DIR` | Persisted manifest JSONs | `<DATA_DIR>/manifests` |
| `MCP_AAS_LOG_DIR` | Logfile directory | `<DATA_DIR>/logs` |
| `MCP_AAS_CONFIG_FILE` | LLM profile TOML (semantic tools) | `config/config.toml` |
| `MCP_AAS_MANIFEST_POLL_INTERVAL` | Manifest crawler interval (s); `0` = off | `0` |
| `MCP_HOST` / `MCP_PORT` | Bind for the SSE/HTTP transport | `127.0.0.1` / `8000` |

### Manifest tools

`lookup_resource_manifest` and `lookup_service_manifest` read manifests built by
a background crawler that periodically scans the AAS server. Enable it by setting
`MCP_AAS_MANIFEST_POLL_INTERVAL` to a positive number of seconds (e.g. `15`) or
passing `--manifest-poll 15`. With the crawler off, these two tools report an
empty manifest.

### Semantic tools

`aas_search_property` and `aas_match_inputs` require the `[semantic]` extra **and**
an Azure-typed LLM profile. Copy `config/config.example.toml` to
`config/config.toml` and fill in credentials (or point `MCP_AAS_CONFIG_FILE` at
your own file).

## License

MIT
