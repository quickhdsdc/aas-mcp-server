# aas-mcp-server

A standalone **Model Context Protocol (MCP) server** that exposes
**Asset Administration Shell (AAS)** tooling to LLM agents, together with a
minimal **BaSyx** AAS environment for running it locally end-to-end.

This repository accompanies our work on neuro-symbolic, agent-driven AAS
interoperability: it lets an LLM parse an AAS instance into an addressable
knowledge graph and read, search, describe, populate, and delegate operations
over a live BaSyx AAS repository.

**Paper:** [Agentic active Asset Administration Shell for circular manufacturing](https://www.sciencedirect.com/science/article/pii/S0278612526001950). The paper presents an AAS-based architecture that connects semantic product-data integration with agent-driven shopfloor operation delegation.

## Repository layout

| Path | What it is |
|------|------------|
| [`MCP_AAS/`](MCP_AAS/) | The installable MCP server (`mcp-aas`) and its AAS tool-set. See [`MCP_AAS/README.md`](MCP_AAS/README.md). |
| [`BaSyxMinimal/`](BaSyxMinimal/) | A Docker Compose BaSyx stack (AAS environment, registries, discovery, MongoDB, Mosquitto) plus sample `.aasx` models. See [`BaSyxMinimal/README.md`](BaSyxMinimal/README.md). |

## Quick start

### 1. Bring up the AAS server

```bash
cd BaSyxMinimal
docker compose up -d
```

This starts the BaSyx AAS environment and registries locally and loads the
sample shells under `BaSyxMinimal/aas/`. Published host ports bind to `127.0.0.1` for local development.

> The Compose file ships throwaway local-dev MongoDB credentials
> (`mongoAdmin` / `mongoPassword`). Change them before any non-local use.

### 2. Install and run the MCP server

The `pyproject.toml` lives in the `MCP_AAS/` subdirectory:

```bash
cd MCP_AAS
pip install -e .            # core BaSyx tools (no LLM dependency)
# pip install -e ".[semantic]"   # + FAISS / Azure-OpenAI semantic tools

mcp-aas                       # stdio transport (default)
# mcp-aas --transport sse --port 8077
```

Or install directly from this repository:

```bash
pip install "git+https://github.com/quickhdsdc/aas-mcp-server.git#subdirectory=MCP_AAS"
```

### 3. Configuration (optional)

Only the two **semantic** tools (`aas_search_property`, `aas_match_inputs`)
need LLM/embedding credentials. Copy the example config and fill it in:

```bash
cp MCP_AAS/config/config.example.toml MCP_AAS/config/config.toml
```

`config/config.toml` is git-ignored, so your credentials stay local.

## License

MIT — see `MCP_AAS/pyproject.toml`.
