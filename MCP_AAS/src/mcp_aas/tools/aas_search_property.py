from mcp_aas.tools.base import BaseTool, ToolResult
import os
from typing import Any, Optional
from mcp_aas.config import config, LLMSettings
# Heavy / optional deps (faiss, numpy, langchain_openai) are imported lazily
# inside execute() so the base install works without the [semantic] extra.

def get_llm_settings(profile: Optional[str] = None) -> LLMSettings:
    profiles = config.llm
    if profile is None:
        profile = "default"
    if profile not in profiles:
        raise KeyError(f"Unknown LLM profile '{profile}'. Available: {list(profiles.keys())}")
    return profiles[profile]

class AASSearchProperty(BaseTool):
    name: str = "aas_search_property"
    description: str = (
        "Searches for a property by name within a specific AAS instance using semantic similarity. "
        "Returns the top 5 closest matches with their idShort, description, and semantic path, "
        "expanded with bounded context from the AAS Knowledge Graph."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "prop_name": {
                "type": "string",
                "description": "The name (idShort or partial name) of the property to search for."
            },
            "aas_idShort": {
                "type": "string",
                "description": "The idShort of the AAS to be searched."
            }
        },
        "required": ["prop_name", "aas_idShort"]
    }

    async def execute(self, prop_name: str, aas_idShort: str, **kwargs) -> ToolResult:
        try:
            import networkx as nx
            import json
            import faiss
            import numpy as np
            from langchain_openai import AzureOpenAIEmbeddings

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

            # Step 2: Build search corpus from graph nodes
            rows = []
            texts = []
            for node_id, node_data in G.nodes(data=True):
                id_short = node_data.get("idShort", "")
                description = node_data.get("description", "")
                cd_definition = node_data.get("cd_definition", "")
                cd_unit = node_data.get("cd_unit", "")

                desc_text = description if description and description != 'None' else ''
                cd_text = cd_definition if cd_definition else ''
                final_desc = cd_text if cd_text else desc_text

                text = f"title: {id_short}, description: {final_desc}"
                if cd_unit:
                    text += f", unit: {cd_unit}"

                texts.append(text)
                rows.append((id_short, description, node_id))

            if not texts:
                return ToolResult(error="No properties found in the AAS graph.")

            # Step 3: Embeddings — always Azure OpenAI text-embedding-3-large-1.
            # Resolve Azure credentials from whichever configured profile is
            # azure-typed: the active [llm] when running on Azure, an explicit
            # [llm.embedding]/[llm.azure] when [llm] is set to vLLM. Embeddings
            # are tool-mediated infrastructure and are kept on Azure regardless
            # of which provider serves chat completions.
            azure_llm = None
            for profile in ("embedding", "azure", "default"):
                try:
                    candidate = get_llm_settings(profile)
                except KeyError:
                    continue
                if (candidate.api_type or "").strip().lower() == "azure":
                    azure_llm = candidate
                    break
            if azure_llm is None:
                return ToolResult(error=(
                    "aas_search_property: no azure-typed profile found for "
                    "embeddings. Add [llm.embedding] (api_type='azure') with "
                    "base_url, api_key, api_version to backend/config/config.toml."
                ))

            embeddings_model = AzureOpenAIEmbeddings(
                model="text-embedding-3-large-1",
                openai_api_version=azure_llm.api_version,
                azure_endpoint=azure_llm.base_url,
                openai_api_type="azure",
                openai_api_key=azure_llm.api_key,
            )
            query_text = prop_name if isinstance(prop_name, list) else [prop_name]
            corpus_vecs = embeddings_model.embed_documents(texts)
            query_vecs = embeddings_model.embed_documents(query_text)
            corpus = np.array(corpus_vecs, dtype="float32")
            queries = np.array(query_vecs, dtype="float32")

            dim = queries.shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(corpus)

            _, indices = index.search(queries, k=5)

            # Step 4: Symbolic Constraint Expansion using the same graph
            result_lines = []
            for i in indices[0]:
                if i < len(rows):
                    matched_idShort, matched_desc, matched_path = rows[i]

                    node_id = matched_path
                    if node_id in G:
                        # Context Expansion
                        parents = list(G.predecessors(node_id))
                        if parents:
                            parent = parents[0]
                            parent_data = G.nodes[parent]
                            siblings = list(G.successors(parent))
                            sibling_id_shorts = [G.nodes[sib].get('idShort', 'Unknown') for sib in siblings]

                            context_block = f"Neural Hit: {matched_idShort} | Match Description: {matched_desc}\n"
                            context_block += f"-> Context Boundary (Parent SMC): {parent_data.get('idShort')} (Type: {parent_data.get('type')})\n"
                            context_block += f"-> Sibling Elements within Boundary: {', '.join(sibling_id_shorts)}\n"
                            result_lines.append(context_block + "\n")
                            continue

                    # Fallback if no graph traversal succeeded
                    result_lines.append(f"idShort: {matched_idShort}, description: {matched_desc}, semantic_path: {matched_path}\n\n")

            output_string = "Search Results:\n\n" + "".join(result_lines)
            return ToolResult(output=output_string)

        except Exception as e:
            return ToolResult(error=f"search_aas_property failed: {str(e)}")
