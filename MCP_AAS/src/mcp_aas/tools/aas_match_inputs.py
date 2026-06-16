from mcp_aas.tools.base import BaseTool, ToolResult
import os
import json
import tomllib
from typing import Any, Optional
from mcp_aas.config import config, LLMSettings, PROJECT_ROOT, get_config_path
import asyncio
import re
from mcp_aas.resource_manager import TEMP_DIR, ATTACHMENTS_DIR, DEFAULT_AAS_ENDPOINT
# Heavy / optional deps (faiss, numpy, langchain_openai, openai) are imported
# lazily inside the functions that use them so the base install works without
# the [semantic] extra.


async def _resolve_aas_idShort(value: str) -> str:
    if not value:
        return value
    if "://" not in value and "/" not in value:
        return value
    try:
        from mcp_aas.aas_utils.basyx_client import BasyxApiClient
        from mcp_aas.aas_utils.aas_loader import encode_id
        client = BasyxApiClient(DEFAULT_AAS_ENDPOINT)
        meta = await client.get(f"/shells/{encode_id(value)}")
        resolved = meta.get("idShort") if isinstance(meta, dict) else None
        return resolved or value.rsplit("/", 1)[-1]
    except Exception:
        return value.rsplit("/", 1)[-1]


def get_llm_settings(profile: Optional[str] = None) -> LLMSettings:
    profiles = config.llm
    if profile is None:
        profile = "default"
    if profile not in profiles:
        raise KeyError(f"Unknown LLM profile '{profile}'. Available: {list(profiles.keys())}")
    return profiles[profile]


def _get_fixed_azure_settings_from_primary_config() -> Optional[LLMSettings]:
    cfg_path = get_config_path()
    if cfg_path is None or not cfg_path.exists():
        return None
    try:
        with cfg_path.open("rb") as f:
            raw = tomllib.load(f)
        llm = raw.get("llm", {})
        if str(llm.get("api_type", "")).strip().lower() != "azure":
            return None
        return LLMSettings(
            model=llm.get("model", ""),
            base_url=llm.get("base_url", ""),
            api_key=llm.get("api_key", ""),
            max_tokens=llm.get("max_tokens", 4096),
            max_completion_tokens=llm.get("max_completion_tokens", 4096),
            max_input_tokens=llm.get("max_input_tokens"),
            temperature=llm.get("temperature", 0.0),
            api_type=llm.get("api_type", "azure"),
            api_version=llm.get("api_version", ""),
        )
    except Exception:
        return None


def make_chat_client(profile: Optional[str] = "default") -> tuple[Any, str, dict]:
    from openai import OpenAI, AzureOpenAI

    llm = _get_fixed_azure_settings_from_primary_config() or get_llm_settings(profile)
    api_type = str(llm.api_type or "").strip().lower()

    if api_type == "azure":
        client = AzureOpenAI(
            api_key=llm.api_key,
            api_version=llm.api_version,
            azure_endpoint=llm.base_url,
        )
        deployment = llm.model
        default_kwargs = {
            "max_completion_tokens": llm.max_completion_tokens,
            "temperature": llm.temperature,
        }
        return client, deployment, default_kwargs

    elif api_type == "openai":
        client = OpenAI(api_key=llm.api_key)
        model = llm.model
        default_kwargs = {
            "model": model,
            "max_completion_tokens": llm.max_completion_tokens,
            "temperature": llm.temperature,
        }
        return client, model, default_kwargs

    else:
        raise ValueError(f"Unsupported api_type: {llm.api_type!r}")


class AASMatchInputs(BaseTool):
    name: str = "aas_match_inputs"
    description: str = (
        "To populate the created empty AAS based on the input file, it is necessary to match the exacted input entities with the AAS entities. This function performs the matching and delivers the results. After execution, you must ask user to confirm the match results (in UI) before populating the AAS."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "input_file_name": {
                "type": "string",
                "description": "Name of input_file, which contains extracted entities, such as PDF_DS_08968020001A0_EN.json."
            },
            "aas_idShort": {
                "type": "string",
                "description": "The idShort of the AAS to be populated (short label, e.g. 'TestNew'). A canonical AAS id (URI) is also accepted and will be resolved automatically. You may first call aas_explore() to get the mapping between idShort and id of AAS, then call aas_parse() to get the details of the target AAS."
            }
        },
        "required": ["input_file_name", "aas_idShort"]
    }

    async def execute(self, input_file_name: str, aas_idShort: str, **kwargs) -> ToolResult:
        import faiss
        import numpy as np
        from langchain_openai import AzureOpenAIEmbeddings
        from openai import OpenAI

        os.makedirs(ATTACHMENTS_DIR, exist_ok=True)
        os.makedirs(TEMP_DIR, exist_ok=True)

        # Coerce canonical AAS id into the idShort label aas_parse cached the
        # graph under, before it leaks into filename construction below.
        aas_idShort = await _resolve_aas_idShort(aas_idShort)

        # Normalize the input filename to get the true base name
        # Handle various formats: .pdf, .json, _edit.json
        filename = os.path.basename(input_file_name)

        # Remove extension first
        base = filename
        for ext in ['.pdf', '.json', '.PDF', '.JSON']:
            if base.endswith(ext):
                base = base[:-len(ext)]
                break

        # Remove _edit suffix if present to get the true base
        if base.endswith('_edit'):
            base = base[:-5]

        # Now base is the true base name (e.g., "PDF_DS_08968020001A0_EN")
        decisions_path = os.path.join(TEMP_DIR, f"{base}_{aas_idShort}_matching_result.json")
        if os.path.exists(decisions_path):
            with open(decisions_path, "r", encoding="utf-8") as f:
                final_decisions = json.load(f)

        else:
            # Try to find the input entities file - prefer _edit version
            json_path = os.path.join(ATTACHMENTS_DIR, f"{base}_edit.json")
            if not os.path.exists(json_path):
                # Fallback to original extraction file
                json_path = os.path.join(ATTACHMENTS_DIR, f"{base}.json")
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    input_entities = json.load(f)

            except Exception as e:
                return ToolResult(error=f"load json input file failed: {str(e)}. should first execute the function input_pdf2entity(pdf_name, n_pages) and let user edit the extracted entities.")

            import networkx as nx
            
            graph_path = os.path.join(TEMP_DIR, f"{aas_idShort}_graph.json")
            if not os.path.exists(graph_path):
                return ToolResult(error=f"Failed to load AAS graph data for '{aas_idShort}'. No graph cache found.")

            def load_graph():
                with open(graph_path, 'r', encoding='utf-8') as f:
                    return nx.node_link_graph(json.load(f))
            
            G = await asyncio.to_thread(load_graph)

            # --- build searchable corpus from JSON Graph
            def _coerce(s):
                return "" if s is None else str(s)

            aas_rows = []
            aas_texts = []
            for i, (node_id, node_data) in enumerate(G.nodes(data=True)):
                name = _coerce(node_data.get("idShort", ""))
                
                # Prioritize ConceptDescription definitions if available
                cd_def = _coerce(node_data.get("cd_definition", ""))
                desc = _coerce(node_data.get("description", ""))
                final_desc = cd_def if cd_def else desc
                
                text = f"name: {name}; description: {final_desc}"
                aas_rows.append(
                    {
                        "row_index": int(i),
                        "idShort": name,
                        "description": final_desc,
                        "node_id": node_id,
                        "full_row": node_data,
                    }
                )
                aas_texts.append(text)

            # --- build query strings from input_entities
            # Each item: {'description': ..., 'entity': ..., 'value': ...}
            query_items = []
            query_texts = []
            for ent in input_entities:
                entity = _coerce(ent.get("entity", ""))
                desc = _coerce(ent.get("description", ""))
                value = _coerce(ent.get("value", ""))
                qtext = f"name: {entity}; description: {desc}"
                query_items.append(
                    {
                        "entity": entity,
                        "description": desc,
                        "value": value,
                    }
                )
                query_texts.append(qtext)

            # --- embeddings
            llm = _get_fixed_azure_settings_from_primary_config() or get_llm_settings("default")
            api_type = str(llm.api_type or "").strip().lower()
            if api_type == "azure":
                embeddings_model = AzureOpenAIEmbeddings(
                    model="text-embedding-3-large-1",
                    openai_api_version=llm.api_version,
                    azure_endpoint=llm.base_url,
                    openai_api_type="azure",
                    openai_api_key=llm.api_key,
                )
                
                # Offload embedding network calls to threads
                corpus_vecs = await asyncio.to_thread(embeddings_model.embed_documents, aas_texts)
                query_vecs = await asyncio.to_thread(embeddings_model.embed_documents, query_texts)
                
                corpus = np.array(corpus_vecs, dtype="float32")
                queries = np.array(query_vecs, dtype="float32")
            elif api_type == "openai":
                client = OpenAI(api_key=llm.api_key)
                
                def get_embeddings(texts):
                    resp = client.embeddings.create(input=texts, model="text-embedding-3-large")
                    return [item.embedding for item in resp.data]
                
                corpus_list = await asyncio.to_thread(get_embeddings, aas_texts)
                corpus = np.array(corpus_list, dtype=np.float32)
                
                query_list = await asyncio.to_thread(get_embeddings, query_texts)
                queries = np.array(query_list, dtype=np.float32)
            else:
                return ToolResult(error=f"Unsupported api_type for embeddings in aas_match_inputs: {llm.api_type!r}")


            if corpus.ndim != 2 or queries.ndim != 2:
                if len(aas_texts) == 0:
                    return ToolResult(
                        error=(
                            f"AAS graph for '{aas_idShort}' has 0 nodes. "
                            f"Re-run aas_parse — the cached graph at "
                            f"{graph_path} is empty (parser likely dropped "
                            f"all objects; check backend logs for "
                            f"[aas_json_parser] and basyx SDK warnings)."
                        )
                    )
                return ToolResult(
                    error=(
                        f"Unexpected embedding shapes: "
                        f"corpus={corpus.shape}, queries={queries.shape}, "
                        f"aas_texts={len(aas_texts)}, query_texts={len(query_texts)}"
                    )
                )

            dim = corpus.shape[1]
            # --- cosine similarity via inner product
            def _l2_normalize(x: np.ndarray) -> np.ndarray:
                norms = np.linalg.norm(x, axis=1, keepdims=True)
                norms[norms == 0.0] = 1.0
                return x / norms

            corpus_norm = _l2_normalize(corpus)
            queries_norm = _l2_normalize(queries)
            
            def run_faiss_search():
                index = faiss.IndexFlatIP(dim)
                index.add(corpus_norm)
                k_val = 5
                return index.search(queries_norm, k_val)

            sims, idxs = await asyncio.to_thread(run_faiss_search)

            # --- assemble results
            results = []
            for qi, (q_meta, sim_row, idx_row) in enumerate(zip(query_items, sims, idxs)):
                top_matches = []
                for sim, ci in zip(sim_row.tolist(), idx_row.tolist()):
                    if ci == -1:
                        continue
                    meta = aas_rows[ci]
                    
                    # Sanitize helper
                    def _san_float(v): return 0.0 if v is None else float(v)
                    def _san_str(v): return "" if v is None else str(v)
                    
                    top_matches.append(
                        {
                            "similarity": _san_float(sim),
                            "idShort": _san_str(meta["idShort"]),
                            "description": _san_str(meta["description"]),
                            "row_index": meta["row_index"],
                            "semantic_path": meta["node_id"],
                            "type": _san_str(meta["full_row"].get('type')),
                            "API_path": _san_str(meta["full_row"].get('API_path')),
                            "semanticId": _san_str(meta["full_row"].get('semanticId')),
                        }
                    )

                results.append(
                    {
                        "query": q_meta,
                        "top_k_matches": top_matches,
                    }
                )
            # --- now we have results: each item has a query and a list of top_k_matches. Then post-retrieval decision per query
            def _safe(s):  # short truncation to keep prompts small
                if s is None:
                    return ""
                s = str(s)
                return s if len(s) <= 400 else (s[:400] + "…")

            STANDARD_PROMPT = """You are an expert on the industrial data mapping task. Given a QUERY entity (name and description) and a list of CANDIDATES (index, name, description, parent_boundary_group),
            select the SINGLE candidate that is the best *semantic* match for the QUERY. Since the value of the QUERY will be used to populate the matched candidate entity, the match should be restrict.
            Pay special attention to the 'parent_boundary_group'. It indicates the semantic Context Boundary (the Submodel or Collection bounding the candidate property). If the given query logically belongs to that Parent Group context, it is highly likely to be the correct match.
            If none of the candidates is appropriate, choose index -1. 
    
            Return ONLY JSON with keys:
            {"index": <int: the candidate index>, "candidate_entity_matched": "<idShort>", "reason_in_brief": "<short reason>"}
            If none of the candidates are suitable, return:
            {"index": -1, "candidate_entity_matched": "", "reason_in_brief": "why none fits"}.
            """

            COLLECTION_PROMPT = """You are an expert on the industrial data mapping task. We have a QUERY entity (name and description) and a CANDIDATE COLLECTION with its title and description.
            Choose the SINGLE best candidate collection that should host the QUERY data. (must choose one collection). For example, query is 'Charging power', a candidate is 'TechnicalProperties'. Obviously, charging power is a technical specification property, should belong to TechnicalProperties.
    
            Return ONLY JSON with keys:
            {"index": <int: collection index>, "idShort": "<idShort>", "reason_in_brief": "<short reason>"}.
            """

            TECHNICAL_ROUTING_PROMPT = """You are a classifier for industrial datasheet entities.
            Task: from the provided entities, select those that represent technical characteristics/properties
            of a product (e.g., electrical, mechanical, environmental ratings, dimensions, standards, approvals,
            material/performance indicators) that should be created under TechnicalData/TechnicalPropertyAreas.

            Exclude pure identifiers, logistics/commercial metadata, and contact/company info (e.g., GTIN,
            manufacturer address/email/phone, website, slogan, country of origin, tariff numbers, creation date).

            Return ONLY JSON in this exact shape:
            {
              "selected_indices": [<int>, ...],
              "reason": "<brief reason>"
            }
            where selected_indices are indices from the provided ENTITIES list.
            """

            def _ask_gpt(client, model, system_prompt, user_payload: dict, temperature=1):
                msg = json.dumps(user_payload, ensure_ascii=False, indent=2)
                req = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": msg},
                    ],
                    "reasoning_effort": "none",
                }
                if temperature is not None:
                    req["temperature"] = temperature

                try:
                    resp = client.chat.completions.create(**req)
                except Exception as e:
                    # Some Azure/OpenAI deployments reject non-default temperature.
                    err = str(e)
                    if "Unsupported value" in err and "temperature" in err:
                        req.pop("temperature", None)
                        resp = client.chat.completions.create(**req)
                    else:
                        raise
                txt = resp.choices[0].message.content.strip()
                try:
                    return json.loads(txt)
                except Exception:
                    # try crude block extraction
                    m = re.search(r"\{[\s\S]*\}", txt)
                    if m:
                        return json.loads(m.group(0))
                    raise

            # Use the local model_for_ranking variable
            client, model_for_ranking, kw = make_chat_client(profile="default")
            
            # Concurrency control
            sem = asyncio.Semaphore(5)

            # Discover TechnicalData/TechnicalPropertyAreas collection in the parsed graph.
            technical_collection = None
            for node_id, node_data in G.nodes(data=True):
                n_idshort = str(node_data.get("idShort", ""))
                n_type = str(node_data.get("type", ""))
                if n_idshort != "TechnicalPropertyAreas":
                    continue
                if n_type not in ("SubmodelElementCollection", "SubmodelElementList"):
                    continue
                if "TechnicalData" not in str(node_id):
                    continue
                technical_collection = {
                    "idShort": n_idshort,
                    "description": str(node_data.get("description", "")),
                    "type": n_type,
                    "API_path": str(node_data.get("API_path", "")),
                    "semantic_path": str(node_id),
                    "semanticId": str(node_data.get("semanticId", "")),
                }
                break

            # Collect low-similarity entities for optional TechnicalData routing.
            low_sim_for_technical = []
            for q_idx, block in enumerate(results):
                raw_cands = block.get("top_k_matches") or []
                filtered_props = [
                    c for c in raw_cands
                    if (c.get("type") or "") not in (
                        "AssetAdministrationShell", "Submodel", "SubmodelElementCollection", "SubmodelElementList"
                    )
                ]
                best_sim = float((filtered_props[0] or {}).get("similarity", 0.0)) if filtered_props else -1.0
                if best_sim < 0.7:
                    q = block.get("query") or {}
                    low_sim_for_technical.append(
                        {
                            "index": q_idx,
                            "entity": _safe(q.get("entity")),
                            "description": _safe(q.get("description")),
                            "value": _safe(q.get("value")),
                            "best_similarity": round(best_sim, 4),
                        }
                    )

            technical_routed_indices = set()
            if technical_collection and low_sim_for_technical:
                try:
                    routing_payload = {
                        "ENTITIES": low_sim_for_technical,
                        "NOTE": "Return indices from ENTITIES[index].index",
                    }
                    routing_sel = await asyncio.to_thread(
                        _ask_gpt,
                        client,
                        model_for_ranking,
                        TECHNICAL_ROUTING_PROMPT,
                        routing_payload,
                        1,
                    )
                    selected = routing_sel.get("selected_indices", []) if isinstance(routing_sel, dict) else []
                    for idx in selected:
                        try:
                            technical_routed_indices.add(int(idx))
                        except Exception:
                            continue
                except Exception:
                    technical_routed_indices = set()

            async def process_single_query(q_idx, block):
                q = block["query"]
                q_name = _safe(q.get("entity"))
                q_desc = _safe(q.get("description"))
                q_value = _safe(q.get("value"))

                # Backend-friendly auto-routing for low-sim industrial technical properties.
                if q_idx in technical_routed_indices and technical_collection:
                    return {
                        "query": q,
                        "primary_selection": {
                            "index": -1,
                            "candidate_entity_matched": "",
                            "reason_in_brief": "routed_to_TechnicalPropertyAreas_by_llm",
                        },
                        "collection_selection": {
                            "index": -1,
                            "idShort": technical_collection.get("idShort", ""),
                            "reason_in_brief": "Classified as industrial technical property for creation",
                        },
                        "chosen_candidate": technical_collection,
                    }

                # 1) Filter candidates we can use for direct matching
                raw_cands = block["top_k_matches"]

                filtered = []
                collections = []
                for j, c in enumerate(raw_cands):
                    ctype = c.get("type") or ""
                    if ctype in ("AssetAdministrationShell", "Submodel"):
                        continue  # discard
                    elif ctype in ("SubmodelElementCollection", "SubmodelElementList"):
                        collections.append((j, c))
                    else:
                        filtered.append((j, c))

                # 2) Decision Logic (Thresholds vs LLM)
                primary_selection = {"index": -1, "candidate_entity_matched": "",
                                     "reason_in_brief": "no suitable candidates"}

                if filtered:
                    # Check top candidate similarity
                    # filtered is list of (original_index, candidate_dict)
                    # candidates are sorted by similarity desc from retrieval
                    best_match_tuple = filtered[0] # (index, candidate)
                    best_match_cand = best_match_tuple[1]
                    best_sim = float(best_match_cand.get("similarity", 0.0))

                    if best_sim > 0.8:
                        # Auto-match high confidence
                        primary_selection = {
                            "index": best_match_tuple[0],
                            "candidate_entity_matched": best_match_cand.get("idShort") or "",
                            "reason_in_brief": f"Auto-match: High similarity ({best_sim:.4f} > 0.8)",
                        }
                    elif best_sim < 0.5:
                        # Auto-skip low confidence
                        primary_selection = {
                            "index": -1,
                            "candidate_entity_matched": "",
                            "reason_in_brief": f"Auto-skip: Low similarity ({best_sim:.4f} < 0.5)",
                        }
                    else:
                        # Ambiguous: Ask LLM with concurrency limit
                        cand_payload = []
                        for j, c in filtered:
                            # Graph Context Boundary Tracing
                            node_id = c.get("semantic_path")
                            parent_group = "Unknown"
                            if node_id and node_id in G:
                                parents = list(G.predecessors(node_id))
                                if parents:
                                    parent_group = str(G.nodes[parents[0]].get('idShort', 'Unknown'))
                                    
                            cand_payload.append(
                                {
                                    "index": j,
                                    "idShort": c.get("idShort"),
                                    "description": _safe(c.get("description")),
                                    "parent_boundary_group": parent_group,
                                    "semantic_similarity": round(float(c.get("similarity", 0.0)), 4),
                                }
                            )

                        user_payload = {
                            "QUERY": {
                                "name": q_name,
                                "description": q_desc,
                            },
                            "CANDIDATES": cand_payload
                        }

                        async with sem:
                            try:
                                # Run blocking LLM call in thread
                                sel = await asyncio.to_thread(
                                    _ask_gpt, client, model_for_ranking, STANDARD_PROMPT, user_payload, kw["temperature"]
                                )
                                sel = {
                                    "index": int(sel.get("index", -1)),
                                    "candidate_entity_matched": sel.get("candidate_entity_matched", "") or "",
                                    "reason_in_brief": _safe(sel.get("reason_in_brief", "")),
                                }
                                primary_selection = sel
                            except Exception as e:
                                primary_selection = {
                                    "index": -1,
                                    "candidate_entity_matched": "",
                                    "reason_in_brief": f"ranking failed: {e}",
                                }

                # If primary succeeded (or auto-matched), finalize and skip collections
                if primary_selection.get("index", -1) != -1:
                    chosen_idx = primary_selection["index"]
                    chosen = raw_cands[chosen_idx] if 0 <= chosen_idx < len(raw_cands) else None

                    return {
                        "query": q,
                        "primary_selection": primary_selection,
                        "collection_selection": None,
                        "chosen_candidate": chosen,
                    }

                # 3) Collection Logic (Only if primary failed)
                collection_selection = {"index": -1, "idShort": "", "reason_in_brief": "no collections to choose from"}
                if collections:
                     cand_collection_payload = []
                     for j, c in collections:
                         cand_collection_payload.append({
                             "index": j,
                             "idShort": c.get("idShort"),
                             "description": _safe(c.get("description")),
                             "semantic_similarity": round(float(c.get("similarity", 0.0)), 4),
                         })
                     
                     user_payload = {
                        "QUERY": {"name": q_name, "description": q_desc},
                        "CANDIDATES": cand_collection_payload
                     }
                     
                     async with sem: 
                        try:
                            sel_collection = await asyncio.to_thread(
                                _ask_gpt, client, model_for_ranking, COLLECTION_PROMPT, user_payload, temperature=1
                            )
                            collection_selection = {
                                "index": int(sel_collection.get("index", -1)),
                                "idShort": sel_collection.get("idShort", "") or "",
                                "reason_in_brief": _safe(sel_collection.get("reason_in_brief", "")),
                            }
                        except Exception as e:
                            collection_selection = {
                                "index": -1, 
                                "idShort": "", 
                                "reason_in_brief": f"collection ranking failed: {e}"
                            }

                chosen_collection = None
                ci = collection_selection.get("index", -1)
                if ci != -1:
                    chosen_collection = raw_cands[ci] if 0 <= ci < len(raw_cands) else None

                return {
                    "query": q,
                    "primary_selection": primary_selection,
                    "collection_selection": collection_selection,
                    "chosen_candidate": chosen_collection,
                }

            # Run all queries in parallel
            tasks = [process_single_query(i, block) for i, block in enumerate(results)]
            final_decisions = await asyncio.gather(*tasks)

            # Optionally persist decisions
            def save_decisions():
                with open(decisions_path, "w", encoding="utf-8") as f:
                    json.dump(final_decisions, f, indent=2, ensure_ascii=False)
            
            await asyncio.to_thread(save_decisions)

        # ---- build human-readable summary lines and return ----
        lines = []
        for dec in final_decisions:
            q = dec.get("query", {}) or {}
            q_entity = q.get("entity", "").strip() or "(unknown entity)"

            chosen = dec.get("chosen_candidate") or {}
            aas_type = (chosen.get("type") or "").strip()
            aas_idshort = (chosen.get("idShort") or "").strip()

            if aas_idshort:
                lines.append(
                    f"{q_entity} in input file matches AAS {aas_type} {aas_idshort}. "
                    f"The value of the input file entity should be written to that AAS entity."
                )
            else:
                # No match found (primary failed and collection either not chosen or -1)
                lines.append(
                    f"{q_entity} in input file has no suitable AAS match. "
                    f"Skip or map manually."
                )

        final_message = f"The input-AAS entity matching results are saved at {decisions_path}"

        return ToolResult(output=final_message)





