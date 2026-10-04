"""Match extracted facts to an AAS, producing a reviewable typed plan."""
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

import httpx

from mcp_aas.tools.base import BaseTool, ToolResult
from mcp_aas.resource_manager import TEMP_DIR, ATTACHMENTS_DIR, DEFAULT_AAS_ENDPOINT


async def resolve_label(value, endpoint):
    if "://" not in value and "/" not in value:
        return value
    from mcp_aas.aas_utils.basyx_client import encode_id
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{endpoint.rstrip('/')}/shells/{encode_id(value)}")
        response.raise_for_status()
        return response.json()["idShort"]


class AASMatchInputs(BaseTool):
    name: str = "aas_match_inputs"
    description: str = (
        "Match extracted entities to writable AAS elements using cached cosine retrieval, "
        "parent boundaries and three-way decisions. Emits a typed plan; review it before "
        "aas_populate_inputs. Unmatched entities may be created in a suitable collection, "
        "or skipped when no suitable host exists. Run aas_parse first."
    )
    parameters: dict = {"type": "object", "properties": {
        "input_file_name": {"type": "string", "description": "Extracted entities JSON filename in the attachments directory."},
        "aas_idShort": {"type": "string", "description": "Target AAS idShort or canonical AAS ID."},
        "endpoint": {"type": "string", "description": "BaSyx endpoint; defaults to AAS_SERVER_ENDPOINT."},
        "theta_high": {"type": "number", "default": 0.8, "description": "Auto-accept when similarity is at least this value."},
        "theta_low": {"type": "number", "default": 0.5, "description": "Reject direct matches below this value."},
        "top_k": {"type": "integer", "default": 5, "description": "Writable candidate count; containers are retrieved separately."},
        "no_arbitration": {"type": "boolean", "default": False, "description": "Use deterministic bands without chat calls; uncertain matches remain unresolved."},
        "create_in": {"type": "string", "description": "Optional existing collection path or unique collection idShort for unmatched facts."},
    }, "required": ["input_file_name", "aas_idShort"]}

    async def execute(self, input_file_name, aas_idShort, endpoint=None, theta_high=0.8,
                      theta_low=0.5, top_k=5, no_arbitration=False, create_in=None, **kwargs):
        from mcp_aas.semantic import match as algorithm
        from mcp_aas.semantic.runtime import load_index, embed, ask, cache_file
        endpoint = endpoint or DEFAULT_AAS_ENDPOINT
        try:
            if not 0 <= theta_low <= theta_high <= 1 or not 1 <= top_k <= 100:
                raise ValueError("Require 0 <= theta_low <= theta_high <= 1 and 1 <= top_k <= 100")
            label = await resolve_label(aas_idShort, endpoint)
            cache_file(label, ".json")
            base = Path(input_file_name.replace("\\", "/")).stem
            if base.endswith("_edit"):
                base = base[:-5]
            if base in ("", ".", ".."):
                raise ValueError("Input filename must identify a JSON entities file")
            source = Path(ATTACHMENTS_DIR) / (base + "_edit.json")
            if not source.exists():
                source = Path(ATTACHMENTS_DIR) / (base + ".json")
            entities = algorithm.load_entities(json.loads(source.read_bytes()))
            index, paths, snapshot_digest = await load_index(label)
            if not index.nodes:
                raise ValueError("The parsed AAS has no submodel elements")
            if create_in:
                choices = [n for n in index.nodes if n.model_type in algorithm.CONTAINER_TYPES
                           and (n.path == create_in or n.id_short == create_in)]
                if len(choices) != 1:
                    raise ValueError("create_in must identify exactly one existing collection/list")
                create_in = choices[0].path
            vectors = await asyncio.to_thread(embed, [entity.query() for entity in entities])
            arbitrate = None if no_arbitration else algorithm.make_arbitrator(ask, None)
            host = None if no_arbitration else algorithm.make_host_chooser(ask, None)
            sem = asyncio.Semaphore(5)

            async def one(entity, vector):
                candidates = algorithm.candidates_for(index, vector, top_k)
                async with sem:
                    decision = await asyncio.to_thread(algorithm.decide, entity, candidates,
                        theta_high=theta_high, theta_low=theta_low, arbitrate=arbitrate,
                        host=host, create_in=create_in)
                if decision.action == "create" and decision.path == create_in:
                    parent = next(n for n in index.nodes if n.path == create_in)
                    container = algorithm.Candidate(parent.path, parent.id_short, parent.model_type,
                        0., parent.parent, type_value_list_element=parent.type_value_list_element)
                    spec, warnings = algorithm.resolve_spec(entity, container, None)
                    decision.model_type, decision.value_type = spec.model_type, spec.value_type
                    decision.min_value, decision.max_value = spec.min_value, spec.max_value
                    decision.content_type = spec.content_type
                    decision.container_type = parent.model_type
                    decision.id_short = None if parent.model_type == "SubmodelElementList" else algorithm.safe_id_short(entity.entity)
                    decision.warnings.extend(warnings)
                selected = paths.get(decision.path) if decision.path else None
                return {"query": asdict(entity), "decision": asdict(decision),
                        "chosen_candidate": selected, "selected": True,
                        "warnings": decision.warnings, "snapshot_digest": snapshot_digest,
                        "aas_id": index.submodel_id,
                        "thresholds": {"high": theta_high, "low": theta_low}}

            decisions = await asyncio.gather(*(one(e, v) for e, v in zip(entities, vectors)))
            output = Path(TEMP_DIR) / f"{base}_{label}_matching_result.json"
            output.write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8")
            counts = {action: sum(d["decision"]["action"] == action for d in decisions) for action in ("write", "create", "skip")}
            return ToolResult(output=f"Matching results saved at {output}. {counts}. Review decisions and warnings before population.")
        except Exception as exc:
            return ToolResult(error=f"aas_match_inputs failed: {exc}")
