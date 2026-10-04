"""Apply reviewed matching decisions with typed values and target ownership checks."""
import json
import re
from dataclasses import fields
from pathlib import Path
from urllib.parse import unquote, urlsplit

from mcp_aas.tools.base import BaseTool, ToolResult
from mcp_aas.aas_utils.basyx_client import BasyxApiClient, encode_id, decode_id
from mcp_aas.resource_manager import TEMP_DIR, DEFAULT_AAS_ENDPOINT


def target_path(candidate, owned):
    path = candidate.get("API_path") or candidate.get("apiPath") or ""
    parsed = urlsplit(path)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("Matching target must be a relative BaSyx API path")
    parts = parsed.path.split("/", 4)
    if len(parts) < 3 or parts[1] != "submodels":
        raise ValueError("Matching target must name a submodel")
    owner = decode_id(unquote(parts[2]))
    if owner not in owned:
        raise ValueError("Matching target is not linked to the requested AAS")
    if len(parts) > 3 and (parts[3] != "submodel-elements" or len(parts) < 5):
        raise ValueError("Invalid submodel-element path")
    if len(parts) == 5 and not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*(?:\[\d+\]|\.[A-Za-z][A-Za-z0-9_]*)*", unquote(parts[4])):
        raise ValueError("Invalid AAS element path")
    return path.rstrip("/")


class AASPopulateInputs(BaseTool):
    name: str = "aas_populate_inputs"
    description: str = "Apply a reviewed match plan. Supports typed writes/creation and legacy matching files; validates AAS ownership and reports failures."
    parameters: dict = {"type": "object", "properties": {
        "match_result_path": {"type": "string", "description": "Reviewed aas_match_inputs result JSON path."},
        "aas_idShort": {"type": "string", "description": "Target AAS idShort."},
        "endpoint": {"type": "string", "description": "BaSyx endpoint."},
        "dry_run": {"type": "boolean", "default": False, "description": "Validate and preview without changing BaSyx."},
        "skip_warnings": {"type": "boolean", "default": False, "description": "Skip decisions carrying warnings, including unit mismatches."},
    }, "required": ["match_result_path", "aas_idShort"]}

    async def execute(self, match_result_path, aas_idShort, endpoint=None, dry_run=False, skip_warnings=False, **kwargs):
        from mcp_aas.semantic.match import (Decision, Entity, infer_element_type, resolve_spec,
            Candidate, safe_id_short, element_for_create, value_for_write, CONTAINER_TYPES)
        from mcp_aas.semantic.runtime import cache_file
        from mcp_aas.aas_utils import aas_loader
        endpoint = endpoint or DEFAULT_AAS_ENDPOINT
        applied, created, skipped, failures = [], [], [], []
        client = BasyxApiClient(endpoint, headers={"accept": "application/json", "Content-Type": "application/json"})
        try:
            cache_file(aas_idShort, ".json")
            items = json.loads(Path(match_result_path).read_text(encoding="utf-8"))
            if not isinstance(items, list):
                raise ValueError("Matching file must contain a list of decisions")
            shells = await client.get_shells()
            targets = [shell for shell in shells if shell.get("idShort") == aas_idShort]
            if len(targets) != 1:
                raise ValueError("Target idShort must identify exactly one live AAS")
            shell = targets[0]
            refs = await client.get(f"/shells/{encode_id(shell['id'])}/submodel-refs")
            refs = refs.get("result", []) if isinstance(refs, dict) else refs
            owned = {k['value'] for ref in refs for k in ref.get('keys', []) if k.get('type') == 'Submodel'}
            for item in items:
                entity_name = (item.get("query") or {}).get("entity", "")
                if not item.get("selected", True):
                    skipped.append({"entity": entity_name, "reason": "not selected"})
                    continue
                if item.get("aas_id") and item["aas_id"] != shell['id']:
                    failures.append({"entity": entity_name, "error": "Plan belongs to a different AAS"})
                    continue
                if skip_warnings and (item.get("warnings") or (item.get("decision") or {}).get("warnings")):
                    skipped.append({"entity": entity_name, "reason": "decision has warnings"})
                    continue
                chosen = item.get("chosen_candidates") or [item.get("chosen_candidate") or {}]
                for candidate in chosen:
                    try:
                        query = item.get("query") or {}
                        entity = Entity.parse(query)
                        typed = item.get("decision")
                        if typed and typed.get("action") == "skip":
                            skipped.append({"entity": entity_name, "reason": typed.get("reason", "no match")})
                            continue
                        path = target_path(candidate, owned)
                        live = await client.get(path)
                        kind = live.get("modelType")
                        if typed:
                            known = {f.name for f in fields(Decision)}
                            decision = Decision(**{k: v for k, v in typed.items() if k in known})
                            if decision.path != candidate.get("semantic_path"):
                                raise ValueError("Decision path and selected candidate disagree")
                            if decision.value != entity.value:
                                raise ValueError("Query value and typed decision disagree; review the typed plan")
                        else:
                            action = "create" if kind in CONTAINER_TYPES + ("Submodel",) else "write"
                            spec = infer_element_type(entity)
                            if action == "create":
                                spec, _ = resolve_spec(entity, Candidate(path, live.get('idShort'), kind, 0., None,
                                    type_value_list_element=live.get('typeValueListElement')), None)
                            decision = Decision(entity.entity, entity.value, entity.unit, entity.source, action,
                                "hosted" if action == "create" else "auto", "reviewed legacy matching result", path=path,
                                model_type=spec.model_type if action == "create" else kind,
                                container_type=kind if action == "create" else None,
                                value_type=spec.value_type, id_short=None if kind == "SubmodelElementList" else safe_id_short(entity.entity),
                                min_value=spec.min_value, max_value=spec.max_value, content_type=spec.content_type)
                        if decision.value is None:
                            raise ValueError("No value to apply")
                        if decision.action == "write":
                            if kind != decision.model_type:
                                raise ValueError("Target type changed since matching; run the match again")
                            payload = value_for_write(decision)
                            if not dry_run:
                                await client.patch(path + "/$value", data=json.dumps(payload), reraise=True)
                            applied.append({"entity": entity_name, "api": path, "written_value": payload})
                        elif decision.action == "create":
                            if kind not in CONTAINER_TYPES + ("Submodel",):
                                raise ValueError("A new element requires a collection, list or submodel host")
                            if decision.container_type and decision.container_type != kind:
                                raise ValueError("Host type changed since matching")
                            if kind == "SubmodelElementList":
                                if live.get("typeValueListElement") != decision.model_type:
                                    raise ValueError("New element violates the list's declared child type")
                                decision.id_short = None
                                if live.get("valueTypeListElement"):
                                    decision.value_type = live["valueTypeListElement"]
                            payload = element_for_create(decision)
                            if kind == "SubmodelElementList" and live.get("semanticIdListElement"):
                                payload["semanticId"] = live["semanticIdListElement"]
                            post_path = path + "/submodel-elements" if kind == "Submodel" else path
                            if not dry_run:
                                await client.post(post_path, data=payload, reraise=True)
                            created.append({"entity": entity_name, "post_url": post_path, "new_element": payload})
                        else:
                            raise ValueError("Unknown decision action")
                    except Exception as exc:
                        failures.append({"entity": entity_name, "error": str(exc)})
            summary = {"aas_idShort": aas_idShort, "dry_run": dry_run, "applied": applied,
                       "created": created, "skipped": skipped, "failures": failures}
            if not dry_run and (applied or created):
                try:
                    snapshot = await aas_loader.get_json(endpoint, shell['id'], TEMP_DIR)
                    await __import__('asyncio').to_thread(aas_loader.aas_json_parser, snapshot)
                except Exception as exc:
                    summary["refresh_error"] = str(exc)
            summary_path = Path(TEMP_DIR) / (Path(match_result_path).stem + "_populating_summary.json")
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            message = f"{'Preview' if dry_run else 'Population'}: {len(applied)} applied, {len(created)} created, {len(skipped)} skipped, {len(failures)} failed. Details: {summary_path}"
            return ToolResult(output=message, error="Some population decisions failed; inspect the summary" if failures else None)
        except Exception as exc:
            return ToolResult(error=f"aas_populate_inputs failed: {exc}")
        finally:
            await client.client.aclose()
