from mcp_aas.tools.base import BaseTool, ToolResult
from mcp_aas.aas_utils.basyx_client import BasyxApiClient
import os
import json
import re
import unicodedata
from typing import Optional
from mcp_aas.resource_manager import TEMP_DIR, DEFAULT_AAS_ENDPOINT
from mcp_aas.aas_utils import aas_loader

IDSHORT_ALLOWED = re.compile(r"[A-Za-z0-9_]*$")
IDSHORT_MUST_START_ALPHA = re.compile(r"^[A-Za-z].*$")

def make_valid_idshort(raw: str, *, prefix_if_needed: str = "X",
                       min_len: int = 1, max_len: Optional[int] = 128) -> str:
    if raw is None:
        raw = ""
    s = str(raw).strip()

    # Transliterate to ASCII
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")

    # Replace disallowed chars with underscore
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)

    # Collapse multiple underscores
    s = re.sub(r"_+", "_", s).strip("_")

    # Ensure starts with a letter
    if not s or not IDSHORT_MUST_START_ALPHA.match(s):
        s = (prefix_if_needed + s).lstrip("_")

    # Enforce min/max length
    if len(s) < min_len:
        s = s + ("_" * (min_len - len(s)))
    if max_len is not None and len(s) > max_len:
        s = s[:max_len]

    # Final hard validation
    if not IDSHORT_ALLOWED.fullmatch(s):
        raise ValueError(f"idShort contains invalid characters after normalization: {s!r}")
    if not IDSHORT_MUST_START_ALPHA.match(s):
        raise ValueError(f"idShort must start with a letter after normalization: {s!r}")

    return s


def _ensure_ml_value(current, new_text: str, lang: str = "en"):
    lang = (lang or "en").lower()
    if not isinstance(current, list):
        return [{"language": lang, "text": str(new_text or "")}]
    # normalize language keys to lowercase, update if exists
    updated = False
    out = []
    for item in current:
        if not isinstance(item, dict):
            continue
        il = str(item.get("language", "")).lower()
        it = str(item.get("text", ""))
        if il == lang:
            out.append({"language": lang, "text": str(new_text or "")})
            updated = True
        else:
            out.append({"language": il, "text": it})
    if not updated:
        out.append({"language": lang, "text": str(new_text or "")})
    return out


class AASPopulateInputs(BaseTool):
    name: str = "aas_populate_inputs"
    description: str = (
        "populates the value of AAS properties according to the AAS-Input entity matching results. If match_result_path does not exist, please call aas_match_inputs() first."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "endpoint": {
                "type": "string",
                "description": f"The base URL of the AAS server. Optional. (default: {DEFAULT_AAS_ENDPOINT})"
            },
            "match_result_path": {
                "type": "string",
                "description": "The path of the AAS-Input entity matching result file."
            },
            "aas_idShort": {
                "type": "string",
                "description": "The idShort of the AAS that to be populated according to the input values."
            }
        },
        "required": ["match_result_path", "aas_idShort"]
    }

    async def execute(self, match_result_path: str, aas_idShort: str, endpoint: Optional[str] = None, **kwargs) -> ToolResult:
        endpoint = endpoint or DEFAULT_AAS_ENDPOINT

        if not os.path.exists(match_result_path):
            return ToolResult(
                error=f"Matching result file not found at {match_result_path}. "
                      f"Please call aas_match_inputs() first to generate it."
            )

        try:
            with open(match_result_path, "r", encoding="utf-8") as f:
                match_results = json.load(f)
        except Exception as e:
            return ToolResult(error=f"Failed to load match result file: {e}")

        def _str_or_empty(v):
            return "" if v is None else str(v)

        client = BasyxApiClient(
            endpoint,
            headers={"accept": "application/json", "Content-Type": "application/json"},
        )

        applied = []    # PATCHed existing properties
        created = []    # POSTed new elements
        skipped = []    # could not process

        for item in match_results:
            # Check if user selected this match (default to True for backward compatibility)
            if not item.get("selected", True):
                continue

            query = item.get("query") or {}
            value = query.get("value")
            entity = _str_or_empty(query.get("entity"))
            description_text = _str_or_empty(query.get("description"))

            # Handle multi-select format (chosen_candidates array) or single select (chosen_candidate)
            chosen_candidates = item.get("chosen_candidates", [])
            if not chosen_candidates:
                # Fall back to single chosen_candidate for backward compatibility
                chosen = item.get("chosen_candidate") or {}
                if chosen and chosen.get("API_path"):
                    chosen_candidates = [chosen]

            if not chosen_candidates:
                skipped.append({"entity": entity, "reason": "no chosen candidates"})
                continue

            # Process each chosen candidate (supports multi-select)
            for chosen in chosen_candidates:
                ctype = _str_or_empty(chosen.get("type"))
                api_path = chosen.get("API_path") or chosen.get("apiPath")
                if not api_path:
                    skipped.append({"entity": entity, "reason": f"missing API_path for {chosen.get('idShort', 'unknown')}"})
                    continue

                try:
                    if ctype == "Submodel":
                        # Create a new SubmodelElement directly under the submodel
                        # root. The candidate's API_path is /submodels/{id}; root
                        # creation targets its /submodel-elements sub-resource.
                        # (The SMC/SML branch below instead POSTs to the element
                        # path so the child lands inside that collection.)
                        try:
                            id_short_safe = make_valid_idshort(entity)
                        except Exception:
                            id_short_safe = entity

                        root_post_url = api_path.rstrip("/") + "/submodel-elements"
                        root_payload = {
                            "modelType": "Property",
                            "idShort": id_short_safe,
                            "valueType": "xs:string",
                            "value": _str_or_empty(value),
                            "description": [
                                {"language": "en", "text": _str_or_empty(description_text)}
                            ],
                        }

                        await client.post(root_post_url, data=root_payload)
                        created.append({
                            "entity": entity,
                            "post_url": root_post_url,
                            "new_element": root_payload,
                        })
                    elif ctype not in ("SubmodelElementCollection", "SubmodelElementList"):
                        elem = await client.get(api_path)
                        model_type = (elem or {}).get("modelType") or ctype  # trust server if present
                        value_api = api_path.rstrip("/") + "/$value"

                        try:
                            if model_type == "MultiLanguageProperty":
                                current_val = elem.get("value") if isinstance(elem, dict) else None
                                elem["value"] = _ensure_ml_value(current_val, _str_or_empty(value), lang="en")
                                await client.put(api_path, data=elem)

                                applied.append({
                                    "entity": entity,
                                    "api": api_path,
                                    "written_value": elem["value"],
                                    "target_type": model_type,
                                    "idShort": elem.get("idShort"),
                                    "method": "PUT",
                                })

                            else:
                                current_val = elem.get("value") if isinstance(elem, dict) else None
                                elem["value"] = _str_or_empty(value)
                                await client.put(api_path, data=elem)

                                applied.append({
                                    "entity": entity,
                                    "api": value_api,
                                    "written_value": _str_or_empty(value),
                                    "target_type": model_type or "Unknown",
                                    "idShort": chosen.get("idShort"),
                                })

                        except Exception as e:
                            skipped.append({
                                "entity": entity,
                                "api_path": value_api,
                                "target_type": model_type,
                                "error": str(e),
                            })
                    else:
                        # --- Create new submodel element at the submodel root:
                        # POST /submodels/{submodelIdentifier}/submodel-elements
                        post_url = api_path+'?level=deep&extent=withoutBlobValue'
                        try:
                            id_short_safe = make_valid_idshort(entity)
                        except Exception as e:
                            id_short_safe = entity

                        new_element_payload = {
                            "modelType": "Property",
                            "idShort": id_short_safe,
                            "valueType": "xs:string",
                            "value": _str_or_empty(value),
                            "description": [
                                {"language": "en", "text": _str_or_empty(description_text)}
                            ]
                        }

                        await client.post(post_url, data=new_element_payload)
                        created.append({
                            "entity": entity,
                            "post_url": post_url,
                            "new_element": new_element_payload,
                        })

                except Exception as e:
                    skipped.append(
                        {
                            "entity": entity,
                            "api_path": api_path,
                            "target_type": ctype,
                            "error": str(e),
                        }
                    )

        summary = {
            "aas_idShort": aas_idShort,
            "endpoint": endpoint,
            "applied": applied,
            "created": created,
            "skipped": skipped,
        }

        # Refresh exported artifacts from BaSyx so temp snapshots can reflect
        # post-population AAS state rather than creation-time files.
        refreshed_json_path = None
        refreshed_aasx_path = None
        try:
            shells = await client.get_shells()
            aas_id = None
            if isinstance(shells, list):
                for shell in shells:
                    if shell.get("idShort") == aas_idShort:
                        aas_id = shell.get("id")
                        break

            if aas_id:
                refreshed_json_path = await aas_loader.get_json(
                    endpoint=endpoint,
                    aas_id=aas_id,
                    base_dir=TEMP_DIR,
                )
                try:
                    refreshed_aasx_path = await aas_loader.get_aasx(
                        endpoint=endpoint,
                        aas_id=aas_id,
                        base_dir=TEMP_DIR,
                    )
                except Exception:
                    refreshed_aasx_path = None
        except Exception:
            refreshed_json_path = None
            refreshed_aasx_path = None

        summary["refreshed_json_path"] = refreshed_json_path
        summary["refreshed_aasx_path"] = refreshed_aasx_path

        # Ensure TEMP_DIR exists
        os.makedirs(TEMP_DIR, exist_ok=True)

        # Prepare summary filename
        # Extract base name from match_result_path (e.g., "DOC_battery2_matching_result.json" -> "DOC_battery2")
        match_base = os.path.basename(match_result_path)
        
        # Remove extension
        if match_base.endswith(".json"):
            match_base = match_base[:-5]
            
        # Recursive stripping of known suffixes to handle combinations like "_matching_result_edit"
        # We loop until no more changes to ensure order doesn't matter as much
        # Suffixes to strip: _edit, _matching_result
        # Also maybe _matching_results (plural)? Just in case.
        
        changed = True
        while changed:
            changed = False
            if match_base.endswith("_edit"):
                match_base = match_base[:-5]
                changed = True
            elif match_base.endswith("_matching_result"):
                match_base = match_base[:-16]
                changed = True
            elif match_base.endswith("_matching_results"):
                match_base = match_base[:-17]
                changed = True
            
        summary_path = os.path.join(TEMP_DIR, f"{match_base}_populating_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        applied_n = len(applied)
        created_n = len(created)
        skipped_n = len(skipped)

        return ToolResult(
            output=(
                f"For {aas_idShort} at {endpoint}: "
                f"{applied_n} applied, {created_n} created, {skipped_n} skipped. "
                f"Details saved to {summary_path}; "
                f"refreshed_json={refreshed_json_path or 'n/a'}; "
                f"refreshed_aasx={refreshed_aasx_path or 'n/a'}"
            )
        )




