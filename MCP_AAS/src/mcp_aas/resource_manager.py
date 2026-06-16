
import asyncio
import base64
import hashlib
import json
import os
from typing import Any, Dict, List, Optional
from mcp_aas.aas_utils.basyx_client import BasyxApiClient, encode_id
from mcp_aas.logger import logger

# ---------------------------------------------------------------------------
# Defaults — all env-driven constants live in mcp_aas.settings. Re-exported
# here so existing `from mcp_aas.resource_manager import DEFAULT_AAS_ENDPOINT`
# call sites in the vendored tools keep working unchanged.
# ---------------------------------------------------------------------------
from mcp_aas.settings import (  # noqa: E402
    DEFAULT_AAS_ENDPOINT,
    DEFAULT_INTERNAL_HOST,
    MANIFEST_DIR,
    TEMP_DIR,
    RECORDS_DIR,
    ATTACHMENTS_DIR,
)

# In-memory caches (also persisted to disk)
_service_manifest: Dict[str, Any] = {}
_resource_manifest: Dict[str, Any] = {}
# Flat lists used by app.decoding to build M2 V-set enums. Refreshed alongside
# the two manifests in _build_manifests().
_property_idshorts: List[str] = []
_submodel_idshorts: List[str] = []
_lock = asyncio.Lock()

# Change-detection for poll_manifests: hash of the last observed shell-list signature.
_last_shell_signature: Optional[str] = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_service_manifest() -> Dict[str, Any]:
    try:
        path = os.path.join(MANIFEST_DIR, "service_manifest.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load service manifest from disk: {e}")
    return _service_manifest


def get_resource_manifest() -> Dict[str, Any]:
    try:
        path = os.path.join(MANIFEST_DIR, "resource_manifest.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load resource manifest from disk: {e}")
    return _resource_manifest


async def refresh_manifests(endpoint: str = DEFAULT_AAS_ENDPOINT) -> None:
    async with _lock:
        await _build_manifests(endpoint)


async def _compute_shell_signature(endpoint: str) -> Optional[str]:
    try:
        client = BasyxApiClient(endpoint)
        shells = await client.get_shells()
    except Exception as e:
        logger.warning(f"poll_manifests: shell-list probe failed against {endpoint}: {e}")
        return None

    if not isinstance(shells, list):
        return None

    triples = sorted(
        (
            shell.get("id", ""),
            shell.get("idShort", ""),
            (shell.get("assetInformation") or {}).get("assetKind", ""),
        )
        for shell in shells
    )
    payload = json.dumps(triples, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


async def poll_manifests(
    interval_seconds: float,
    endpoint: str = DEFAULT_AAS_ENDPOINT,
) -> None:
    global _last_shell_signature

    if interval_seconds <= 0:
        logger.info("poll_manifests: polling disabled (interval <= 0)")
        return

    # Seed the signature from the current state so the first tick doesn't
    # trigger a redundant refresh right after startup.
    _last_shell_signature = await _compute_shell_signature(endpoint)
    logger.info(
        f"poll_manifests: started, interval={interval_seconds}s, "
        f"endpoint={endpoint}, initial_signature={_last_shell_signature!r}"
    )

    while True:
        try:
            await asyncio.sleep(interval_seconds)
            sig = await _compute_shell_signature(endpoint)
            if sig is None:
                continue
            if sig != _last_shell_signature:
                logger.info(
                    f"poll_manifests: shell signature changed "
                    f"({_last_shell_signature!r} -> {sig!r}); refreshing manifests"
                )
                await refresh_manifests(endpoint)
                _last_shell_signature = sig
        except asyncio.CancelledError:
            logger.info("poll_manifests: cancelled")
            raise
        except Exception as e:
            logger.warning(f"poll_manifests: tick failed, continuing: {e}")


# ---------------------------------------------------------------------------
# Internal: Manifest builders
# ---------------------------------------------------------------------------

async def _build_manifests(endpoint: str) -> None:
    global _service_manifest, _resource_manifest, _property_idshorts, _submodel_idshorts
    client = BasyxApiClient(endpoint)
    prop_idshorts: set = set()
    sm_idshorts: set = set()

    try:
        shells = await client.get_shells()
    except Exception as e:
        logger.error(f"ResourceManager: failed to fetch shells: {e}")
        return

    if not isinstance(shells, list):
        logger.warning("ResourceManager: shells response is not a list")
        return

    new_service: Dict[str, Any] = {}
    new_resource: Dict[str, Any] = {}

    for shell in shells:
        aas_id = shell.get("id", "")
        id_short = shell.get("idShort", "")
        # Use `or {}` rather than the `.get(..., {})` default — BaSyx may return
        # `"assetInformation": null` and dict.get treats None-value as present.
        asset_info = shell.get("assetInformation") or {}
        asset_kind = asset_info.get("assetKind", "Instance")

        if not aas_id or not id_short:
            continue

        # Only index AAS instances, skip blueprints (assetKind: Type)
        if asset_kind == "Type":
            continue

        # Determine blueprint name from idShort (strip instance suffix like _M1)
        blueprint = _infer_blueprint(id_short)

        # Fetch submodel references
        b64_aas = encode_id(aas_id)
        try:
            refs_resp = await client.get(f"/shells/{b64_aas}/submodel-refs")
            sm_refs = refs_resp.get("result", [])
        except Exception as e:
            logger.warning(f"ResourceManager: failed to get SM refs for {id_short}: {e}")
            sm_refs = []

        # Build per-asset entries
        asset_services: Dict[str, Any] = {}
        asset_submodels: Dict[str, Any] = {}

        for ref in sm_refs:
            try:
                sm_id = ref["keys"][0]["value"]
            except (KeyError, IndexError):
                continue

            # Fetch submodel metadata
            try:
                sm = await client.get(f"/submodels/{encode_id(sm_id)}")
            except Exception as e:
                logger.warning(f"ResourceManager: failed to fetch SM {sm_id}: {e}")
                continue

            sm_id_short = sm.get("idShort", "")
            sm_elements = sm.get("submodelElements", [])
            if sm_id_short:
                sm_idshorts.add(sm_id_short)
            # Collect every Property/MultiLanguageProperty idShort under this
            # submodel so app.decoding can offer them as M2 enum values for
            # prop_idShort args.
            _collect_property_idshorts(sm_elements, prop_idshorts)

            # Resource Manifest entry for this submodel
            sm_entry: Dict[str, Any] = {
                "sm_id": sm_id,
                "property_count": _count_properties(sm_elements),
            }

            # Check if this is a delegated services submodel
            is_delegated = (
                "services_delegated" in sm_id_short.lower()
                or "services_delegated" in sm_id.lower()
            )

            if is_delegated:
                operations = _extract_operations(sm_elements, sm_id)
                sm_entry["operations"] = list(operations.keys())
                asset_services.update(operations)

            asset_submodels[sm_id_short] = sm_entry

        # Only add to manifests if there's content
        if asset_submodels:
            new_resource[id_short] = {
                "aas_id": aas_id,
                "blueprint": blueprint,
                "asset_kind": asset_kind,
                "submodels": asset_submodels,
            }

        if asset_services:
            new_service[id_short] = {
                "aas_id": aas_id,
                "blueprint": blueprint,
                "asset_kind": asset_kind,
                "services": asset_services,
            }

    _service_manifest = new_service
    _resource_manifest = new_resource
    _property_idshorts = sorted(prop_idshorts)
    _submodel_idshorts = sorted(sm_idshorts)

    # Persist to disk
    _persist_manifests()

    logger.info(
        f"ResourceManager: refreshed manifests — "
        f"{len(new_resource)} assets, {sum(len(v.get('services', {})) for v in new_service.values())} services"
    )


# ---------------------------------------------------------------------------
# Internal: Operation extraction
# ---------------------------------------------------------------------------

def _collect_property_idshorts(elements: List[dict], out: set) -> None:
    if not isinstance(elements, list):
        return
    for elem in elements:
        if not isinstance(elem, dict):
            continue
        id_short = elem.get("idShort")
        if id_short:
            out.add(id_short)
        model_type = elem.get("modelType", "")
        if model_type == "SubmodelElementCollection":
            _collect_property_idshorts(elem.get("value", []) or [], out)


def get_v_set() -> Dict[str, List[str]]:
    aas_idshorts = sorted(_resource_manifest.keys() | _service_manifest.keys())
    submodel_ids: set = set()
    operation_sme_paths: set = set()
    for asset in _resource_manifest.values():
        for sm_entry in (asset.get("submodels") or {}).values():
            sm_id = sm_entry.get("sm_id")
            if sm_id:
                submodel_ids.add(sm_id)
    for asset in _service_manifest.values():
        for op_entry in (asset.get("services") or {}).values():
            sme_path = op_entry.get("sme_path")
            if sme_path:
                operation_sme_paths.add(sme_path)
    return {
        "aas_idShort": aas_idshorts,
        "submodel_id": sorted(submodel_ids),
        "submodel_idShort": list(_submodel_idshorts),
        "operation_sme_path": sorted(operation_sme_paths),
        "property_idShort": list(_property_idshorts),
    }


def _extract_operations(elements: List[dict], sm_id: str, prefix: str = "") -> Dict[str, Any]:
    ops: Dict[str, Any] = {}

    for elem in elements:
        model_type = elem.get("modelType", "")
        elem_id = elem.get("idShort", "")
        current_path = f"{prefix}.{elem_id}" if prefix else elem_id

        if model_type == "SubmodelElementCollection":
            # Recurse into collections (e.g., "DelegatedServices")
            child_elements = elem.get("value", [])
            if isinstance(child_elements, list):
                ops.update(_extract_operations(child_elements, sm_id, current_path))

        elif model_type == "Operation":
            # Extract input/output variable schemas
            inputs = _extract_variables(elem.get("inputVariables", []))
            outputs = _extract_variables(elem.get("outputVariables", []))

            # Try to extract delegation endpoint from qualifiers
            endpoint = _extract_qualifier_endpoint(elem.get("qualifiers", []))

            ops[elem_id] = {
                "sm_id": sm_id,
                "sme_path": current_path,
                "description": _get_description_text(elem.get("description", [])),
                "endpoint": endpoint,
                "inputs": inputs,
                "outputs": outputs,
            }

    return ops


def _extract_variables(variables: List[dict]) -> List[dict]:
    result = []
    for var in variables:
        value = var.get("value", {})
        if isinstance(value, dict):
            result.append({
                "idShort": value.get("idShort", ""),
                "valueType": value.get("valueType", ""),
                "description": _get_description_text(value.get("description", [])),
            })
    return result


def _extract_qualifier_endpoint(qualifiers: List[dict]) -> Optional[str]:
    for q in qualifiers:
        q_type = q.get("type", "")
        if q_type in ("invocationDelegation", "invocationDelegationUrl"):
            return q.get("value", "")
    return None


# ---------------------------------------------------------------------------
# Internal: Helpers
# ---------------------------------------------------------------------------

def _infer_blueprint(id_short: str) -> str:
    # If the idShort ends with _<something>, strip the suffix
    parts = id_short.rsplit("_", 1)
    if len(parts) == 2 and (parts[1].isdigit() or parts[1].startswith("M") or len(parts[1]) <= 5):
        return parts[0]
    return id_short


def _count_properties(elements: List[dict]) -> int:
    count = 0
    for elem in elements:
        model_type = elem.get("modelType", "")
        if model_type == "SubmodelElementCollection":
            child_elements = elem.get("value", [])
            if isinstance(child_elements, list):
                count += _count_properties(child_elements)
        elif model_type in ("Property", "MultiLanguageProperty", "Blob", "File", "Range"):
            count += 1
    return count


def _get_description_text(descriptions) -> str:
    if not descriptions:
        return ""
    if isinstance(descriptions, list):
        for desc in descriptions:
            if isinstance(desc, dict):
                lang = desc.get("language", "")
                if lang.lower().startswith("en"):
                    return desc.get("text", "")
        # Fallback to first description
        if descriptions and isinstance(descriptions[0], dict):
            return descriptions[0].get("text", "")
    return ""


def _persist_manifests() -> None:
    try:
        os.makedirs(MANIFEST_DIR, exist_ok=True)
        with open(os.path.join(MANIFEST_DIR, "service_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(_service_manifest, f, indent=2, ensure_ascii=False)
        with open(os.path.join(MANIFEST_DIR, "resource_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(_resource_manifest, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"ResourceManager: failed to persist manifests: {e}")
