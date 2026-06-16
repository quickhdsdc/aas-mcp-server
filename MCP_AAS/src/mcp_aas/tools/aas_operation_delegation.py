import base64
import json
import httpx
from typing import Dict, Any, List, Optional
from urllib.parse import urlsplit, urlunsplit
from mcp_aas.tools.base import BaseTool, ToolResult
from mcp_aas.resource_manager import DEFAULT_AAS_ENDPOINT, DEFAULT_INTERNAL_HOST


LOCAL_MOCK_SERVICE_HOSTS = {
    "agv-fleet-service",
    "assembly-controller-service",
    "battery-test-service",
    "mes-service",
    "printer-service",
    "repair-station-service",
    "robot-controller-service",
    "robot-vision-service",
    "scheduler-service",
}

class AASOperationDelegation(BaseTool):
    name: str = "aas_operation_delegation"
    description: str = (
        "Delegate an external tool invocation through AAS operation modeling. "
        "If you are looking for a domain function, but not in the tool list, you may use delegation to find it. "
        "sme_path is the path to the Operation element in the Submodel, e.g. 'DelegatedServices.MyOperation'"
        "Use a FLAT inputArguments dict {idShort: value}, e.g. {'order_id': 'ORD-765', 'required_cell_count': 145} "
        "The tool wraps values into BaSyx Property payloads internally — do NOT pre-wrap them as "
        "{'value': {'modelType': 'Property', ...}} and do NOT pass a list. "
        "If required format of inputArguments is not clear, you may use aas_describe_property for this Operation element "
        "— note that aas_describe_property returns the Operation model (declaration), not the call format."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "endpoint": {
                "type": "string",
                "description": f"The base URL of the AAS server. Optional. (default: {DEFAULT_AAS_ENDPOINT})"
            },
            "sm_id": {
                "type": "string",
                "description": "Submodel ID, which contains the underlying Operation, e.g. https://fraunhoferIPA/ids/sm/services_delegated_assemblyshop"
            },
            "sme_path": {
                "type": "string",
                "description": "Submodel element path, e.g. 'DelegatedServices.MyOperation', which starts at the first SMC level without aas or submodel idShort."
            },
            "inputArguments": {
                "type": "object",
                "description": (
                    "Optional. A FLAT dictionary mapping each input idShort to its raw value. "
                    "Leave empty or omit if the delegated operation requires no inputs. "
                    "Correct example: {'material': 'Ti-6Al-4V', 'power': 350.0}. "
                    "Incorrect (do NOT do this): "
                    "{'material': {'value': {'modelType': 'Property', 'idShort': 'material', 'value': 'Ti-6Al-4V'}}} "
                    "or a list [{'value': {...}}, ...]. The tool wraps the flat values into BaSyx Property objects automatically."
                )
            }
        },
        "required": [
            "sm_id",
            "sme_path"
        ],
        "additionalProperties": False
    }

    def _encode_id(self, id_str: str) -> str:
        return base64.urlsafe_b64encode(bytes(id_str, 'utf-8')).decode('ascii')

    def _infer_value_type(self, val: Any) -> str:
        if isinstance(val, int):
            return "xs:integer"
        elif isinstance(val, float):
            return "xs:double"
        elif isinstance(val, bool):
            return "xs:boolean"
        return "xs:string"

    def _normalize_input_arguments(self, input_args: Any) -> Dict[str, Any]:
        if input_args is None:
            return {}

        def _unwrap_property(maybe_prop: Any) -> Any:
            if not isinstance(maybe_prop, dict):
                return maybe_prop
            # Case: {"value": {"modelType":"Property","idShort":..,"value":..}}
            inner = maybe_prop.get("value")
            if isinstance(inner, dict) and ("idShort" in inner or "modelType" in inner):
                return inner.get("value", inner)
            # Case: {"modelType":"Property","idShort":..,"value":..}
            if "modelType" in maybe_prop and "value" in maybe_prop and "idShort" in maybe_prop:
                return maybe_prop.get("value")
            return maybe_prop

        if isinstance(input_args, list):
            flat: Dict[str, Any] = {}
            for entry in input_args:
                if not isinstance(entry, dict):
                    continue
                inner = entry.get("value") if isinstance(entry.get("value"), dict) else entry
                id_short = inner.get("idShort") if isinstance(inner, dict) else None
                if id_short:
                    flat[id_short] = inner.get("value")
            return flat

        if isinstance(input_args, dict):
            return {k: _unwrap_property(v) for k, v in input_args.items()}

        return {}

    def _normalize_local_mock_service_url(self, delegated_url: str) -> str:
        parts = urlsplit(delegated_url)
        hostname = parts.hostname
        if not hostname:
            return delegated_url

        if hostname == "host.docker.internal" or hostname == "localhost" or hostname == "127.0.0.1" or hostname in LOCAL_MOCK_SERVICE_HOSTS:
            port = f":{parts.port}" if parts.port else ""
            auth = ""
            if parts.username:
                auth = parts.username
                if parts.password:
                    auth += f":{parts.password}"
                auth += "@"
            # Use the configured internal host instead of hardcoded localhost
            netloc = f"{auth}{DEFAULT_INTERNAL_HOST}{port}"
            return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))

        return delegated_url
        
    # Tool-parameter names that must NEVER be treated as operation inputs even
    # if the agent passes them at the top level alongside real inputs.
    _RESERVED_TOP_LEVEL_KEYS = {"endpoint", "sm_id", "sme_path", "inputArguments"}

    async def execute(
        self,
        sm_id: str,
        sme_path: str,
        endpoint: Optional[str] = None,
        inputArguments: Dict[str, Any] = None,
        **kwargs
    ) -> ToolResult:
        endpoint = endpoint or DEFAULT_AAS_ENDPOINT

        # Guard: sme_path must be a plain dotted element path, never a function
        # call. Agents sometimes write "Op(arg=value)" directly into sme_path
        # instead of splitting args into inputArguments. Catch this locally so
        # the agent gets a directive error instead of a generic 404 from BaSyx.
        if sme_path and any(ch in sme_path for ch in "()="):
            return ToolResult(error=(
                f"Invalid sme_path '{sme_path}'. sme_path must be the plain "
                f"dotted element path (e.g. 'DelegatedServices.MyOperation'), "
                f"without arguments or parentheses. Pass call arguments via "
                f"inputArguments as a flat {{idShort: value}} dict."
            ))

        # Tolerate common agent mis-formats: list of BaSyx entries, or dict of
        # pre-wrapped BaSyx Properties. Normalize to a flat {idShort: value} dict.
        inputArguments = self._normalize_input_arguments(inputArguments)
        # Agents often pass operation inputs at the TOP LEVEL alongside
        # endpoint/sm_id/sme_path (e.g. jobs, schedule_request_id) instead of
        # wrapping them under `inputArguments`. Fold those extras in so the
        # delegated service sees them. Explicit `inputArguments` wins on key
        # collision.
        extras = {k: v for k, v in kwargs.items() if k not in self._RESERVED_TOP_LEVEL_KEYS}
        if extras:
            merged = dict(extras)
            merged.update(inputArguments)  # explicit inputArguments override
            inputArguments = merged
            
        try:
            # 1. Fetch the Operation SME from the AAS server to find the delegated endpoint
            b64_sm_id = self._encode_id(sm_id)
            sme_url = f"{endpoint}/submodels/{b64_sm_id}/submodel-elements/{sme_path}"
            
            try:
                async with httpx.AsyncClient() as client:
                    sme_resp = await client.get(sme_url, timeout=10.0)
            except httpx.ConnectError as ce:
                return ToolResult(
                    error=f"AAS mock delegation failed to connect to AAS server at {sme_url}. Error: {ce}"
                )
                
            if sme_resp.status_code >= 400:
                return ToolResult(
                    error=f"AAS mock delegation failed to fetch SME {sme_path}: {sme_resp.status_code} {sme_resp.text}"
                )
                
            sme_data = sme_resp.json()
            
            # 2. Extract delegated URL from qualifiers
            delegated_url = None
            qualifiers = sme_data.get("qualifiers", [])
            for q in qualifiers:
                if q.get("type") == "invocationDelegation":
                    delegated_url = q.get("value")
                    break
                    
            if not delegated_url:
                return ToolResult(
                    error=f"No qualifier of type 'invocationDelegation' found in AAS operation {sme_path}."
                )

            # Normalize Docker-style mock-service hostnames for local host execution.
            delegated_url = self._normalize_local_mock_service_url(delegated_url)

            # 3. Prepare the invocation payload mimicking BaSyx format
            basyx_input_args = []
            for id_short, value in inputArguments.items():
                basyx_input_args.append({
                    "value": {
                        "modelType": "Property",
                        "valueType": self._infer_value_type(value),
                        "idShort": id_short,
                        "value": str(value)
                    }
                })

            payload = {
                "inputArguments": basyx_input_args,
                "inoutputArguments": [],
                "clientTimeoutDuration": "PT60S"
            }

            # 4. Invoke the delegated service directly
            # Passing params=inputArguments as fallback for simple FastAPI endpoints predicting query params
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(delegated_url, json=payload, params=inputArguments, timeout=60.0)
                    if response.status_code == 405:
                        # Fallback to GET for REST endpoints that are strictly mapped as such (e.g. /api/status)
                        response = await client.get(delegated_url, params=inputArguments, timeout=60.0)
            except httpx.ConnectError as ce:
                return ToolResult(
                    error=f"AAS mock delegation failed to connect to the delegated service at {delegated_url}. Error: {ce}"
                )
                
            if response.status_code >= 400:
                return ToolResult(
                    error=f"AAS Operation Delegation failed with status {response.status_code}: {response.text}"
                )
                
            response_data = response.json()
            
            # Format output in a clean way if it's following the standard BaSyx invocation format
            if isinstance(response_data, dict) and "outputArguments" in response_data:
                return ToolResult(output=json.dumps(response_data["outputArguments"], ensure_ascii=False))
            
            return ToolResult(output=json.dumps(response_data, ensure_ascii=False))

        except Exception as e:
            return ToolResult(error=f"AAS operation delegation failed: {str(e)}")
