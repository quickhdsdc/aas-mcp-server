from mcp_aas.tools.base import BaseTool, ToolResult
import os
import json
import logging

logger = logging.getLogger(__name__)
import ast
import httpx
from basyx.aas.model import AssetAdministrationShell, AssetInformation, AssetKind, ModelReference
from mcp_aas.aas_utils.basyx_client import BasyxApiClient, encode_id, decode_id
from mcp_aas.aas_utils import aas_loader
from mcp_aas.resource_manager import TEMP_DIR, DEFAULT_AAS_ENDPOINT

logger = logging.getLogger(__name__)


class AASfromSMT(BaseTool):
    name: str = "aas_from_smt"
    description: str = (
        "Construct a new Asset Administration Shell (AAS) from the given submodel templates (SMT)'s id. "
        "and post the created AAS and submodels to the AAS server at the given endpoint."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "endpoint": {
                "type": "string",
                "description": f"The base URL of the AAS server. Optional. (default: {DEFAULT_AAS_ENDPOINT})"
            },
            "aas_idShort": {
                "type": "string",
                "description": "The ID Short for the AAS to be created (e.g., 'NewProductAAS', 'Motor_XM2000')."
            },
            "list_smt_id": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Submodel template IDs, e.g. ['https://admin-shell.io/ZVEI/TechnicalData/Submodel/1/2']."
            }
        },
        "required": ["aas_idShort", "list_smt_id"]
    }

    @staticmethod
    def extract_list_from_text(text: str):
        start = text.find("[")
        end = text.rfind("]")

        if start != -1 and end != -1 and start < end:
            list_part = text[start:end + 1]  # extract only the [ ... ]
            try:
                return ast.literal_eval(list_part)
            except Exception:
                return []
        return []


    async def execute(self, aas_idShort: str, list_smt_id: list, endpoint: str = None, **kwargs) -> ToolResult:
        endpoint = endpoint or DEFAULT_AAS_ENDPOINT

        try:
            if isinstance(list_smt_id, str):
                try:
                    list_smt_id = self.extract_list_from_text(list_smt_id)
                except Exception as e:
                    return ToolResult(error="smt_csv_path must be a stringified list.")

            try:
                basyx_client = BasyxApiClient(endpoint, headers={
                    "accept": "application/json",
                    "Content-Type": "application/json"
                })
                submodels = await basyx_client.get_submodels(list_smt_id)

            except Exception as e:
                return ToolResult(error=f"Basyx client failure: {str(e)}")

            if not submodels:
                return ToolResult(error="No submodels loaded. Check AAS server.")

            # Check if AAS with this idShort already exists
            existing_shells = await basyx_client.get_shells()
            existing_aas = None
            if isinstance(existing_shells, list):
                for shell in existing_shells:
                    if shell.get("idShort") == aas_idShort:
                        existing_aas = shell
                        break
            
            if existing_aas:
                new_aas_id = existing_aas.get("id")
                logger.info("AAS '%s' already exists with ID '%s'. Updating/Appending submodels.", aas_idShort, new_aas_id)
            else:
                 # Create new AAS
                try:
                    new_aas_id = f'https://fraunhoferIPA/ids/aas/{aas_idShort}'
                    new_aas = AssetAdministrationShell(
                        id_=new_aas_id,
                        id_short=aas_idShort,
                        asset_information=AssetInformation(
                            asset_kind=AssetKind.INSTANCE,
                            global_asset_id=f'https://fraunhoferIPA/id/assets/{aas_idShort}',
                        ),
                    )
                    await basyx_client.register_aas(new_aas)
                except Exception as e:
                     return ToolResult(error=f"Failed to register AAS: {str(e)}")

            new_aas_id_encoded = encode_id(new_aas_id)
            sm_ref_path = f"/shells/{new_aas_id_encoded}/submodel-refs"
            linked = []
            for submodel in submodels:
                sm_id = submodel.get("id")
                if not sm_id:
                    continue
                submodel["kind"] = "Instance"
                if aas_idShort:
                    submodel["id"] = f"{sm_id}_{aas_idShort}"
                    sm_id = submodel["id"]
                await basyx_client.register_submodel(submodel)
                body = {
                    "type": "ModelReference",
                    "keys": [{"type": "Submodel", "value": sm_id}]
                }
                try:
                    await basyx_client.post(sm_ref_path, body)
                    linked.append(sm_id)
                except Exception as e:
                    return ToolResult(error=f"Failed to link submodel {sm_id}: {str(e)}")

            aas_json_filepath = await aas_loader.get_json(endpoint=endpoint, aas_id=new_aas_id, base_dir=TEMP_DIR)
            
            # Also try to save as AASX for robustness
            aas_aasx_filepath = None
            try:
                aas_aasx_filepath = await aas_loader.get_aasx(endpoint=endpoint, aas_id=new_aas_id, base_dir=TEMP_DIR)
            except Exception as e:
                logger.warning("Failed to save AASX after creation: %s", e)

            # Success: summary output
            return ToolResult(output=json.dumps({
                "created_aas": new_aas_id,
                "linked_submodels": linked,
                "aas_json_filepath": aas_json_filepath,
                "aas_aasx_filepath": aas_aasx_filepath
            }, indent=2))

        except Exception as e:
            return ToolResult(error=f"AASfromSMT failed: {str(e)}")




