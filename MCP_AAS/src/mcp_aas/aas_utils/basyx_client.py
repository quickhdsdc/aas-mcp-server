from typing import List
import httpx
from pydantic_core import ValidationError
import base64
from mcp_aas.logger import logger
import os
import urllib
import json
from basyx.aas.model import AssetAdministrationShell, Submodel
from basyx.aas.adapter.json import AASToJsonEncoder

def encode_id(id: str):
    return base64.urlsafe_b64encode(bytes(id, 'utf-8')).decode('ascii')


def decode_id(id: str):
    id_dec = base64.urlsafe_b64decode(id).decode('ascii')
    return id_dec


class BasyxApiClient:


    def __init__(self, base_url, auth_token=None, headers={}):

        self.base_url = base_url.rstrip('/')
        self.auth_token = auth_token
        self.headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else headers
        # In internal deployments a self-signed cert is common on the AAS endpoint.
        # Set AAS_SSL_VERIFY=false to disable TLS cert verification for these calls.
        ssl_verify_raw = os.getenv("AAS_SSL_VERIFY", "false").strip().lower()
        ssl_verify = ssl_verify_raw in ("1", "true", "yes", "on")
        self.client = httpx.AsyncClient(headers=self.headers, verify=ssl_verify, follow_redirects=True)

    def _build_url(self, path: str, base_url: str = None) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        base = (base_url or self.base_url).rstrip('/')
        return f"{base}/{path.lstrip('/')}"


    async def get_shell(self, aas_id: str):

        response = await self.get(f"/shells/{encode_id(aas_id)}")
        return response
    

    async def get_shells(self):

        response = await self.get("/shells")
        shells = response.get('result')
        # logger.info(f"shells on the server: {shells}")
        return shells
    

    async def add_shell(self, shell):
        try:
            _url = self.base_url + "/shells"
            # logger.info(f"Posting AAS '{shell.get('idShort')}' at '{_url}'")
            http_response = await self.post(_url, data=shell)
            logger.info(f"Response: {http_response}")
        except Exception as e:
            logger.error(f"Error during AAS registration: {e}")
            raise e


    async def delete_shell(self, shell_id):
        
        _url = self.base_url + f"/shells/{encode_id(shell_id)}"
        logger.info(f"Deleting AAS '{shell_id}' at '{_url}'")
        try:
            http_response = await self.client.delete(_url)
            logger.info(f"Response: {http_response}")
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error occurred: {e}")
            raise e
        except httpx.RequestError as e:
            logger.error(f"Request error occurred: {e}")
            raise e
        except Exception as e:
            logger.error(f"An error occurred while deleting '{shell_id}': {e}")
            raise e


    async def delete_submodel(self, submodel_id):
        
        _url = self.base_url + f"/submodels/{encode_id(submodel_id)}"
        logger.info(f"Deleting Submodel '{submodel_id}' at '{_url}'")
        try:
            http_response = await self.client.delete(_url)
            logger.info(f"Response: {http_response}")
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error occurred: {e}")
            raise e
        except httpx.RequestError as e:
            logger.error(f"Request error occurred: {e}")
            raise e
        except Exception as e:
            logger.error(f"An error occurred while deleting '{submodel_id}': {e}")
            raise e


    async def get_submodels(self, submodel_ids : List[str]):

        submodels = []
        for submodel_id in submodel_ids:
            try:
                response = await self.get_submodel(submodel_id)
                submodels.append(response)
            except Exception as e:
                logger.error(f"An error occurred while fetching submodels: {e}")
        return submodels


    async def get_submodel(self, submodel_id):

        _url = f"/submodels/{encode_id(submodel_id)}"
        logger.info(f"Getting Submodel '{submodel_id}' at '{_url}'")
        try:
            submodel = await self.get(_url)
        except Exception as e:
            logger.error(f"An error occurred while fetching submodel '{submodel_id}': {e}")
            raise e # Raise here too? Yes probably.
        return submodel


    async def submodel_exists(self, submodel_id: str) -> bool:
        _path = f"/submodels/{encode_id(submodel_id)}"
        _url = self._build_url(_path)
        response = await self.client.get(_url)
        if response.status_code == 404:
            return False
        response.raise_for_status()
        return True


    async def add_submodel(self, submodel):
        try:
            _url = f"{self.base_url}/submodels"
            logger.info(f"Posting Submodel '{submodel.get('idShort')}' at '{_url}'")
            http_response = await self.post(_url, data=submodel)
            logger.info(f"Response: {http_response}")
        except Exception as e:
            logger.error(f"Error during submodel registration: {e}")
            raise e


    async def append_submodel(self, submodel, aas_id):

        try:
            logger.info(f"Appending Submodel '{submodel.get('idShort')}' to AAS '{aas_id}'")
            _url = f"{self.base_url}/submodels"
            http_response = await self.post(_url, data=submodel)
            logger.info(f"Create Response: {http_response}")
            _url = f"{self.base_url}/shells/{aas_id}/submodel-refs"
            http_response = await self.post(_url, data={
                "type": "ModelReference",
                "keys": [{"value": submodel.get('id'), "type": "Submodel"}]
            })
            logger.info(f"Registering Response: {http_response}")
        except Exception as e:
            logger.error(f"Error during AAS registration: {e}")
            raise e


    async def post_submodel(self, submodel):
        try:
            _url = f"{self.base_url}/submodels"
            logger.info(f"Posting Submodel '{submodel.get('idShort')}' at '{_url}'")
            http_response = await self.post(_url, data=submodel)
            # logger.info(f"Response: {http_response}")
            return http_response
        except Exception as e:
            logger.error(f"Error posting submodel: {e}")
            raise e

    async def create_submodel_reference(self, aas_id, submodel_id):
        try:
            # First format check: URL encoded ID?
            _url = f"{self.base_url}/shells/{encode_id(aas_id)}/submodel-refs"
            
            payload = {
                "type": "ModelReference",
                "keys": [{"type": "Submodel", "value": submodel_id}]
            }
            
            logger.info(f"Creating SM Reference for AAS '{aas_id}' to SM '{submodel_id}' at '{_url}'")
            http_response = await self.post(_url, data=payload)
            # logger.info(f"Response: {http_response}")
            return http_response
        except Exception as e:
            logger.error(f"Error creating submodel reference: {e}")
            raise e

    async def invoke_operation(self, submodel_id: str, operation_path_id_short: str, payload: dict):
        try:
            _url = f"{self.base_url}/submodels/{encode_id(submodel_id)}/submodel-elements/{operation_path_id_short}/invoke"
            logger.info(f"Invoking operation '{operation_path_id_short}' on Submodel '{submodel_id}' at '{_url}'")
            http_response = await self.post(_url, data=payload)
            return http_response
        except Exception as e:
            logger.error(f"Error invoking operation: {e}")
            raise e


    async def register_aas(self, aas: AssetAdministrationShell):
        logger.info("Registering AAS")
        try:
            aas_obj = json.loads(json.dumps(aas, cls=AASToJsonEncoder))
            await self.add_shell(aas_obj)
        except Exception as e:
            logger.error(f"Error during AAS registration: {e}")
            return False
        return True

    async def register_submodel(self, submodel: Submodel):
        logger.info("Registering submodels")
        try:
            # submodel_obj = json.loads(json.dumps(submodel, cls=AASToJsonEncoder))
            await self.add_submodel(submodel)
        except Exception as e:
            logger.error(f"Error during submodel registration: {e}")
            return False
        return True


    async def get(self, path, params=None):
        try:
            _url = self._build_url(path)
            response = await self.client.get(_url, params=params)
            response.raise_for_status()
            try:
                return response.json()
            except ValueError as e:
                _url = str(response.request.url)
                logger.error(f"Invalid JSON response from {_url}: {response.text[:300]}")
                raise e
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error occurred: {e}")
            raise e
        except httpx.RequestError as e:
            logger.error(f"Request error occurred: {e}")
            raise e
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise e

    async def patch(self, path, data, reraise=False):
        try:
            _url = self._build_url(path)
            if isinstance(data, (str, bytes)):
                response = await self.client.patch(_url, content=data)
            else:
                response = await self.client.patch(_url, json=data)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error occurred: {e.response.status_code} --- {e.__class__.__name__}")
            if reraise: raise e
        except httpx.RequestError as e:
            logger.error(f"Request error occurred: {e}")
            if reraise: raise e
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            if reraise: raise e


    async def post(self, path, data, reraise=False):
        try:
            _url = self._build_url(path)
            if isinstance(data, (str, bytes)):
                response = await self.client.post(_url, content=data)
            else:
                response = await self.client.post(_url, json=data)
            response.raise_for_status()
            if not response.text:
                return None
            try:
                return response.json()
            except ValueError:
                # Some endpoints may return 204/empty or non-JSON payloads on success.
                return {"raw": response.text}
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error occurred: {e.response.status_code} --- {e.__class__.__name__}")
            if reraise: raise e
        except httpx.RequestError as e:
            logger.error(f"Request error occurred: {e}")
            if reraise: raise e
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            if reraise: raise e

    async def put(self, path, data, reraise=False):
        try:
            _url = self._build_url(path)
            if isinstance(data, (str, bytes)):
                response = await self.client.put(_url, content=data)
            else:
                response = await self.client.put(_url, json=data)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error occurred: {e.response.status_code} --- {e.__class__.__name__}")
            if reraise: raise e
        except httpx.RequestError as e:
            logger.error(f"Request error occurred: {e}")
            if reraise: raise e
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            if reraise: raise e

    async def _resolve_submodel_ids(self, aas_id: str) -> List[str]:
        try:
            b64_aas_id = urllib.parse.quote(
                base64.urlsafe_b64encode(aas_id.encode()).decode(), safe=""
            )
            refs = await self.get(f"/shells/{b64_aas_id}/submodel-refs")
            all_ids = [
                ref["keys"][0]["value"]
                for ref in (refs or {}).get("result", [])
                if ref.get("keys")
            ]
        except Exception as e:
            logger.warning(
                f"[_resolve_submodel_ids] could not list submodel refs for {aas_id}: {e}"
            )
            return []

        existing: List[str] = []
        for sm_id in all_ids:
            try:
                if await self.submodel_exists(sm_id):
                    existing.append(sm_id)
                else:
                    logger.warning(
                        f"[_resolve_submodel_ids] dropping dangling ref (404): {sm_id}"
                    )
            except Exception as e:
                logger.warning(
                    f"[_resolve_submodel_ids] existence probe failed for {sm_id}: {e}"
                )
        return existing

    async def _serialize_aas(
        self,
        aas_id: str,
        submodel_ids: List[str],
        accept: str,
    ) -> "httpx.Response":
        b64_aas_id = urllib.parse.quote(
            base64.urlsafe_b64encode(aas_id.encode()).decode(), safe=""
        )
        url = self._build_url("/serialization")
        headers = {"Accept": accept}

        if submodel_ids:
            params = [
                ("aasIds", b64_aas_id),
                ("includeConceptDescriptions", "true"),
            ]
            for sm_id in submodel_ids:
                b64 = urllib.parse.quote(
                    base64.urlsafe_b64encode(sm_id.encode()).decode(), safe=""
                )
                params.append(("submodelIds", b64))

            logger.info(
                f"[_serialize_aas] GET {url} aas_id={aas_id} "
                f"submodel_count={len(submodel_ids)}"
            )
            try:
                response = await self.client.get(url, params=params, headers=headers)
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as e:
                if 500 <= e.response.status_code < 600:
                    logger.warning(
                        f"[_serialize_aas] {e.response.status_code} with submodelIds; "
                        f"retrying without (some BaSyx setups reject the param)"
                    )
                else:
                    raise

        params = {
            "aasIds": b64_aas_id,
            "includeConceptDescriptions": "true",
        }
        logger.info(f"[_serialize_aas] GET {url} aas_id={aas_id} (shell-only)")
        response = await self.client.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response

    async def download_aas_package(self, aas_id: str, submodel_ids: List[str], filepath: str):
        try:
            if not submodel_ids:
                submodel_ids = await self._resolve_submodel_ids(aas_id)

            response = await self._serialize_aas(
                aas_id,
                submodel_ids,
                accept="application/asset-administration-shell-package+xml",
            )

            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "wb") as f:
                f.write(response.content)

            logger.info(f"AASX package downloaded to: {filepath}")
            return filepath

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error occurred during AASX download: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during AASX download: {e}")
            raise

    async def download_aas_json(self, aas_id: str, submodel_ids: List[str], filepath: str):
        try:
            if not submodel_ids:
                submodel_ids = await self._resolve_submodel_ids(aas_id)

            response = await self._serialize_aas(
                aas_id,
                submodel_ids,
                accept="application/json",
            )

            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(response.json(), f, indent=2)

            logger.info(f"AAS JSON downloaded to: {filepath}")
            return filepath

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error occurred during JSON download: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during JSON download: {e}")
            raise


    async def register_submodels(self, submodel: Submodel):
        try:
            await self.client.add_submodel(submodel)
        except Exception as e:
            logger.error(f"Error during submodel registration: {e}")
            return False
        return True

# Example usage
async def main():
    from mcp_aas.resource_manager import DEFAULT_AAS_ENDPOINT
    api_client = BasyxApiClient(DEFAULT_AAS_ENDPOINT)
    response = await api_client.get_shells()
    print(response)



if __name__ == "__main__":
    import asyncio
    asyncio.run(main())