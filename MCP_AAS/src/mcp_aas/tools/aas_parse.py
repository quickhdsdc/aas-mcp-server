from mcp_aas.tools.base import BaseTool, ToolResult
from mcp_aas.aas_utils import aas_loader
from mcp_aas.aas_utils.basyx_client import BasyxApiClient
from mcp_aas.resource_manager import TEMP_DIR, DEFAULT_AAS_ENDPOINT
from mcp_aas.logger import logger
import httpx
import os
import re
import asyncio
import base64
import binascii
import urllib.parse
from typing import Optional


async def download_aas_attachment(endpoint: str, api_path: str, idShort: str, save_dir: str = TEMP_DIR):
    url = f"{endpoint.rstrip('/')}/{api_path.lstrip('/')}/attachment"

    async with httpx.AsyncClient() as client:
        response = await client.get(url)

        if response.status_code != 200:
            raise RuntimeError(f"Failed to download file: {response.status_code} - {response.text}")

        # Extract filename from Content-Disposition header
        content_disp = response.headers.get("content-disposition", "")
        match = re.search(r'filename="?([^"]+)"?', content_disp)
        full_filename = match.group(1) if match else "downloaded_file"
        # Split on the known idShort and get the part after it
        if idShort in full_filename:
            parts = full_filename.split(f"{idShort}-", maxsplit=1)
            filename = parts[1] if len(parts) == 2 else full_filename
        else:
            filename = full_filename  # fallback if idShort not in name

        # Ensure target directory exists
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, filename)

        # Write file content to disk
        with open(file_path, "wb") as f:
            f.write(response.content)

        return file_path


class AASParse(BaseTool):
    name: str = "aas_parse"
    description: str = (
        "To parse AAS for more detailed information. Only works with AAS id"
        "Given the AAS id and server endpoint, this tool fetches the full AAS and extracts all its properties "
        "into a graph-based knowledge representation."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "endpoint": {
                "type": "string",
                "description": f"The base URL of the AAS server. Optional. (default: {DEFAULT_AAS_ENDPOINT})",
            },
            "aas_id": {
                "type": "string",
                "description": "Canonical AAS identifier, e.g. https://fraunhoferIPA/ids/aas/TestingRepair_097. not the idShort of AAS, e.g. TestingRepair_097",
            },
        },
        "required": ["aas_id"],
    }

    @staticmethod
    def _normalize_endpoint(endpoint: str) -> str:
        return re.sub(r"/(?:shells|submodels)/?$", "", (endpoint or "").strip())

    async def _normalize_aas_id(self, endpoint: str, aas_id_raw: str) -> tuple[Optional[str], str]:
        candidate = (aas_id_raw or "").strip()
        if not candidate:
            return None, "Empty aas_id"

        # Common misuse: passing a Submodel/SMT id to aas_parse.
        # aas_parse expects a shell id (AAS id), not submodel ids.
        lowered = candidate.lower()
        if (
            "/ids/sm/" in lowered
            or lowered.endswith("/submodel")
            or "submodeltemplate" in lowered
            or (lowered.startswith("urn:") and "#" in lowered and "submodel" in lowered)
        ):
            return None, (
                "Input looks like a Submodel id, not an AAS shell id. "
                "Use aas_explore() to get aas_id / aas_idShort, or call aas_from_smt() first and parse its created_aas."
            )

        # Fast path: already a non-shell URL, treat as canonical AAS id.
        if "://" in candidate and "/shells/" not in candidate:
            return candidate, "aas_id already canonical"

        shell_tail = None
        if "/shells/" in candidate:
            shell_tail = candidate.split("/shells/", 1)[1].strip("/")
        elif candidate.startswith("shells/"):
            shell_tail = candidate.split("shells/", 1)[1].strip("/")

        if shell_tail:
            unquoted = urllib.parse.unquote(shell_tail)

            # Try to decode aasIdentifier (UTF8-BASE64-URL as BaSyx API spec).
            try:
                padded = unquoted + ("=" * ((4 - len(unquoted) % 4) % 4))
                decoded = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
                if "://" in decoded:
                    return decoded, "decoded aas_id from /shells/{aasIdentifier}"
            except (ValueError, binascii.Error, UnicodeDecodeError):
                pass

            # If tail isn't base64, it may be idShort.
            candidate = unquoted

        # Resolve idShort -> canonical aas_id using live shell list.
        try:
            client = BasyxApiClient(endpoint)
            shells = await client.get_shells()
            if isinstance(shells, list):
                for shell in shells:
                    if shell.get("idShort") == candidate:
                        shell_id = shell.get("id")
                        if shell_id:
                            return shell_id, f"resolved idShort '{candidate}' to canonical aas_id"
        except Exception:
            # Leave fallback handling to caller diagnostics.
            pass

        # Last resort: keep original input so downstream diagnostics still report clearly.
        return candidate, "unable to normalize via shell path/idShort; using aas_id as provided"

    async def execute(self, aas_id: str, endpoint: Optional[str] = None) -> ToolResult:
        endpoint = self._normalize_endpoint(endpoint or DEFAULT_AAS_ENDPOINT)
        diag = []  # running diagnostic log surfaced in error messages
        diag.append(f"Step 0: endpoint='{endpoint}', aas_id_input='{aas_id}'")
        logger.info(f"[aas_parse] Step 0: endpoint='{endpoint}', aas_id_input='{aas_id}'")

        # ── Step 1: TEMP_DIR ──────────────────────────────────────────────────
        try:
            os.makedirs(TEMP_DIR, exist_ok=True)
            diag.append(f"Step 1 OK: TEMP_DIR ready at '{TEMP_DIR}'")
            logger.info(f"[aas_parse] Step 1 OK: TEMP_DIR ready at '{TEMP_DIR}'")
        except Exception as e:
            diag.append(f"TEMP_DIR creation FAILED: {e}")
            logger.error(f"[aas_parse] Step 1 FAILED: TEMP_DIR creation error: {e}")
            return ToolResult(error="\n".join(diag))

        # ── Step 2: Connectivity probe ────────────────────────────────────────
        try:
            probe_client = BasyxApiClient(endpoint)
            shells_probe = await probe_client.get_shells()
            if not isinstance(shells_probe, list):
                raise RuntimeError("Shells probe did not return a list")
            diag.append(f"Step 2 OK: connectivity probe returned {len(shells_probe)} shell(s)")
            logger.info(f"[aas_parse] Step 2 OK: connectivity probe returned {len(shells_probe)} shell(s)")
        except Exception as e:
            diag.append(f"Server probe FAILED: {type(e).__name__}: {e}")
            logger.error(f"[aas_parse] Step 2 FAILED: {type(e).__name__}: {e}")
            return ToolResult(error="Cannot reach AAS server.\n" + "\n".join(diag))

        # ── Step 2.5: Normalize aas_id into canonical shell id ───────────────
        normalized_aas_id, normalize_note = await self._normalize_aas_id(endpoint, aas_id)
        if not normalized_aas_id:
            diag.append(f"AAS id normalization FAILED: {normalize_note}")
            logger.error(f"[aas_parse] Step 2.5 FAILED: {normalize_note}")
            return ToolResult(error="Invalid aas_id input.\n" + "\n".join(diag))
        diag.append(f"Step 2.5 OK: normalized_aas_id='{normalized_aas_id}' ({normalize_note})")
        logger.info(f"[aas_parse] Step 2.5 OK: normalized_aas_id='{normalized_aas_id}' ({normalize_note})")

        # ── Step 3: Download AASX ─────────────────────────────────────────────
        aasx_filepath = None
        try:
            aasx_candidate = await aas_loader.get_aasx(endpoint=endpoint, aas_id=normalized_aas_id, base_dir=TEMP_DIR)
            diag.append(f"Step 3: AASX candidate path='{aasx_candidate}'")
            if aasx_candidate and os.path.exists(aasx_candidate):
                aasx_filepath = aasx_candidate
                diag.append(f"Step 3 OK: AASX file exists at '{aasx_filepath}'")
                logger.info(f"[aas_parse] Step 3 OK: AASX file exists at '{aasx_filepath}'")
            else:
                diag.append("AASX download produced no local file.")
                logger.warning(f"[aas_parse] Step 3 FAILED: no local AASX file. candidate='{aasx_candidate}'")
        except Exception as e:
            diag.append(f"AASX download FAILED: {type(e).__name__}: {e}. This could be caused by wrong aas_id.")
            logger.error(f"[aas_parse] Step 3 FAILED: {type(e).__name__}: {e}")

        # ── Step 4: Download JSON ─────────────────────────────────────────────
        json_aas_filepath = None
        try:
            json_candidate = await aas_loader.get_json(endpoint=endpoint, aas_id=normalized_aas_id, base_dir=TEMP_DIR)
            diag.append(f"Step 4: JSON candidate path='{json_candidate}'")
            if json_candidate and os.path.exists(json_candidate):
                json_aas_filepath = json_candidate
                diag.append(f"Step 4 OK: JSON file exists at '{json_aas_filepath}'")
                logger.info(f"[aas_parse] Step 4 OK: JSON file exists at '{json_aas_filepath}'")
            else:
                diag.append("JSON download produced no local file.")
                logger.warning(f"[aas_parse] Step 4 FAILED: no local JSON file. candidate='{json_candidate}'")
        except Exception as e:
            diag.append(f"JSON download FAILED: {type(e).__name__}: {e}. This could be caused by wrong aas_id.")
            logger.error(f"[aas_parse] Step 4 FAILED: {type(e).__name__}: {e}")

        if not aasx_filepath and not json_aas_filepath:
            logger.error("[aas_parse] Step 3/4 FAILED: both downloads unavailable")
            return ToolResult(error="Both AASX and JSON downloads failed.\n" + "\n".join(diag))

        # ── Step 5: Parse to graph ────────────────────────────────────────────
        import networkx as nx
        import json

        graph_path = None
        parsed_ok = False

        # JSON path is preferred. The FabOS BaSyx /serialization endpoint
        # returns HTTP 500 for the AASX accept type when submodelIds are
        # included, so get_aasx falls back to a shell-only AASX with no
        # submodel content — aasx_parser then succeeds quietly with a
        # 1-node graph and the upstream caller never knows. Trying JSON
        # first sidesteps that variant entirely and stays correct on the
        # other BaSyx variants where both formats work.
        if json_aas_filepath and os.path.exists(json_aas_filepath):
            try:
                await asyncio.to_thread(aas_loader.aas_json_parser, json_aas_filepath)
                graph_path = json_aas_filepath.replace(".json", "_graph.json")
                parsed_ok = os.path.exists(graph_path)
                diag.append(f"Step 5: JSON parse output graph='{graph_path}', exists={parsed_ok}")
                logger.info(f"[aas_parse] Step 5: JSON parse output graph='{graph_path}', exists={parsed_ok}")
            except Exception as e:
                diag.append(f"JSON parse FAILED: {type(e).__name__}: {e}")
                logger.error(f"[aas_parse] Step 5 JSON parse FAILED: {type(e).__name__}: {e}")

        if not parsed_ok and aasx_filepath and os.path.exists(aasx_filepath):
            try:
                await asyncio.to_thread(aas_loader.aasx_parser, aasx_filepath)
                graph_path = aasx_filepath.replace(".aasx", "_graph.json")
                parsed_ok = os.path.exists(graph_path)
                diag.append(f"Step 5: AASX parse output graph='{graph_path}', exists={parsed_ok}")
                logger.info(f"[aas_parse] Step 5: AASX parse output graph='{graph_path}', exists={parsed_ok}")
            except Exception as e:
                diag.append(f"AASX parse FAILED: {type(e).__name__}: {e}")
                logger.error(f"[aas_parse] Step 5 AASX parse FAILED: {type(e).__name__}: {e}")

        if not parsed_ok or not graph_path:
            logger.error(f"[aas_parse] Step 5 FAILED: parsed_ok={parsed_ok}, graph_path='{graph_path}'")
            return ToolResult(error="Parsing failed — no graph produced.\n" + "\n".join(diag))

        # ── Step 6: Load graph & summarise ────────────────────────────────────
        try:
            def load_graph():
                with open(graph_path, 'r', encoding='utf-8') as f:
                    return nx.node_link_graph(json.load(f))
            G = await asyncio.to_thread(load_graph)
        except Exception as e:
            diag.append(f"Graph load FAILED: {type(e).__name__}: {e}")
            return ToolResult(error="Graph load failed.\n" + "\n".join(diag))

        # Render the AAS structure as an indented hierarchy. We walk the graph
        # edges (which correctly encode parent-child links across both '/'
        # submodel boundaries and '.' SMC-to-child boundaries) rather than
        # relying on string-path heuristics. Each node gets a short type tag
        # so the agent can read structure at a glance and skip extra
        # describe-property round trips.
        TYPE_TAG = {
            "AssetAdministrationShell": "AAS",
            "Submodel": "SM",
            "SubmodelElementCollection": "SMC",
            "SubmodelElementList": "SML",
            "Property": "SME",
            "MultiLanguageProperty": "SME",
            "Range": "SME",
            "ReferenceElement": "SME",
            "Entity": "SME",
            "Blob": "Blob",
            "File": "File",
            "RelationshipElement": "Rel",
            "AnnotatedRelationshipElement": "Rel",
            "Operation": "Op",
            "Capability": "Cap",
        }

        def _short_idshort(nid: str, data: dict) -> str:
            ids = data.get("idShort")
            if ids:
                return ids
            tail = nid.rsplit("/", 1)[-1]
            return tail.rsplit(".", 1)[-1]

        tree_lines: list = []
        visited: set = set()

        def _visit(node: str, depth: int) -> None:
            if node in visited:
                return
            visited.add(node)
            data = G.nodes[node]
            tag = TYPE_TAG.get(data.get("type") or "", data.get("type") or "?")
            tree_lines.append(f"{'  ' * depth}{tag} {_short_idshort(node, data)}")
            for child in sorted(G.successors(node)):
                _visit(child, depth + 1)

        # Roots = nodes with no incoming edges; usually one shell, but render
        # any orphans too so nothing is silently dropped.
        roots = sorted(n for n in G.nodes() if G.in_degree(n) == 0)
        for r in roots:
            _visit(r, 0)
        # Any disconnected nodes (shouldn't happen, but guard anyway)
        for n in sorted(G.nodes()):
            if n not in visited:
                _visit(n, 0)
        id_short_list_str = "\n".join(tree_lines)

        # ── Step 7: File attachments (optional) ───────────────────────────────
        file_nodes = [
            (node_id, data) for node_id, data in G.nodes(data=True)
            if data.get("type") == "File"
        ]

        if file_nodes:
            downloaded: list[tuple[str, str]] = []
            failures: list[tuple[str, str]] = []

            for node_id, node_data in file_nodes:
                api_path = node_data.get("API_path")
                id_short = node_data.get("idShort") or ""
                if not api_path:
                    continue
                try:
                    local_path = await download_aas_attachment(
                        endpoint=endpoint,
                        idShort=id_short,
                        api_path=api_path,
                    )
                    downloaded.append((id_short or api_path, str(local_path)))
                except Exception as e:
                    failures.append((id_short or api_path, str(e)))

            parts = [
                f"AAS parsed and saved as graph at {graph_path}.\n"
                f"Structure (AAS = shell, SM = submodel, SMC = collection, SME = property, Op = operation, Rel = relationship):\n"
                f"{id_short_list_str}",
            ]
            if downloaded:
                if len(downloaded) == 1:
                    parts.append(f"Attachment downloaded to: {downloaded[0][1]}")
                else:
                    parts.append(
                        "Attachments downloaded:\n" +
                        "\n".join(f"- {k}: {p}" for k, p in downloaded)
                    )
            return ToolResult(output="\n\n".join(parts))
        else:
            return ToolResult(
                output=f"AAS parsed and saved as graph at {graph_path}.\n"
                       f"Structure (AAS = shell, SM = submodel, SMC = collection, SME = property, Op = operation, Rel = relationship):\n"
                       f"{id_short_list_str}"
            )
