import base64
import urllib.parse
from mcp_aas.aas_utils.basyx_client import BasyxApiClient
import os
from pyecma376_2 import ZipPackageReader
from basyx.aas import model
from basyx.aas.adapter import aasx
from basyx.aas.adapter.xml import read_aas_xml_file
from basyx.aas.adapter.json import read_aas_json_file
import io
import pandas as pd
import zipfile
import xml.etree.ElementTree as ET
import re
import logging
from dateutil import parser as date_parser


def get_external_related_parts(aasx_filepath, rel_type_filter=None):
    rels_path = "aasx/_rels/aasx-origin.rels"

    with zipfile.ZipFile(aasx_filepath, "r") as zipf:
        if rels_path not in zipf.namelist():
            return []

        with zipf.open(rels_path) as f:
            tree = ET.parse(f)
            root = tree.getroot()

            # Define namespace
            ns = {"rel": "http://schemas.openxmlformats.org/package/2006/relationships"}
            matches = []

            for rel in root.findall("rel:Relationship", ns):
                rel_type = rel.attrib.get("Type")
                target = rel.attrib.get("Target")
                target_mode = rel.attrib.get("TargetMode", "Internal")

                if rel_type_filter is None or rel_type == rel_type_filter:
                    matches.append({
                        "type": rel_type,
                        "target": target,
                        "mode": target_mode
                    })

            return matches


def encode_id(id_str: str) -> str:
    b64_raw = base64.urlsafe_b64encode(id_str.encode()).decode()
    return urllib.parse.quote(b64_raw, safe='')


async def get_aasx(endpoint, aas_id, base_dir):

    client = BasyxApiClient(endpoint)
    b64_aas_id = encode_id(aas_id)
    
    # Fetch the AAS metadata to get idShort
    shell_metadata = await client.get(f"/shells/{b64_aas_id}")
    id_short = shell_metadata.get("idShort")
    if not id_short:
        id_short = aas_id.split("/")[-1]

    os.makedirs(base_dir, exist_ok=True)
    filename = id_short + ".aasx"
    filepath = os.path.join(base_dir, filename)

    # Download package by AAS id only. Passing submodelIds can trigger BaSyx 500 on some setups.
    result = await client.download_aas_package(aas_id, [], filepath)
    if not result or not os.path.exists(filepath):
        raise RuntimeError(
            f"AASX download did not produce local file at '{filepath}' for aas_id '{aas_id}' (endpoint='{endpoint}')"
        )
    print(f"Downloaded AASX file to {filepath}")

    return filepath


async def get_json(endpoint, aas_id, base_dir):

    client = BasyxApiClient(endpoint)
    b64_aas_id = encode_id(aas_id)
    
    # Fetch the AAS metadata to get idShort
    shell_metadata = await client.get(f"/shells/{b64_aas_id}")
    id_short = shell_metadata.get("idShort")
    if not id_short:
        id_short = aas_id.split("/")[-1]

    os.makedirs(base_dir, exist_ok=True)
    filename = id_short + ".json"
    filepath = os.path.join(base_dir, filename)

    # Download JSON by AAS id only. Passing submodelIds can trigger BaSyx 500 on some setups.
    result = await client.download_aas_json(aas_id, [], filepath)
    if not result or not os.path.exists(filepath):
        raise RuntimeError(
            f"JSON download did not produce local file at '{filepath}' for aas_id '{aas_id}' (endpoint='{endpoint}')"
        )
    print(f"Downloaded json file to {filepath}")

    return filepath


def aasx_parser(aasx_filepath: str) -> pd.DataFrame:
    aas_store: model.DictObjectStore[model.Identifiable] = model.DictObjectStore()
    file_store = aasx.DictSupplementaryFileContainer()
    with ZipPackageReader(aasx_filepath) as reader:
        rel_type = "http://admin-shell.io/aasx/relationships/aas-spec"
        related_parts = get_external_related_parts(aasx_filepath, rel_type_filter=rel_type)
        if related_parts:
            for rel in related_parts:
                aas_part = rel.get("target")
                if not aas_part:
                    continue  # skip invalid
                if aas_part.endswith(".xml"):
                    with reader.open_part(aas_part) as p:
                        # Sanitize XML content
                        content = p.read() # bytes
                        sanitized_content = sanitize_aas_xml_values(content)
                        
                        f_pseudo = io.BytesIO(sanitized_content)
                        
                        # Suppress basyx logs during parsing
                        # Use CRITICAL to suppress ERROR logs from failsafe mode
                        logging.getLogger("basyx").setLevel(logging.CRITICAL)
                        try:
                            aas_objs = read_aas_xml_file(f_pseudo, failsafe=True)
                        finally:
                            logging.getLogger("basyx").setLevel(logging.WARNING) 
                        
                        for obj in aas_objs:
                            aas_store.add(obj)
                elif aas_part.endswith(".json"):
                    with reader.open_part(aas_part) as p:
                        # aas_objs = read_aas_json_file(io.TextIOWrapper(p, encoding='utf-8-sig'))
                        
                        import json
                        # Read raw content
                        content = io.TextIOWrapper(p, encoding='utf-8-sig').read()
                        data = json.loads(content)
                        # Sanitize — structural first, then value-level
                        data = _sanitize_aas_structural(data)
                        data = sanitize_aas_json_values(data)
                        
                        # Write back to pseudo-file for basyx
                        f_pseudo = io.StringIO(json.dumps(data))
                        
                        # Use CRITICAL to suppress ERROR logs from failsafe mode
                        logging.getLogger("basyx").setLevel(logging.CRITICAL)
                        try:
                            aas_objs = read_aas_json_file(f_pseudo)
                        finally:
                             logging.getLogger("basyx").setLevel(logging.WARNING)

                        for obj in aas_objs:
                            aas_store.add(obj)
                else:
                    print(f"Unsupported file format: {aas_part}")
    df_aas, G = flatten_aas_object_store(aas_store, return_graph=True)
    
    import networkx as nx
    import json
    graph_path = aasx_filepath.replace(".aasx", "_graph.json")
    with open(graph_path, 'w', encoding='utf-8') as f:
        json.dump(nx.node_link_data(G), f, indent=2)
        
    return df_aas


# AASd-122: first key of an ExternalReference must be one of these.
_GLOBALLY_IDENTIFIABLE_KEY_TYPES = frozenset({"GlobalReference", "FragmentReference"})

# AASd-123: first key of a ModelReference must be an AasIdentifiable type.
_AAS_IDENTIFIABLE_KEY_TYPES = frozenset({
    "AssetAdministrationShell", "ConceptDescription", "Identifiable", "Submodel",
})

# basyx-python-sdk 2.0.0's XSD_TYPE_CLASSES is missing a handful of otherwise
# valid XSD types. Remap each to the nearest supported supertype so the Property
# still deserializes (xs:unsignedInt's 0..4294967295 range fits xs:unsignedLong).
_UNSUPPORTED_XSD_REMAP = {
    "xs:unsignedInt": "xs:unsignedLong",
}


def _sanitize_aas_structural(data):
    if not isinstance(data, dict):
        return data

    # Shell-scoped fixes.
    for shell in data.get("assetAdministrationShells", []) or []:
        if not isinstance(shell, dict):
            continue
        for ref in shell.get("submodels", []) or []:
            if isinstance(ref, dict) and ref.get("type") in (None, "ExternalReference"):
                ref["type"] = "ModelReference"
        ai = shell.get("assetInformation")
        if ai is None:
            shell["assetInformation"] = {
                "assetKind": "Instance",
                "globalAssetId": shell.get("id") or "urn:placeholder",
            }
        elif isinstance(ai, dict):
            if not ai.get("globalAssetId") and not ai.get("specificAssetIds"):
                ai["globalAssetId"] = shell.get("id") or "urn:placeholder"

    # AASd-122 / AASd-123 — full-tree walk; rewrite illegal first-key types.
    def _walk_refs(o):
        if isinstance(o, dict):
            ref_type = o.get("type")
            keys = o.get("keys")
            has_keys = isinstance(keys, list) and keys and isinstance(keys[0], dict)
            if ref_type == "ExternalReference" and has_keys:
                if keys[0].get("type") not in _GLOBALLY_IDENTIFIABLE_KEY_TYPES:
                    keys[0]["type"] = "GlobalReference"
            elif ref_type == "ModelReference" and has_keys:
                # AASd-123: a non-identifiable first key (e.g. SAMM valueId
                # pointing at a global URN with key type DataElement) is really
                # an external reference. Demote it so basyx accepts it.
                if keys[0].get("type") not in _AAS_IDENTIFIABLE_KEY_TYPES:
                    o["type"] = "ExternalReference"
                    keys[0]["type"] = "GlobalReference"
            for v in o.values():
                _walk_refs(v)
        elif isinstance(o, list):
            for x in o:
                _walk_refs(x)
    _walk_refs(data)

    # AASd-120 — a direct child of a SubmodelElementList must not carry an
    # idShort. Strip any present so basyx keeps the list instead of rejecting
    # it; members then become index-addressed (see flatten_aas_object_store,
    # which keys them as ``Parent[i]``).
    def _strip_sml_member_idshorts(o):
        if isinstance(o, dict):
            if o.get("modelType") == "SubmodelElementList":
                for child in o.get("value", []) or []:
                    if isinstance(child, dict):
                        child.pop("idShort", None)
            for v in o.values():
                _strip_sml_member_idshorts(v)
        elif isinstance(o, list):
            for x in o:
                _strip_sml_member_idshorts(x)
    _strip_sml_member_idshorts(data)

    return data


def sanitize_aas_json_values(obj):
    if isinstance(obj, dict):
        # specific check for AAS Property with valueType
        val_type = obj.get("valueType") or obj.get("value_type")
        val = obj.get("value")

        # Remap XSD types basyx-python-sdk can't resolve (independent of value).
        if val_type in _UNSUPPORTED_XSD_REMAP:
            val_type = _UNSUPPORTED_XSD_REMAP[val_type]
            if "valueType" in obj:
                obj["valueType"] = val_type
            if "value_type" in obj:
                obj["value_type"] = val_type

        if val_type and val is not None:
            val_str = str(val).strip()
            
            # 1. Booleans
            if val_type in ["xs:boolean", "boolean"]:
                if val_str.lower() not in ["true", "false", "1", "0"]:
                    obj["valueType"] = "xs:string"
            
            # 2. Integers
            elif val_type in ["xs:int", "xs:integer", "xs:long", "xs:short", "xs:byte", 
                              "xs:unsignedInt", "xs:unsignedLong", "xs:unsignedShort", "xs:unsignedByte",
                              "xs:nonNegativeInteger", "xs:positiveInteger", "xs:nonPositiveInteger", "xs:negativeInteger",
                              "int", "integer"]:
                try:
                    # Try converting to int
                    # Handle "1.0" as int if strictly integer
                    # But easiest check is int()
                    int(float(val_str)) # allow "1.0" -> 1
                except ValueError:
                     obj["valueType"] = "xs:string"

            # 3. Floats
            elif val_type in ["xs:float", "xs:double", "xs:decimal", "float", "double"]:
                try:
                    float(val_str)
                except ValueError:
                     obj["valueType"] = "xs:string"
            
            # 4. Dates
            elif val_type in ["xs:date", "xs:dateTime", "xs:time", "xs:gYear", "xs:gYearMonth", "xs:gMonth", "xs:gMonthDay", "xs:gDay",
                              "date", "dateTime", "time", "gYear", "gYearMonth", "gMonth", "gMonthDay", "gDay"]:
                if val_str:
                    try:
                        date_parser.parse(val_str)
                    except Exception:
                        obj["valueType"] = "xs:string"
                else:
                    obj["value"] = None # Empty string date is still problematic unless we call it string? 
                                        # But empty string is valid xs:string.
                    obj["valueType"] = "xs:string"

        # Check for idShort
        if "idShort" in obj and obj["idShort"]:
             # regex for valid: [a-zA-Z0-9_]+
             # replace anything else with _
             new_id_short = re.sub(r'[^a-zA-Z0-9_]', '_', obj["idShort"])
             if new_id_short != obj["idShort"]:
                 obj["idShort"] = new_id_short
                 
        # Check for ShortNameTypeIEC61360
        if "shortName" in obj and isinstance(obj["shortName"], list):
            for item in obj["shortName"]:
                if isinstance(item, dict) and "text" in item:
                    text = item["text"]
                    if text is None:
                        item["text"] = "?"
                    else:
                        if len(text) > 18:
                            item["text"] = text[:18]
                        elif len(text) == 0:
                            item["text"] = "?"

        # Check for MultiLanguageTextType (e.g. description, displayName, etc.)
        # Structure: "description": [ { "language": "en", "text": "..." }, ... ]
        # We need to check any list of dicts that has "language" and "text" keys.
        # This is a heuristic but safe enough for AAS JSON.
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict) and "language" in item and "text" in item:
                             text = item["text"]
                             lang = str(item.get("language", ""))
                             # BCP 47 Strict validation: force 2 lower letters
                             lang = re.sub(r'[^a-zA-Z]', '', lang)
                             if len(lang) >= 2:
                                 item["language"] = lang[:2].lower()
                             else:
                                 item["language"] = "en"
                                 
                             if text is None:
                                 item["text"] = "?" # Min length 1
                             elif len(text) == 0:
                                 item["text"] = "?" # Min length 1
                                 
        # Fix for empty ValueTypeIEC61360 strings causing basyx loader exceptions
        for key in ["dataType", "valueFormat", "valueType"]:
            if key in obj and isinstance(obj[key], str) and len(obj[key]) == 0:
                obj[key] = "STRING"
                
        keys_to_delete = []
        # Globally replace empty strings and strip malformed references for strict basyx validators
        for k in list(obj.keys()):
            v = obj[k]
            if isinstance(v, str) and len(v) == 0:
                obj[k] = "_"
            elif isinstance(v, dict) and "keys" in v and isinstance(v["keys"], list) and len(v["keys"]) == 0:
                keys_to_delete.append(k)
                
        for k in keys_to_delete:
            del obj[k]
                                 
        for k, v in obj.items():
            sanitize_aas_json_values(v)
            
    elif isinstance(obj, list):
        for item in obj:
            sanitize_aas_json_values(item)
    return obj


def sanitize_aas_xml_values(xml_content: bytes) -> bytes:
    # Detect and register all namespaces from the input
    # This is critical because ET.write generates new prefixes (ns0, ns1...) for unregistered namespaces,
    # and Basyx's strict parser may rely on specific prefixes or fail to handle the generated ones correctly.
    try:
        events = io.BytesIO(xml_content)
        for event, (prefix, uri) in ET.iterparse(events, events=['start-ns']):
            ET.register_namespace(prefix, uri)
    except Exception:
        pass # Best effort

    # Specific override for standard AAS if not found or to ensure 'aas' prefix
    ET.register_namespace("aas", "https://admin-shell.io/aas/3/0")
    ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")
    
    try:
        root = ET.fromstring(xml_content)
        print(f"DEBUG: Root tag: {root.tag}")
        
        count = 0
        # Iterate over all elements
        for elem in root.iter():
            count += 1
            if count < 5:
                print(f"DEBUG: Iterating element: {elem.tag}")
            
            # Check if it looks like a Property (has valueType and value children)
            # We don't rely on tag name ending in "Property" strictly, 
            # checking children is more robust for submodel elements.
            
            val_type_elem = None
            value_elem = None
            
            # Scan children
            # Note: ET.fromstring preserves prefixes in tags like {ns}tag
            id_short_text = "Unknown"
            
            for child in elem:
                if child.tag.endswith("valueType"):
                    val_type_elem = child
                elif child.tag.endswith("value"):
                    value_elem = child
                elif child.tag.endswith("idShort"):
                    id_short_text = child.text
            
            if val_type_elem is not None and value_elem is not None:
                val_type = val_type_elem.text
                val_text = value_elem.text
                
                is_invalid = False
                
                if val_type and val_text is not None:
                    s_val = val_text.strip()
                    
                    # 1. Booleans
                    if val_type in ["xs:boolean", "boolean"]:
                        if s_val.lower() not in ["true", "false", "1", "0"]:
                            is_invalid = True
                            
                    # 2. Integers
                    elif val_type in ["xs:int", "xs:integer", "xs:long", "xs:short", "xs:byte", 
                                      "xs:unsignedInt", "xs:unsignedLong", "xs:unsignedShort", "xs:unsignedByte",
                                      "xs:nonNegativeInteger", "xs:positiveInteger", "xs:nonPositiveInteger", "xs:negativeInteger",
                                      "int", "integer"]:
                        try:
                            if s_val == "":
                                is_invalid = True
                            else:
                                int(float(s_val))
                        except ValueError:
                            is_invalid = True

                    # 3. Floats
                    elif val_type in ["xs:float", "xs:double", "xs:decimal", "float", "double"]:
                        try:
                            if s_val == "":
                                is_invalid = True
                            else:
                                float(s_val)
                        except ValueError:
                            is_invalid = True
                
                elif val_type and val_text is None:
                    # Empty tag <value /> usually means None text in ET, or empty string.
                    # If None/Empty is found for these types, treat as invalid (or just remove to be safe 'empty')
                     if val_type in ["xs:boolean", "boolean", "xs:int", "xs:integer", "int", "integer", "xs:float", "xs:double", "float", "double",
                                     "xs:date", "xs:dateTime", "xs:time", "xs:gYear", "xs:gYearMonth", "xs:gMonth", "xs:gMonthDay", "xs:gDay",
                                     "date", "dateTime", "time", "gYear", "gYearMonth", "gMonth", "gMonthDay", "gDay"]:
                         is_invalid = True
                
                # Check for date/time validity
                if not is_invalid and val_text is not None and val_type in ["xs:date", "xs:dateTime", "xs:time", "xs:gYear", "xs:gYearMonth", "xs:gMonth", "xs:gMonthDay", "xs:gDay",
                                                                            "date", "dateTime", "time", "gYear", "gYearMonth", "gMonth", "gMonthDay", "gDay"]:
                    try:
                        # minimal validation using dateutil
                        # valid if it can be parsed
                        date_parser.parse(val_text)
                        
                        # Extra check for YearMonth/Year/etc if needed, but dateutil is lenient.
                        # basyx might be stricter.
                        # If simple parse fails, it's definitely invalid.
                        # If it passes, it *might* still be invalid for XSD but let's hope it's enough.
                        pass
                    except Exception:
                        is_invalid = True


                if is_invalid:
                    # Fallback to string type instead of removing the value
                    # This allows preservation of data even if it doesn't match the declared type
                    val_type_elem.text = "xs:string"

            # Check for shortName elements (used in IEC61360)
            # They have constraints: max length 18, min length 1
            if elem.tag.endswith("shortName"):
                # Iterate over children (LangStrings)
                for lang_string in elem:
                    # Expect children: language, text
                    text_elem = None
                    for child in lang_string:
                        if child.tag.endswith("text"):
                            text_elem = child
                            break
                    
                    if text_elem is not None:
                        if text_elem.text is not None:
                            # 1. Check length > 18
                            if len(text_elem.text) > 18:
                                # Truncate to 18
                                text_elem.text = text_elem.text[:18]
                            # 2. Check empty
                            elif len(text_elem.text) == 0:
                                # Empty text is invalid for shortName
                                text_elem.text = "?"
                        else:
                            # None text is also invalid (empty)
                            text_elem.text = "?"
                            
            # Check for idShort (AASd-002: letters, digits, underscore, starting with letter)
            # We will just replace invalid chars with _
            # This is applied to ALL elements, but typically Referables have idShort.
            # We look for a child named idShort.
            for child in elem:
                if child.tag.endswith("idShort") and child.text:
                    # regex for valid: [a-zA-Z0-9_]+
                    # replace anything else with _
                    new_id_short = re.sub(r'[^a-zA-Z0-9_]', '_', child.text)
                    if new_id_short != child.text:
                        child.text = new_id_short

        # Write back
        out = io.BytesIO()
        ET.ElementTree(root).write(out, encoding='utf-8', xml_declaration=True)
        return out.getvalue()

    except Exception as e:
        print(f"XML Sanitization warning: {e}")
        return xml_content



def aas_json_parser(aasx_filepath: str) -> pd.DataFrame:
    aas_store: model.DictObjectStore[model.Identifiable] = model.DictObjectStore()

    import json
    with open(aasx_filepath, "r", encoding='utf-8-sig') as f:
        data = json.load(f)

    # Structural fixes (null assetInformation, shell.submodel.type, AASd-122)
    # must run before the value-level pass since they're not empty-strings.
    data = _sanitize_aas_structural(data)
    # Pre-process to fix invalid booleans + empty strings + numeric values
    data = sanitize_aas_json_values(data)

    # Now parse the sanitized data
    json_str = json.dumps(data)
    json_bytes_len = len(json_str)

    # Mirror basyx.aas WARNING records into our logger so failsafe drops are
    # visible alongside the rest of the parse trace (otherwise they vanish into
    # the basyx package logger).
    basyx_logger = logging.getLogger("basyx")
    captured = []

    class _Capture(logging.Handler):
        def emit(self, record):
            captured.append(record.getMessage())

    capture_handler = _Capture(level=logging.WARNING)
    basyx_logger.addHandler(capture_handler)
    prev_level = basyx_logger.level
    basyx_logger.setLevel(min(prev_level or logging.WARNING, logging.WARNING))

    try:
        f_pseudo = io.StringIO(json_str)
        aas_objs = list(read_aas_json_file(f_pseudo, failsafe=True))
    finally:
        basyx_logger.removeHandler(capture_handler)
        basyx_logger.setLevel(prev_level)

    src_aas_count = len(data.get("assetAdministrationShells", []) or [])
    src_sm_count = len(data.get("submodels", []) or [])
    src_cd_count = len(data.get("conceptDescriptions", []) or [])
    logging.getLogger(__name__).info(
        f"[aas_json_parser] source={json_bytes_len}B "
        f"shells={src_aas_count} submodels={src_sm_count} CDs={src_cd_count} "
        f"=> SDK parsed {len(aas_objs)} objects (failsafe=True)"
    )
    if captured:
        logging.getLogger(__name__).warning(
            f"[aas_json_parser] basyx SDK warnings during parse "
            f"({len(captured)} entries; first 5):"
        )
        for msg in captured[:5]:
            logging.getLogger(__name__).warning(f"  basyx: {msg}")

    for obj in aas_objs:
        aas_store.add(obj)
    df_aas, G = flatten_aas_object_store(aas_store, return_graph=True)

    logging.getLogger(__name__).info(
        f"[aas_json_parser] graph: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges"
    )

    import networkx as nx
    graph_path = aasx_filepath.replace(".json", "_graph.json")
    with open(graph_path, 'w', encoding='utf-8') as f:
        json.dump(nx.node_link_data(G), f, indent=2)

    return df_aas



def flatten_aas_object_store(object_store: model.DictObjectStore[model.Identifiable], with_entity: bool = False, return_graph: bool = False):

    def get_description_text(desc_dict, language='en'):
        if not desc_dict:
            return 'None'
        try:
            # Try to get the preferred language
            if language in desc_dict:
                return desc_dict[language]
            # Fallback: return the first available language's text
            return next(iter(desc_dict.values()), 'None')
        except Exception:
            return 'None'

    # Helper function to extract a semanticId string (e.g., first key’s value if available)
    def get_semantic_id_str(sem_id):
        if sem_id is None:
            return ""
        try:
            keys = getattr(sem_id, 'key', None)
            if keys and len(keys) > 0:
                return str(keys[0].value)
        except Exception:
            pass
        return ""

    cds = {}
    for obj in object_store:
        if isinstance(obj, model.ConceptDescription):
            cd_id = obj.id
            desc = get_description_text(getattr(obj, 'description', None))
            unit = ""
            for eds in getattr(obj, "embedded_data_specifications", []):
                ds_content = getattr(eds, "data_specification_content", None)
                if ds_content and hasattr(ds_content, "unit"):
                    unit_val = getattr(ds_content, "unit", "")
                    if unit_val: unit = str(unit_val)
            cds[cd_id] = {"definition": desc, "unit": unit}

    import networkx as nx
    G = nx.DiGraph()
    rows = []

    for identifiable in object_store:
        if isinstance(identifiable, model.AssetAdministrationShell):
            aas = identifiable
            aas_id = aas.id
            aas_id_short = aas.id_short
            encoded_aas_id = encode_id(aas_id)

            node_id = aas_id_short
            G.add_node(node_id, idShort=aas_id_short, type=type(aas).__name__, description=get_description_text(aas.description), value=None, API_path=f"/shells/{encoded_aas_id}", semanticId='None', cd_definition='', cd_unit='')

            rows.append({
                "idShort": aas_id_short,
                "type": type(aas).__name__,
                "description": get_description_text(aas.description),
                "value": None,
                "semantic_path": f"{aas_id_short}",
                "API_path": f"/shells/{encoded_aas_id}",
                "semanticId": 'None',
                "cd_definition": '',
                "cd_unit": ''
            })

            # For each submodel reference in the AAS
            for ref in aas.submodel:
                submodel_id = ref.key[0].value
                submodel = object_store.get(submodel_id)
                if not isinstance(submodel, model.Submodel):
                    continue

                sub_id_short = submodel.id_short
                sub_descrip = get_description_text(submodel.description)
                semantic_path_sm = f"{aas_id_short}/{sub_id_short}"
                sub_sem = get_semantic_id_str(submodel.semantic_id)
                sub_cd = cds.get(sub_sem, {})

                sub_node_id = f"{aas_id_short}/{sub_id_short}"
                G.add_node(sub_node_id, idShort=sub_id_short, type=type(submodel).__name__, description=sub_descrip, value=None, API_path=f"/submodels/{encode_id(submodel.id)}", semanticId=sub_sem, cd_definition=sub_cd.get('definition', ''), cd_unit=sub_cd.get('unit', ''))
                G.add_edge(node_id, sub_node_id)

                rows.append({
                    "idShort": sub_id_short,
                    "type": type(submodel).__name__,
                    "description": sub_descrip,
                    "value": None,
                    "semantic_path": semantic_path_sm,
                    "API_path": f"/submodels/{encode_id(submodel.id)}",
                    "semanticId": sub_sem,
                    "cd_definition": sub_cd.get('definition', ''),
                    "cd_unit": sub_cd.get('unit', '')
                })

                def _flatten_element(elem: model.SubmodelElement, parent_path: str, parent_node_id: str, visited=None, index=None):
                    if visited is None:
                        visited = set()

                    # Prevent circular recursion
                    obj_id = id(elem)
                    if obj_id in visited:
                        return
                    visited.add(obj_id)
                    # SML members are positionally addressed (AASd-120: no idShort).
                    # basyx fabricates a placeholder idShort for them, so key on the
                    # list index instead, per the AAS idShort-path convention.
                    elem_id_short = f"[{index}]" if index is not None else elem.id_short
                    elem_type = type(elem).__name__
                    elem_desc = get_description_text(getattr(elem, 'description', []))
                    elem_sem = get_semantic_id_str(getattr(elem, 'semantic_id', None))
                    elem_cd = cds.get(elem_sem, {})
                    
                    # Extract value based on type
                    elem_val = None
                    if isinstance(elem, model.Property):
                        elem_val = str(elem.value) if elem.value is not None else None
                    elif isinstance(elem, model.MultiLanguageProperty):
                        try:
                             val = getattr(elem, 'value', None)
                             if val:
                                 texts = [f"{ls.language}: {ls.text}" for ls in val] if hasattr(val, '__iter__') else str(val)
                                 elem_val = str(texts)
                        except:
                            elem_val = str(getattr(elem, 'value', ''))
                    elif isinstance(elem, model.File):
                        elem_val = str(elem.value) if elem.value is not None else None
                    elif isinstance(elem, model.Range):
                         elem_val = f"min={elem.min}, max={elem.max}"
                    elif isinstance(elem, model.ReferenceElement):
                        # Extract keys from reference
                        ref_val = elem.value
                        if ref_val:
                             keys = [f"{k.type}={k.value}" for k in ref_val.keys]
                             elem_val = ";".join(keys)

                    # Build the path segment for this element. SML members append
                    # the index in brackets (Parent[i]); everything else dot-joins
                    # the idShort (Parent.idShort).
                    if index is not None:
                        full_id_path = elem_id_short if parent_path == "" else f"{parent_path}{elem_id_short}"
                    else:
                        full_id_path = elem_id_short if parent_path == "" else f"{parent_path}.{elem_id_short}"
                    elem_path = f"/submodels/{encode_id(submodel_id)}/submodel-elements/{full_id_path}"
                    
                    elem_node_id = f"{aas_id_short}/{sub_id_short}/{full_id_path}"
                    G.add_node(elem_node_id, idShort=elem_id_short, type=elem_type, description=elem_desc, value=elem_val, API_path=elem_path, semanticId=elem_sem, cd_definition=elem_cd.get('definition', ''), cd_unit=elem_cd.get('unit', ''))
                    G.add_edge(parent_node_id, elem_node_id)
                    
                    # Add this element's info to the list
                    rows.append({
                        "idShort": elem_id_short,
                        "type": elem_type,
                        "description": elem_desc,
                        "value": elem_val,
                        "semantic_path": f"{semantic_path_sm}/{full_id_path}",
                        "API_path": elem_path,
                        "semanticId": elem_sem,
                        "cd_definition": elem_cd.get('definition', ''),
                        "cd_unit": elem_cd.get('unit', '')
                    })
                    if isinstance(elem, model.SubmodelElementList):
                        # Members are index-addressed; pass the position so each
                        # child path becomes Parent[i].
                        for i, child in enumerate(getattr(elem, 'value', []) or []):
                            _flatten_element(child, full_id_path, elem_node_id, visited, index=i)
                    elif isinstance(elem, model.SubmodelElementCollection):
                        for child in getattr(elem, 'value', []) or []:
                            _flatten_element(child, full_id_path, elem_node_id, visited)
                    elif isinstance(elem, model.Entity):
                        if with_entity:
                            for child in getattr(elem, 'statement', []) or []:
                                child_sem = get_semantic_id_str(getattr(child, 'semantic_id', None))
                                child_cd = cds.get(child_sem, {})
                                c_node_id = f"{elem_node_id}/{child.id_short}"
                                
                                child_val = getattr(child, 'value', None) if hasattr(child, 'value') and not isinstance(child, (model.SubmodelElementCollection, model.SubmodelElementList)) else None
                                if child_val is not None and not isinstance(child_val, (str, int, float, bool, list, dict)):
                                    child_val = str(child_val)
                                    
                                G.add_node(c_node_id, idShort=child.id_short, type=type(child).__name__, description=get_description_text(getattr(child, 'description', [])), value=child_val, API_path=f"{elem_path}.{child.id_short}", semanticId=child_sem, cd_definition=child_cd.get('definition',''), cd_unit=child_cd.get('unit',''))
                                G.add_edge(elem_node_id, c_node_id)
                                rows.append({
                                    "idShort": child.id_short,
                                    "type": type(child).__name__,
                                    "description": get_description_text(getattr(child, 'description', [])),
                                    "value": child_val,
                                    "semantic_path": f"/{submodel_id}/{full_id_path}/{child.id_short}",
                                    "API_path": f"{elem_path}.{child.id_short}",
                                    "semanticId": child_sem,
                                    "cd_definition": child_cd.get('definition',''),
                                    "cd_unit": child_cd.get('unit','')
                                })

                for elem in submodel.submodel_element:
                    _flatten_element(elem, "", sub_node_id)

    df = pd.DataFrame(rows, columns=["idShort", "type", "description", "value", "semantic_path", "API_path", "semanticId", "cd_definition", "cd_unit"])
    if return_graph:
        return df, G
    return df