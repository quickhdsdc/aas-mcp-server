"""Standalone AAS model algorithms, adapted from the updated aasctl implementation."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
VALUE_TYPES = ('xs:anyURI', 'xs:base64Binary', 'xs:boolean', 'xs:byte', 'xs:date', 'xs:dateTime', 'xs:decimal', 'xs:double', 'xs:duration', 'xs:float', 'xs:gDay', 'xs:gMonth', 'xs:gMonthDay', 'xs:gYear', 'xs:gYearMonth', 'xs:hexBinary', 'xs:int', 'xs:integer', 'xs:long', 'xs:negativeInteger', 'xs:nonNegativeInteger', 'xs:nonPositiveInteger', 'xs:positiveInteger', 'xs:short', 'xs:string', 'xs:time', 'xs:unsignedByte', 'xs:unsignedInt', 'xs:unsignedLong', 'xs:unsignedShort')
ELEMENT_TYPES = ('Property', 'MultiLanguageProperty', 'Range', 'Blob', 'File', 'ReferenceElement', 'RelationshipElement', 'SubmodelElementCollection', 'SubmodelElementList', 'Operation', 'Entity')
KIND_VALUES = ('Instance', 'Template')

class ModelError(ValueError):
    pass

def external_reference(value: str) -> Dict[str, Any]:
    return {'type': 'ExternalReference', 'keys': [{'type': 'GlobalReference', 'value': value}]}

def model_reference(keys: List[Dict[str, str]]) -> Dict[str, Any]:
    return {'type': 'ModelReference', 'keys': keys}

def submodel_reference(submodel_id: str) -> Dict[str, Any]:
    return model_reference([{'type': 'Submodel', 'value': submodel_id}])

def qualifier(kind: str, value: str, value_type: str='xs:string') -> Dict[str, Any]:
    return {'type': kind, 'valueType': value_type, 'value': value}

def cardinality_qualifier(value: str) -> Dict[str, Any]:
    out = qualifier('SMT/Cardinality', value)
    out['kind'] = 'TemplateQualifier'
    return out

def delegation_qualifier(url: str) -> Dict[str, Any]:
    return qualifier('invocationDelegation', url)

def _base(id_short: Optional[str], model_type: str, *, semantic_id: Optional[str]=None, description: Optional[str]=None, display_name: Optional[str]=None, qualifiers: Optional[List[Dict[str, Any]]]=None) -> Dict[str, Any]:
    element: Dict[str, Any] = {'modelType': model_type}
    if id_short:
        element['idShort'] = id_short
    if semantic_id:
        element['semanticId'] = external_reference(semantic_id)
    if description:
        element['description'] = [{'language': 'en', 'text': description}]
    if display_name:
        element['displayName'] = [{'language': 'en', 'text': display_name}]
    if qualifiers:
        element['qualifiers'] = qualifiers
    return element

def make_element(model_type: str, id_short: Optional[str]=None, *, value: Any=None, value_type: str='xs:string', semantic_id: Optional[str]=None, description: Optional[str]=None, display_name: Optional[str]=None, qualifiers: Optional[List[Dict[str, Any]]]=None, min_value: Optional[str]=None, max_value: Optional[str]=None, content_type: Optional[str]=None, first: Optional[Dict[str, Any]]=None, second: Optional[Dict[str, Any]]=None, type_value_list_element: Optional[str]=None, value_type_list_element: Optional[str]=None, order_relevant: bool=True, entity_type: str='SelfManagedEntity', global_asset_id: Optional[str]=None, input_variables: Optional[List[Dict[str, Any]]]=None, output_variables: Optional[List[Dict[str, Any]]]=None) -> Dict[str, Any]:
    if model_type not in ELEMENT_TYPES:
        raise ModelError(f"unknown modelType {model_type!r}; expected one of {', '.join(ELEMENT_TYPES)}")
    element = _base(id_short, model_type, semantic_id=semantic_id, description=description, display_name=display_name, qualifiers=qualifiers)
    if model_type == 'Property':
        if value_type not in VALUE_TYPES:
            raise ModelError(f"unknown valueType {value_type!r}; expected one of {', '.join(VALUE_TYPES)}")
        element['valueType'] = value_type
        element['value'] = None if value is None else str(value)
    elif model_type == 'MultiLanguageProperty':
        if value is None:
            element['value'] = []
        elif isinstance(value, list):
            element['value'] = value
        else:
            element['value'] = [{'language': 'en', 'text': str(value)}]
    elif model_type == 'Range':
        element['valueType'] = value_type
        element['min'] = min_value
        element['max'] = max_value
    elif model_type == 'Blob':
        element['contentType'] = content_type or 'application/octet-stream'
        if value is not None:
            element['value'] = value
    elif model_type == 'File':
        element['contentType'] = content_type or 'application/octet-stream'
        element['value'] = value
    elif model_type == 'ReferenceElement':
        if value is None:
            raise ModelError('ReferenceElement requires --value (the target reference)')
        element['value'] = value if isinstance(value, dict) else external_reference(str(value))
    elif model_type == 'RelationshipElement':
        if not first or not second:
            raise ModelError('RelationshipElement requires --first and --second references')
        element['first'] = first
        element['second'] = second
    elif model_type == 'SubmodelElementCollection':
        element['value'] = []
    elif model_type == 'SubmodelElementList':
        if not type_value_list_element:
            raise ModelError('SubmodelElementList requires --type-value-list-element (the modelType every child must have)')
        element['typeValueListElement'] = type_value_list_element
        element['orderRelevant'] = order_relevant
        if value_type_list_element:
            element['valueTypeListElement'] = value_type_list_element
        element['value'] = []
    elif model_type == 'Operation':
        element['inputVariables'] = input_variables or []
        element['outputVariables'] = output_variables or []
    elif model_type == 'Entity':
        element['entityType'] = entity_type
        element['statements'] = []
        if global_asset_id:
            element['globalAssetId'] = global_asset_id
    return element

def lang_strings(raw: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for part in raw.split('|'):
        text, sep, lang = part.rpartition('@')
        if not sep or not lang or ' ' in lang:
            text, lang = (part, 'en')
        out.append({lang.strip(): text.strip()})
    return out

def operation_variable(element: Dict[str, Any]) -> Dict[str, Any]:
    return {'value': element}

def make_submodel(submodel_id: str, id_short: str, *, semantic_id: Optional[str]=None, kind: str='Instance', description: Optional[str]=None) -> Dict[str, Any]:
    if kind not in KIND_VALUES:
        raise ModelError(f"kind must be one of {', '.join(KIND_VALUES)}")
    submodel = _base(id_short, 'Submodel', semantic_id=semantic_id, description=description)
    submodel['id'] = submodel_id
    submodel['kind'] = kind
    submodel['submodelElements'] = []
    return submodel

def make_shell(shell_id: str, id_short: str, *, global_asset_id: Optional[str]=None, asset_kind: str='Instance', description: Optional[str]=None) -> Dict[str, Any]:
    shell = _base(id_short, 'AssetAdministrationShell', description=description)
    shell['id'] = shell_id
    shell['assetInformation'] = {'assetKind': asset_kind, 'globalAssetId': global_asset_id or shell_id}
    shell['submodels'] = []
    return shell
