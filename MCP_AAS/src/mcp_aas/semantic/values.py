"""Construct the ValueOnly payload of a live AAS element."""
from .match import Entity, Decision, infer_element_type, value_for_write


def typed_value(element, value):
    kind = element.get("modelType")
    if kind == "Property":
        if isinstance(value, (dict, list)):
            raise ValueError("A Property requires a scalar value")
        return str(value).lower() if isinstance(value, bool) else str(value)
    if kind == "MultiLanguageProperty":
        if isinstance(value, dict):
            return [{key: str(text)} for key, text in value.items()]
        if isinstance(value, list):
            result = []
            for entry in value:
                if not isinstance(entry, dict):
                    raise ValueError("Language values must be objects")
                result.append({entry['language']: entry['text']} if 'language' in entry and 'text' in entry else entry)
            return result
    if kind == "Range" and isinstance(value, dict):
        if not {"min", "max"}.issubset(value):
            raise ValueError("A Range requires min and max")
        return {key: str(value[key]) for key in ("min", "max")}
    if kind in ("File", "Blob") and isinstance(value, dict):
        if not {"contentType", "value"}.issubset(value):
            raise ValueError("A File/Blob requires contentType and value")
        return {key: value[key] for key in ("contentType", "value")}
    if kind == "ReferenceElement" and isinstance(value, dict):
        if not {"type", "keys"}.issubset(value):
            raise ValueError("A ReferenceElement requires a reference with type and keys")
        return value
    text = str(value)
    spec = infer_element_type(Entity(element.get("idShort", "Value"), text))
    decision = Decision("Value", text, "", "", "write", "auto", "direct write",
        model_type=kind, min_value=spec.min_value, max_value=spec.max_value,
        content_type=element.get("contentType") or spec.content_type)
    return value_for_write(decision)
