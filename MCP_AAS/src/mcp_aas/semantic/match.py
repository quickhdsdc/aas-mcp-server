"""Standalone AAS match algorithms, adapted from the updated aasctl implementation."""

from __future__ import annotations
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
from .index import Filter, Index, search
from .model import external_reference, lang_strings, make_element
THETA_HIGH = 0.8
THETA_LOW = 0.5
CONTAINER_TYPES = ('SubmodelElementCollection', 'SubmodelElementList')
NON_CANDIDATE_TYPES = ('AssetAdministrationShell', 'Submodel', 'Capability', 'Operation', 'Entity', 'BasicEventElement', 'RelationshipElement', 'AnnotatedRelationshipElement')
CREATABLE_TYPES = ('Property', 'MultiLanguageProperty', 'Range', 'File', 'ReferenceElement')
_IDSHORT_SAFE = re.compile('[^A-Za-z0-9_]+')

class MatchError(ValueError):
    pass

@dataclass
class Entity:
    entity: str
    value: Optional[str] = None
    description: str = ''
    unit: str = ''
    source: str = ''

    @classmethod
    def parse(cls, raw: Dict[str, Any]) -> 'Entity':
        name = raw.get('entity') or raw.get('name') or ''
        if not name:
            raise MatchError(f'entity has no name: {json.dumps(raw)[:120]}')
        value = raw.get('value')
        return cls(entity=str(name), value=None if value is None else str(value), description=str(raw.get('description') or ''), unit=str(raw.get('unit') or ''), source=str(raw.get('source') or raw.get('location') or ''))

    def query(self) -> str:
        return f'name: {self.entity}; description: {self.description}'

def load_entities(raw: Any) -> List[Entity]:
    if isinstance(raw, dict):
        raw = raw.get('entities', raw.get('result', []))
    if not isinstance(raw, list):
        raise MatchError("entities file must be a JSON list, or an object with an 'entities' list")
    return [Entity.parse(item) for item in raw if isinstance(item, dict)]

@dataclass
class Candidate:
    path: str
    id_short: Optional[str]
    model_type: str
    score: float
    parent: Optional[str]
    unit: str = ''
    definition: str = ''
    type_value_list_element: Optional[str] = None

@dataclass
class Decision:
    entity: str
    value: Optional[str]
    unit: str
    source: str
    action: str
    band: str
    reason: str
    path: Optional[str] = None
    model_type: Optional[str] = None
    value_type: Optional[str] = None
    id_short: Optional[str] = None
    min_value: Optional[str] = None
    max_value: Optional[str] = None
    content_type: Optional[str] = None
    container_type: Optional[str] = None
    score: float = 0.0
    margin: float = 0.0
    warnings: List[str] = field(default_factory=list)
    candidates: List[Dict[str, Any]] = field(default_factory=list)

    def line(self) -> str:
        mark = {'write': 'write ', 'create': 'create', 'skip': 'skip  '}[self.action]
        target = self.path or '-'
        warn = f"  !! {'; '.join(self.warnings)}" if self.warnings else ''
        return f'{mark} {self.score:.3f} +{self.margin:.3f} {self.band:<10} {self.entity[:26]:<26} -> {target}{warn}'

@dataclass
class Plan:
    submodel_id: str
    theta_high: float
    theta_low: float
    model: Optional[str]
    decisions: List[Decision] = field(default_factory=list)

    def counts(self) -> Dict[str, int]:
        out = {'write': 0, 'create': 0, 'skip': 0, 'auto': 0, 'arbitrated': 0, 'warnings': 0}
        for decision in self.decisions:
            out[decision.action] += 1
            if decision.band in ('auto', 'arbitrated'):
                out[decision.band] += 1
            out['warnings'] += len(decision.warnings)
        return out

    def calibration_note(self) -> Optional[str]:
        if not self.decisions:
            return None
        best = max((d.score for d in self.decisions))
        if best >= self.theta_high:
            return None
        return f'no entity reached theta_high={self.theta_high} (best was {best:.3f}), so the auto-accept band never fired and every decision above theta_low went to arbitration. Check the threshold against this descriptor and embedding model before reading the band counts as a result.'

    def to_json(self) -> Dict[str, Any]:
        return {'submodel': self.submodel_id, 'thresholds': {'high': self.theta_high, 'low': self.theta_low}, 'arbitration_model': self.model, 'counts': self.counts(), 'decisions': [asdict(d) for d in self.decisions]}

    @classmethod
    def from_json(cls, raw: Dict[str, Any]) -> 'Plan':
        try:
            thresholds = raw['thresholds']
            plan = cls(submodel_id=raw['submodel'], theta_high=float(thresholds['high']), theta_low=float(thresholds['low']), model=raw.get('arbitration_model'))
            for item in raw['decisions']:
                known = {k: v for k, v in item.items() if k in Decision.__annotations__}
                plan.decisions.append(Decision(**known))
        except (KeyError, TypeError, ValueError) as exc:
            raise MatchError(f'not a match plan: {exc}') from exc
        return plan
_DATE = re.compile('^\\d{4}-\\d{2}-\\d{2}$')
_DATETIME = re.compile('^\\d{4}-\\d{2}-\\d{2}[T ]\\d{2}:\\d{2}')
_IDENTIFIER_DIGITS = 10

def infer_value_type(value: Optional[str], unit: str='') -> str:
    if value is None:
        return 'xs:string'
    text = value.strip()
    if not text:
        return 'xs:string'
    if text.lower() in ('true', 'false'):
        return 'xs:boolean'
    if _DATETIME.match(text):
        return 'xs:dateTime'
    if _DATE.match(text):
        return 'xs:date'
    digits = text.lstrip('+-')
    if digits.isdigit():
        if len(digits) > 1 and digits[0] == '0':
            return 'xs:string'
        if not unit.strip() and len(digits) >= _IDENTIFIER_DIGITS:
            return 'xs:string'
        return 'xs:integer'
    try:
        float(text)
        return 'xs:double'
    except ValueError:
        return 'xs:string'

@dataclass
class ElementSpec:
    model_type: str = 'Property'
    value_type: Optional[str] = None
    min_value: Optional[str] = None
    max_value: Optional[str] = None
    content_type: Optional[str] = None
    reason: str = ''

    def merged_with(self, other: Optional['ElementSpec']) -> 'ElementSpec':
        if other is None:
            return self
        return ElementSpec(model_type=other.model_type or self.model_type, value_type=other.value_type or self.value_type, min_value=other.min_value if other.min_value is not None else self.min_value, max_value=other.max_value if other.max_value is not None else self.max_value, content_type=other.content_type or self.content_type, reason=other.reason or self.reason)
_RANGE_SEPARATOR = '(?:\\.\\.\\.|…|\\.\\.|--|–|—|~|-|\\bto\\b|\\bbis\\b)'
_NUMBER = '[+-]?\\d+(?:[.,]\\d+)?'
_RANGE = re.compile(f'^\\s*({_NUMBER})\\s*{_RANGE_SEPARATOR}\\s*({_NUMBER})\\s*(.*?)\\s*$', re.IGNORECASE)
_UNIT_SHAPED = re.compile('^[A-Za-z°µΩ%/\\s.^*-]{1,12}$')
_RANGE_WORDS = re.compile('\\b(range|interval|span|limits?|min(?:imum)?|max(?:imum)?|between|bereich|spanne|grenzwerte?)\\b', re.IGNORECASE)
_LANG_TAGGED = re.compile('@[a-z]{2}(?:[-_][A-Za-z]{2,4})?\\s*(?:\\||$)')
_URLISH = re.compile('^(?:https?://|ftp://|file://|/|\\./|\\.\\./|[A-Za-z]:[\\\\/])')
_DOCUMENT_SUFFIX = re.compile('\\.(pdf|png|jpe?g|gif|svg|tiff?|dwg|dxf|stp|step|igs|iges|xlsx?|docx?|pptx?|csv|txt|xml|json|zip|aasx)$', re.IGNORECASE)
_CONTENT_TYPES = {'pdf': 'application/pdf', 'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'gif': 'image/gif', 'svg': 'image/svg+xml', 'tif': 'image/tiff', 'tiff': 'image/tiff', 'csv': 'text/csv', 'txt': 'text/plain', 'xml': 'application/xml', 'json': 'application/json', 'zip': 'application/zip', 'aasx': 'application/asset-administration-shell-package', 'xls': 'application/vnd.ms-excel', 'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'doc': 'application/msword', 'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'}

def infer_element_type(entity: Entity) -> ElementSpec:
    text = (entity.value or '').strip()
    context = f'{entity.entity} {entity.description}'
    if not text:
        return ElementSpec(model_type='Property', value_type='xs:string', reason='no value to read a shape from')
    if _LANG_TAGGED.search(text):
        return ElementSpec(model_type='MultiLanguageProperty', reason='value carries language tags')
    found = _RANGE.match(text)
    if found:
        low, high, tail = (found.group(1), found.group(2), found.group(3))
        tail_is_unit = bool(tail) and bool(_UNIT_SHAPED.match(tail))
        corroborated = bool(entity.unit.strip()) or bool(_RANGE_WORDS.search(context))
        if (not tail or tail_is_unit) and (corroborated or tail_is_unit):
            low_type = infer_value_type(low, entity.unit)
            high_type = infer_value_type(high, entity.unit)
            value_type = 'xs:double' if 'xs:double' in (low_type, high_type) else low_type
            return ElementSpec(model_type='Range', value_type=value_type, min_value=low, max_value=high, reason=f'value reads as the interval {low}..{high}')
    if _URLISH.match(text) or _DOCUMENT_SUFFIX.search(text):
        suffix = text.rsplit('.', 1)[-1].lower() if '.' in text else ''
        return ElementSpec(model_type='File', content_type=_CONTENT_TYPES.get(suffix, 'application/octet-stream'), reason='value reads as a document path or URL')
    return ElementSpec(model_type='Property', value_type=infer_value_type(entity.value, entity.unit))

def resolve_spec(entity: Entity, container: Optional[Candidate], chosen: Optional[ElementSpec]) -> Tuple[ElementSpec, List[str]]:
    warnings: List[str] = []
    spec = infer_element_type(entity).merged_with(chosen)
    if container is not None and container.model_type == 'SubmodelElementList':
        declared = container.type_value_list_element
        if declared and declared != spec.model_type:
            if declared in CREATABLE_TYPES:
                warnings.append(f'{container.path} is a SubmodelElementList of {declared}; creating a {declared} rather than the {spec.model_type} chosen')
                spec = ElementSpec(model_type=declared, value_type=spec.value_type, min_value=spec.min_value, max_value=spec.max_value, content_type=spec.content_type, reason=spec.reason)
            else:
                warnings.append(f'{container.path} declares children of type {declared}, which no extracted value can be created as')
    if spec.model_type not in CREATABLE_TYPES:
        warnings.append(f'{spec.model_type!r} is not a modelType an extracted value can be created as; creating a Property instead')
        spec = ElementSpec(model_type='Property', value_type=infer_value_type(entity.value, entity.unit))
    if spec.model_type == 'Range' and (spec.min_value is None or spec.max_value is None):
        warnings.append('Range was chosen but the value gives no min and max; creating a Property instead')
        spec = ElementSpec(model_type='Property', value_type=infer_value_type(entity.value, entity.unit))
    if spec.model_type in ('Property', 'Range') and (not spec.value_type):
        spec.value_type = infer_value_type(entity.value, entity.unit)
    if spec.model_type == 'File' and (not spec.content_type):
        spec.content_type = 'application/octet-stream'
    return (spec, warnings)

def safe_id_short(raw: str, fallback: str='Entity') -> str:
    import unicodedata
    text = unicodedata.normalize('NFKD', str(raw or '')).encode('ascii', 'ignore').decode()
    text = _IDSHORT_SAFE.sub('_', text).strip('_')
    text = re.sub('_+', '_', text)
    if not text or not text[0].isalpha():
        text = f'{fallback}_{text}'.strip('_')
    return text[:128]
_UNIT_ALIASES = {'kilogram': 'kg', 'gram': 'g', 'milligram': 'mg', 'tonne': 't', 'volt': 'v', 'millivolt': 'mv', 'kilovolt': 'kv', 'ampere': 'a', 'milliampere': 'ma', 'watt': 'w', 'milliwatt': 'mw', 'kilowatt': 'kw', 'amperehour': 'ah', 'milliamperehour': 'mah', 'watthour': 'wh', 'kilowatthour': 'kwh', 'ohm': 'ohm', 'Ω': 'ohm', 'Ω': 'ohm', 'henry': 'h_ind', 'millihenry': 'mh', 'farad': 'f_cap', 'microfarad': 'µf', 'second': 's', 'minute': 'min', 'minuteunitoftime': 'min', 'hour': 'h', 'day': 'd', 'week': 'wk', 'month': 'mo', 'months': 'mo', 'year': 'y', 'years': 'y', 'annum': 'y', 'a': 'y', 'metre': 'm', 'meter': 'm', 'millimetre': 'mm', 'millimeter': 'mm', 'centimetre': 'cm', 'centimeter': 'cm', 'kilometre': 'km', 'kilometer': 'km', 'percent': '%', 'percentage': '%', 'cycle': 'cycles', 'piece': 'pcs', 'pieces': 'pcs', 'degreecelsius': '°c', 'celsius': '°c', 'degc': '°c', 'c': '°c', 'degreefahrenheit': '°f', 'fahrenheit': '°f', 'degf': '°f', 'kelvin': 'k'}
_UNIT_LOOKALIKES = str.maketrans({'º': '°', 'Ω': 'Ω', 'μ': 'µ'})

def _normalise_unit(unit: str) -> str:
    import unicodedata
    text = unicodedata.normalize('NFC', (unit or '').strip())
    text = text.translate(_UNIT_LOOKALIKES)
    text = re.sub('[\\s._]', '', text).lower()
    if not text:
        return ''
    if text in _UNIT_ALIASES:
        return _UNIT_ALIASES[text]
    stripped = text.lstrip('°')
    return _UNIT_ALIASES.get(stripped, text)

def unit_warning(entity_unit: str, target_unit: str) -> Optional[str]:
    left, right = (_normalise_unit(entity_unit), _normalise_unit(target_unit))
    if not left or not right or left == right:
        return None
    return f'unit mismatch: document says {entity_unit!r}, element declares {target_unit!r}'

def partition(candidates: Sequence[Candidate]) -> Tuple[List[Candidate], List[Candidate]]:
    leaves = [c for c in candidates if c.model_type not in CONTAINER_TYPES + NON_CANDIDATE_TYPES]
    containers = [c for c in candidates if c.model_type in CONTAINER_TYPES]
    return (leaves, containers)

def band_for(score: float, theta_high: float, theta_low: float) -> str:
    if score >= theta_high:
        return 'auto'
    if score < theta_low:
        return 'below'
    return 'arbitrated'

def _as_candidate(hit: Any) -> Candidate:
    return Candidate(path=hit.node.path, id_short=hit.node.id_short, model_type=hit.node.model_type, score=hit.score, parent=hit.parent.path if hit.parent else None, unit=hit.node.unit, definition=hit.node.definition or hit.node.description, type_value_list_element=hit.node.type_value_list_element)

def candidates_for(index: Index, query_vector: Any, top_k: int, *, host_k: int=3) -> List[Candidate]:
    # Rank writable targets independently of operations and containers.
    hits = search(index, query_vector, top_k=len(index.nodes))
    out = [_as_candidate(h) for h in hits
           if h.node.model_type not in CONTAINER_TYPES + NON_CANDIDATE_TYPES][:top_k]
    if host_k > 0:
        seen = {c.path for c in out}
        for kind in CONTAINER_TYPES:
            for hit in search(index, query_vector, top_k=host_k, where=Filter(model_type=kind)):
                if hit.node.path in seen:
                    continue
                seen.add(hit.node.path)
                out.append(_as_candidate(hit))
    return out

def decide(entity: Entity, candidates: Sequence[Candidate], *, theta_high: float=THETA_HIGH, theta_low: float=THETA_LOW, arbitrate: Optional[Any]=None, create_in: Optional[str]=None, host: Optional[Any]=None) -> Decision:
    leaves, containers = partition(candidates)
    shown = [{'path': c.path, 'score': round(c.score, 4), 'modelType': c.model_type, 'unit': c.unit} for c in candidates]

    def skip(reason: str, band: str, score: float=0.0) -> Decision:
        return Decision(entity=entity.entity, value=entity.value, unit=entity.unit, source=entity.source, action='skip', band=band, reason=reason, score=score, margin=margin_of(leaves), candidates=shown)

    def write(target: Candidate, band: str, reason: str) -> Decision:
        warnings = []
        note = unit_warning(entity.unit, target.unit)
        if note:
            warnings.append(note)
        if entity.value is None:
            warnings.append('entity carries no value; nothing would be written')
        if not entity.source:
            warnings.append('no source recorded for this value')
        shape = infer_element_type(entity)
        min_value = max_value = content_type = None
        if target.model_type == 'Range':
            if shape.model_type == 'Range':
                min_value, max_value = (shape.min_value, shape.max_value)
            else:
                warnings.append(f'{target.path} is a Range, but {entity.value!r} gives no min and max; nothing can be written')
        elif target.model_type in ('File', 'Blob'):
            content_type = shape.content_type or 'application/octet-stream'
        return Decision(entity=entity.entity, value=entity.value, unit=entity.unit, source=entity.source, action='write', band=band, reason=reason, path=target.path, model_type=target.model_type, id_short=target.id_short, value_type=infer_value_type(entity.value, entity.unit), min_value=min_value, max_value=max_value, content_type=content_type, score=target.score, margin=margin_of(leaves), warnings=warnings, candidates=shown)
    chosen: Optional[Candidate] = None
    band = 'none'
    reason = 'no candidate element'
    if leaves:
        best = leaves[0]
        band = band_for(best.score, theta_high, theta_low)
        if band == 'auto':
            return write(best, band, f'similarity {best.score:.4f} >= {theta_high}')
        if band == 'arbitrated' and arbitrate is not None:
            picked = arbitrate(entity, leaves)
            if picked is not None:
                chosen, reason = picked
                if chosen is not None:
                    return write(chosen, 'arbitrated', reason)
            else:
                reason = 'arbitration returned no choice'
        elif band == 'arbitrated':
            return skip(f'similarity {best.score:.4f} is in the arbitration band [{theta_low}, {theta_high}) and no arbitration model was available', 'arbitrated', best.score)
        else:
            reason = f'similarity {best.score:.4f} < {theta_low}'
    target_container: Optional[Candidate] = None
    host_reason = ''
    chosen: Optional[ElementSpec] = None
    if host is not None and containers:
        picked = host(entity, containers)
        if picked is not None:
            target_container, host_reason = (picked[0], picked[1])
            if len(picked) > 2:
                chosen = picked[2]
    if target_container is None and create_in:
        target_container = Candidate(path=create_in, id_short=create_in.rsplit('.', 1)[-1], model_type='SubmodelElementCollection', score=0.0, parent=None)
        host_reason = f'no element matched; --create-in {create_in}'
    if target_container is None:
        return skip(reason, band if band != 'none' else 'none', leaves[0].score if leaves else 0.0)
    spec, warnings = resolve_spec(entity, target_container, chosen)
    if not entity.source:
        warnings.append('no source recorded for this value')
    in_list = target_container.model_type == 'SubmodelElementList'
    placement = host_reason or reason
    return Decision(entity=entity.entity, value=entity.value, unit=entity.unit, source=entity.source, action='create', band='hosted', reason=f'{placement}; {spec.reason}' if spec.reason else placement, path=target_container.path, model_type=spec.model_type, container_type=target_container.model_type, id_short=None if in_list else safe_id_short(entity.entity), value_type=spec.value_type, min_value=spec.min_value, max_value=spec.max_value, content_type=spec.content_type, score=leaves[0].score if leaves else 0.0, margin=margin_of(leaves), warnings=warnings, candidates=shown)
NOT_WRITABLE = CONTAINER_TYPES + NON_CANDIDATE_TYPES

def value_for_write(decision: Decision) -> Any:
    value = decision.value
    model_type = decision.model_type or 'Property'
    if decision.action == 'write' and decision.band == 'hosted':
        raise MatchError(f'{decision.path} was chosen as a *home* for a new element, not as a slot to write into, so this decision has to be a create. Re-run the match to rebuild the plan.')
    if model_type in NOT_WRITABLE:
        raise MatchError(f'{decision.path} is a {model_type}, which holds no value. ' + ('An Operation is invoked, not written to.' if model_type == 'Operation' else 'To put a value in here the decision has to be a create, not a write.'))
    if model_type == 'MultiLanguageProperty':
        return lang_strings(str(value))
    if model_type == 'Range':
        if decision.min_value is None or decision.max_value is None:
            raise MatchError(f'{decision.path} is a Range and the plan carries no min and max for {value!r}; edit the plan or write it with `aas set --json-value`')
        return {'min': decision.min_value, 'max': decision.max_value}
    if model_type in ('File', 'Blob'):
        return {'contentType': decision.content_type or 'application/octet-stream', 'value': value}
    if model_type == 'ReferenceElement':
        return external_reference(str(value))
    return value

def element_for_create(decision: Decision) -> Dict[str, Any]:
    model_type = decision.model_type or 'Property'
    value: Any = decision.value
    if model_type == 'MultiLanguageProperty' and value is not None:
        value = [{'language': language, 'text': text} for entry in lang_strings(str(value)) for language, text in entry.items()]
    return make_element(model_type, id_short=decision.id_short, value=value, value_type=decision.value_type or 'xs:string', description=decision.source or None, min_value=decision.min_value, max_value=decision.max_value, content_type=decision.content_type)
ARBITRATE_PROMPT = 'You map entities extracted from a supplier document onto elements of an Asset Administration Shell.\n\nGiven a QUERY entity and a list of CANDIDATE elements, choose the single candidate that is the correct semantic match. The query\'s value will be written into whichever you choose, so a wrong choice writes wrong data into a machine\'s model. Be strict: if no candidate is right, answer -1.\n\nWeigh `within` heavily. It names the collection that contains the candidate, which is the semantic boundary the element lives in. An entity that belongs to that boundary is far more likely to be the right match than one that merely shares a word with the candidate\'s name.\n\nReturn ONLY JSON:\n{"index": <int, or -1>, "reason": "<one short sentence>"}'
HOST_PROMPT = 'You map entities extracted from a supplier document onto an Asset Administration Shell.\n\nNo existing element matches this QUERY entity, so it has to be created. Answer two things about it.\n\n**Where.** Choose the single CANDIDATE collection that should contain it. Pick the collection whose subject the entity belongs to -- a charging power belongs among technical properties, a contact address does not. If none of them is a defensible home, answer index -1: creating an element in the wrong collection is worse than reporting that there is no place for it.\n\n**What.** Choose the modelType that can actually hold this value:\n\n* `Property` -- one scalar. The default; choose it whenever the others do not   clearly apply.\n* `Range` -- the value is an interval with two ends ("250-550", "-20 to +60").   Give `min` and `max` separately, as bare numbers with no unit.\n* `MultiLanguageProperty` -- the value is human-readable text that is or should   be given per language. A name, a designation, a description.\n* `File` -- the value is a path or URL to a document. Give `contentType`.\n* `ReferenceElement` -- the value identifies another element or a global   concept, and is meant to be followed rather than read.\n\nSHAPE is what reading the value alone suggests. It has no idea what the entity *means*, which you do -- correct it when the meaning says otherwise, and keep it when it does not.\n\nIf a candidate reports `holds`, it is a list and every child of it must have that modelType. Choosing it means choosing that type.\n\nReturn ONLY JSON:\n{"index": <int, or -1>, "modelType": "<one of the five above>", "valueType": "<xsd type, only for Property or Range>", "min": "<only for Range>", "max": "<only for Range>", "contentType": "<only for File>", "reason": "<one short sentence>"}'
CHOICE_SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['index', 'reason', 'modelType', 'valueType', 'min', 'max', 'contentType'], 'properties': {'index': {'type': 'integer'}, 'reason': {'type': 'string'}, 'modelType': {'type': ['string', 'null'], 'enum': [*CREATABLE_TYPES, None]}, 'valueType': {'type': ['string', 'null']}, 'min': {'type': ['string', 'null']}, 'max': {'type': ['string', 'null']}, 'contentType': {'type': ['string', 'null']}}}

def _ask(provider, config, system, payload):
    answer = provider(system, payload, CHOICE_SCHEMA)
    if not isinstance(answer, dict):
        raise MatchError('Model response must be a JSON object')
    return answer

def margin_of(leaves: Sequence[Candidate]) -> float:
    return round(leaves[0].score - leaves[1].score, 4) if len(leaves) > 1 else 0.0

def _payload(entity: Entity, candidates: Sequence[Candidate]) -> Dict[str, Any]:
    return {'QUERY': {'name': entity.entity, 'description': entity.description[:400], 'unit': entity.unit}, 'CANDIDATES': [{'index': i, 'idShort': c.id_short, 'path': c.path, 'within': c.parent or '<submodel root>', 'meaning': (c.definition or '')[:300], 'unit': c.unit, 'similarity': round(c.score, 4), **({'holds': c.type_value_list_element} if c.type_value_list_element else {})} for i, c in enumerate(candidates)]}

def make_host_chooser(provider: Any, config: Any, prompt: str=HOST_PROMPT):

    def host(entity: Entity, candidates: Sequence[Candidate]) -> Optional[Tuple[Optional[Candidate], str, Optional[ElementSpec]]]:
        payload = _payload(entity, candidates)
        shape = infer_element_type(entity)
        payload['SHAPE'] = {'modelType': shape.model_type, 'valueType': shape.value_type, 'min': shape.min_value, 'max': shape.max_value, 'contentType': shape.content_type, 'why': shape.reason or 'value reads as a plain scalar'}
        try:
            answer = _ask(provider, config, prompt, payload)
        except Exception as exc:
            return (None, f'hosting failed: {exc}', None)
        try:
            index = int(answer.get('index', -1))
        except (TypeError, ValueError):
            index = -1
        reason = str(answer.get('reason') or '').strip()[:200]
        if not 0 <= index < len(candidates):
            return (None, reason or 'no collection is a defensible home', None)
        model_type = str(answer.get('modelType') or '').strip()
        notes = []
        if model_type and model_type not in CREATABLE_TYPES:
            notes.append(f'model asked for {model_type!r}, which is not creatable here')
            model_type = ''
        spec = ElementSpec(model_type=model_type, value_type=str(answer['valueType']).strip() if answer.get('valueType') else None, min_value=None if answer.get('min') is None else str(answer['min']).strip(), max_value=None if answer.get('max') is None else str(answer['max']).strip(), content_type=str(answer['contentType']).strip() if answer.get('contentType') else None, reason='; '.join(notes))
        return (candidates[index], reason or 'chosen by hosting', spec)
    return host

def make_arbitrator(provider: Any, config: Any, prompt: str=ARBITRATE_PROMPT):

    def arbitrate(entity: Entity, candidates: Sequence[Candidate]) -> Optional[Tuple[Optional[Candidate], str]]:
        try:
            answer = _ask(provider, config, prompt, _payload(entity, candidates))
        except Exception as exc:
            return (None, f'arbitration failed: {exc}')
        try:
            index = int(answer.get('index', -1))
        except (TypeError, ValueError):
            index = -1
        reason = str(answer.get('reason') or '').strip()[:200]
        if 0 <= index < len(candidates):
            return (candidates[index], reason or 'chosen by arbitration')
        return (None, reason or 'arbitration found no suitable candidate')
    return arbitrate
