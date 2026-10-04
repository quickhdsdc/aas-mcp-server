"""Standalone AAS index algorithms, adapted from the updated aasctl implementation."""

from __future__ import annotations
import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
DEFAULT_EMBEDDING_MODEL = 'text-embedding-3-large'
VALUE_CHARS = 120
DEFAULT_INDEX_HOME = Path(os.environ.get('MCP_AAS_DATA_DIR', '.mcp_aas')) / 'semantic'
_CHILD_KEYS = ('submodelElements', 'value', 'statements', 'annotations')
_WORD_BOUNDARY = re.compile('(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|_+')
_CITATION = re.compile('\\s*\\b[\\w .\\-]{0,40}?(chapter|section|clause|paragraph)\\s+reference\\s*:?[^\\n]*', re.IGNORECASE)

def _split_words(raw: str) -> str:
    return _WORD_BOUNDARY.sub(' ', raw or '').strip()

class IndexError_(RuntimeError):
    pass

@dataclass
class Node:
    path: str
    id_short: Optional[str]
    model_type: str
    parent: Optional[str]
    description: str = ''
    semantic_id: Optional[str] = None
    definition: str = ''
    preferred_name: str = ''
    unit: str = ''
    value: Optional[str] = None
    submodel: Optional[str] = None
    submodel_id_short: Optional[str] = None
    submodel_semantic_id: Optional[str] = None
    value_type: Optional[str] = None
    type_value_list_element: Optional[str] = None

    @property
    def in_list(self) -> bool:
        return '[' in self.path

    def descriptor(self, with_values: bool=False, *, parent_depth: int=1, include_submodel: bool=False) -> str:
        title = _split_words(self.preferred_name or self.id_short or self.path)
        text = _CITATION.sub('', self.definition or self.description or '').strip()
        out = f'title: {title}, description: {text}'
        if self.unit:
            out += f', unit: {self.unit}'
        within = self.ancestry(parent_depth)
        if within:
            out += f', within: {within}'
        if include_submodel and self.submodel_id_short:
            out += f', submodel: {_split_words(self.submodel_id_short)}'
        if self.id_short and self.preferred_name and (self.id_short != self.preferred_name):
            out += f', idShort: {self.id_short}'
        if (with_values or self.in_list) and self.value:
            out += f', value: {self.value[:VALUE_CHARS]}'
        return out

    def ancestry(self, depth: int=1) -> str:
        if not self.parent or depth == 0:
            return ''
        segments = [s for s in re.split('\\.|\\[\\d+\\]', self.parent) if s]
        if depth > 0:
            segments = segments[-depth:]
        return ' '.join((_split_words(s) for s in segments))

def walk(submodel: Dict[str, Any]) -> List[Node]:
    nodes: List[Node] = []
    owner = submodel.get('id')
    owner_short = submodel.get('idShort')
    owner_semantic = _semantic_id(submodel)

    def visit(node: Dict[str, Any], path: str, parent: Optional[str]) -> None:
        in_list = node.get('modelType') == 'SubmodelElementList'
        for key in _CHILD_KEYS:
            children = node.get(key)
            if not isinstance(children, list):
                continue
            for position, child in enumerate(children):
                if not isinstance(child, dict) or 'modelType' not in child:
                    continue
                if in_list:
                    child_path = f'{path}[{position}]'
                else:
                    id_short = child.get('idShort')
                    if not id_short:
                        continue
                    child_path = f'{path}.{id_short}' if path else id_short
                nodes.append(Node(path=child_path, id_short=child.get('idShort'), model_type=child.get('modelType', '?'), parent=path or None, description=_description(child), semantic_id=_semantic_id(child), value=_value_text(child), submodel=owner, submodel_id_short=owner_short, submodel_semantic_id=owner_semantic, value_type=child.get('valueType'), type_value_list_element=child.get('typeValueListElement')))
                visit(child, child_path, path or None)
    visit(submodel, '', None)
    return nodes

def _description(node: Dict[str, Any]) -> str:
    for entry in node.get('description') or []:
        if isinstance(entry, dict) and entry.get('text'):
            return str(entry['text'])
    return ''

def _semantic_id(node: Dict[str, Any]) -> Optional[str]:
    keys = (node.get('semanticId') or {}).get('keys') or []
    return keys[0].get('value') if keys else None

def _value_text(node: Dict[str, Any]) -> Optional[str]:
    model_type = node.get('modelType')
    if model_type == 'MultiLanguageProperty':
        parts = node.get('value') or []
        return '; '.join((f'{k}: {v}' for entry in parts if isinstance(entry, dict) for k, v in entry.items())) or None
    if model_type == 'Range':
        return f"min={node.get('min')}, max={node.get('max')}"
    if model_type in ('SubmodelElementCollection', 'SubmodelElementList'):
        return None
    value = node.get('value')
    return None if value is None else str(value)

def enrich(nodes: Sequence[Node], concepts: Dict[str, Dict[str, str]]) -> int:
    hits = 0
    for node in nodes:
        concept = concepts.get((node.semantic_id or '').strip())
        if not concept:
            continue
        node.definition = concept.get('definition', '')
        node.preferred_name = concept.get('preferred_name', '')
        node.unit = concept.get('unit', '')
        hits += 1
    return hits

def concept_map(concept_descriptions: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for concept in concept_descriptions:
        identifier = (concept.get('id') or '').strip()
        if not identifier:
            continue
        content: Dict[str, Any] = {}
        for spec in concept.get('embeddedDataSpecifications') or []:
            candidate = spec.get('dataSpecificationContent') or {}
            if candidate.get('modelType') == 'DataSpecificationIec61360':
                content = candidate
                break
        out[identifier] = {'definition': _lang_text(content.get('definition')), 'preferred_name': _lang_text(content.get('preferredName')), 'unit': str(content.get('unit') or '')}
    return out

def _lang_text(entries: Any, language: str='en') -> str:
    if not isinstance(entries, list):
        return ''
    for entry in entries:
        if isinstance(entry, dict) and entry.get('language') == language and entry.get('text'):
            return str(entry['text'])
    for entry in entries:
        if isinstance(entry, dict) and entry.get('text'):
            return str(entry['text'])
    return ''

@dataclass
class Index:
    submodel_id: str
    id_short: Optional[str]
    model: str
    nodes: List[Node] = field(default_factory=list)
    vectors: Any = None
    enriched: int = 0
    with_values: bool = False
    built_at: float = field(default_factory=time.time)

    def save(self, home: Path=DEFAULT_INDEX_HOME) -> Path:
        import numpy as np
        home.mkdir(parents=True, exist_ok=True)
        stem = home / _slug(self.submodel_id)
        stem.with_suffix('.json').write_text(json.dumps({'submodel_id': self.submodel_id, 'idShort': self.id_short, 'model': self.model, 'enriched': self.enriched, 'with_values': self.with_values, 'built_at': self.built_at, 'nodes': [asdict(n) for n in self.nodes]}, ensure_ascii=False), encoding='utf-8')
        np.save(stem.with_suffix('.npy'), self.vectors)
        return stem.with_suffix('.json')

    @classmethod
    def load(cls, submodel_id: str, home: Path=DEFAULT_INDEX_HOME) -> 'Index':
        import numpy as np
        stem = home / _slug(submodel_id)
        meta_path = stem.with_suffix('.json')
        if not meta_path.exists():
            raise IndexError_(f'no index for {submodel_id!r}. Build one first: aas index {submodel_id}')
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        owner = meta['submodel_id']
        known = set(Node.__dataclass_fields__)
        nodes: List[Node] = []
        for raw in meta['nodes']:
            node = Node(**{k: v for k, v in raw.items() if k in known})
            if node.submodel is None and owner != ALL:
                node.submodel = owner
                node.submodel_id_short = node.submodel_id_short or meta.get('idShort')
            nodes.append(node)
        return cls(submodel_id=owner, id_short=meta.get('idShort'), model=meta.get('model', DEFAULT_EMBEDDING_MODEL), nodes=nodes, vectors=np.load(stem.with_suffix('.npy')), enriched=meta.get('enriched', 0), with_values=meta.get('with_values', False), built_at=meta.get('built_at', 0.0))

    @staticmethod
    def drop(submodel_id: str, home: Path=DEFAULT_INDEX_HOME) -> bool:
        stem = home / _slug(submodel_id)
        removed = False
        for suffix in ('.json', '.npy'):
            target = stem.with_suffix(suffix)
            if target.exists():
                target.unlink()
                removed = True
        return removed

    @staticmethod
    def cached(home: Path=DEFAULT_INDEX_HOME) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        if not home.exists():
            return rows
        for meta_path in sorted(home.glob('*.json')):
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            rows.append({'submodel_id': meta['submodel_id'], 'idShort': meta.get('idShort'), 'nodes': len(meta.get('nodes') or []), 'enriched': meta.get('enriched', 0), 'with_values': meta.get('with_values', False), 'built_at': meta.get('built_at', 0.0), 'model': meta.get('model'), 'path': str(meta_path)})
        return rows

def _slug(submodel_id: str) -> str:
    digest = hashlib.sha256(submodel_id.encode('utf-8')).hexdigest()[:12]
    readable = re.sub('[^A-Za-z0-9]+', '-', submodel_id).strip('-')[-40:]
    return f'{readable}-{digest}' if readable else digest

@dataclass
class Hit:
    node: Node
    score: float
    parent: Optional[Node] = None
    siblings: List[str] = field(default_factory=list)
    also_in: List[str] = field(default_factory=list)

    def block(self) -> List[str]:
        lines = [f'{self.score:.3f}  {self.node.path}  ({self.node.model_type})']
        label = self.node.preferred_name or self.node.id_short
        if label and label != self.node.path:
            lines.append(f'        name:     {label}')
        if self.node.definition or self.node.description:
            lines.append(f'        meaning:  {(self.node.definition or self.node.description)[:150]}')
        if self.node.unit:
            lines.append(f'        unit:     {self.node.unit}')
        if self.node.semantic_id:
            lines.append(f'        semantic: {self.node.semantic_id}')
        if self.node.value is not None:
            lines.append(f'        value:    {self.node.value[:80]}')
        if self.parent is not None:
            lines.append(f"        within:   {self.parent.path or '<submodel root>'} ({self.parent.model_type})")
        if self.also_in:
            shown = ', '.join(self.also_in[:3])
            more = f' (+{len(self.also_in) - 3} more)' if len(self.also_in) > 3 else ''
            lines.append(f'        also in:  {shown}{more}')
        if self.siblings:
            shown = ', '.join(self.siblings[:12])
            more = f' (+{len(self.siblings) - 12} more)' if len(self.siblings) > 12 else ''
            lines.append(f'        beside:   {shown}{more}')
        return lines

@dataclass
class Filter:
    semantic_id: Optional[str] = None
    model_type: Optional[str] = None
    value_type: Optional[str] = None
    unit: Optional[str] = None
    submodel: Optional[str] = None
    in_template: Optional[str] = None
    id_short_like: Optional[str] = None
    has_value: Optional[bool] = None
    under: Optional[str] = None

    def empty(self) -> bool:
        return all((getattr(self, f) is None for f in self.__dataclass_fields__))

    def keeps(self, node: Node) -> bool:
        if self.semantic_id and (node.semantic_id or '') != self.semantic_id:
            return False
        if self.model_type and node.model_type != self.model_type:
            return False
        if self.value_type and (node.value_type or '') != self.value_type:
            return False
        if self.unit and self.unit.lower() not in (node.unit or '').lower():
            return False
        if self.submodel and (node.submodel or '') != self.submodel:
            return False
        if self.in_template and (node.submodel_semantic_id or '') != self.in_template:
            return False
        if self.id_short_like and self.id_short_like.lower() not in (node.id_short or '').lower():
            return False
        if self.has_value is not None and bool(node.value) != self.has_value:
            return False
        if self.under and (not (node.path == self.under or node.path.startswith(f'{self.under}.') or node.path.startswith(f'{self.under}['))):
            return False
        return True

    def describe(self) -> str:
        parts = [f'{name}={value!r}' for name, value in ((f, getattr(self, f)) for f in self.__dataclass_fields__) if value is not None]
        return ', '.join(parts) or '(none)'

def _dedupe_key(node: Node) -> Any:
    if hasattr(node, '_relative_path'):
        return (node._relative_path, node.model_type, node.semantic_id,
                node.submodel_semantic_id or node.submodel)
    return (node.path, node.model_type)

def rows_matching(index: Index, where: Optional[Filter]) -> List[int]:
    if where is None or where.empty():
        return list(range(len(index.nodes)))
    return [i for i, node in enumerate(index.nodes) if where.keeps(node)]

def search(index: Index, query_vector: Any, top_k: int=5, min_score: float=0.0, where: Optional[Filter]=None, dedupe: bool=True) -> List[Hit]:
    import numpy as np
    if index.vectors is None or len(index.nodes) == 0:
        return []
    rows = rows_matching(index, where)
    if not rows:
        return []
    corpus = _normalise(np.asarray(index.vectors, dtype='float32')[rows])
    query = _normalise(np.asarray(query_vector, dtype='float32').reshape(1, -1))[0]
    scores = corpus @ query
    order = np.argsort(-scores)
    by_path = {(n.submodel, n.path): n for n in index.nodes}
    wanted = max(top_k, 0)
    horizon = max(wanted * 20, 200)
    hits: List[Hit] = []
    seen: Dict[tuple, int] = {}
    for scanned, position in enumerate(order):
        if len(hits) >= wanted and (not dedupe or scanned >= horizon):
            break
        score = float(scores[position])
        if score < min_score:
            break
        node = index.nodes[rows[int(position)]]
        if dedupe:
            key = _dedupe_key(node)
            if key in seen:
                kept = hits[seen[key]]
                if node.submodel and node.submodel not in kept.also_in:
                    kept.also_in.append(node.submodel)
                continue
            if len(hits) >= wanted:
                continue
            seen[key] = len(hits)
        parent = by_path.get((node.submodel, node.parent)) if node.parent else None
        hits.append(Hit(node=node, score=score, parent=parent, siblings=_siblings(index, node)))
    return hits[:wanted]

def _siblings(index: Index, node: Node) -> List[str]:
    out: List[str] = []
    for other in index.nodes:
        if other.submodel == node.submodel and other.parent == node.parent and (other.path != node.path):
            out.append(other.id_short or other.path.rsplit('.', 1)[-1])
    return out

def _normalise(matrix: Any) -> Any:
    import numpy as np
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)
ALL = '*'
