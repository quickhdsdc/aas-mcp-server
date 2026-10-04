"""Regression tests for MCP adapters without external model calls."""
import json
from pathlib import Path

import numpy as np
import pytest

from mcp_aas.config import LLMSettings
from mcp_aas.semantic import runtime
from mcp_aas.semantic.index import Index, Node
from mcp_aas.semantic.match import candidates_for, make_arbitrator, make_host_chooser, Entity
from mcp_aas.semantic.values import typed_value
from mcp_aas.tools.aas_populate_inputs import target_path
from mcp_aas.aas_utils.basyx_client import encode_id


def snapshot(tmp_path):
    raw = {"assetAdministrationShells": [{"id": "urn:aas:A", "idShort": "A",
        "submodels": [{"keys": [{"type": "Submodel", "value": "urn:sm"}]}]}],
        "submodels": [{"id": "urn:sm", "idShort": "TechnicalData", "submodelElements": [
            {"modelType": "SubmodelElementCollection", "idShort": "Group", "value": [
                {"modelType": "Property", "idShort": "RatedCapacity", "valueType": "xs:double", "value": "1"}]}]}]}
    (tmp_path / "A.json").write_text(json.dumps(raw), encoding="utf-8")
    return raw


@pytest.mark.asyncio
async def test_corpus_cache_reused_and_invalidated_by_descriptor_or_provider(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "TEMP_DIR", str(tmp_path))
    monkeypatch.setattr(runtime, "profile", lambda *_: LLMSettings(api_type="openai", api_key="test", embedding_model="test"))
    raw = snapshot(tmp_path)
    calls = []
    def embed(texts):
        calls.append(list(texts))
        return np.ones((len(texts), 3), dtype="float32")
    monkeypatch.setattr(runtime, "embed", embed)
    await runtime.load_index("A")
    await runtime.load_index("A")
    assert len(calls) == 1
    assert "within: Group" in calls[0][1]
    assert "urn:sm" not in calls[0][1]
    raw['submodels'][0]['submodelElements'][0]['value'][0]['description'] = [{"language": "en", "text": "changed"}]
    (tmp_path / 'A.json').write_text(json.dumps(raw), encoding='utf-8')
    await runtime.load_index('A')
    assert len(calls) == 2
    monkeypatch.setattr(runtime, "profile", lambda *_: LLMSettings(api_type="openai", api_key="test", embedding_model="different"))
    await runtime.load_index('A')
    assert len(calls) == 3


def test_exact_cache_label_never_selects_another_asset(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "TEMP_DIR", str(tmp_path))
    snapshot(tmp_path)
    with pytest.raises(FileNotFoundError):
        runtime.snapshot_nodes('AA')
    with pytest.raises(ValueError):
        runtime.cache_file('../A', '.json')


def test_matching_does_not_let_operations_starve_writable_or_container_candidates():
    nodes = [Node('Run', 'Run', 'Operation', None), Node('Capacity', 'Capacity', 'Property', None),
             Node('Group', 'Group', 'SubmodelElementCollection', None)]
    index = Index('urn:sm', 'SM', 'test', nodes=nodes, vectors=np.array([[1,0], [.8,.6], [0,1]]))
    candidates = candidates_for(index, np.array([1,0]), top_k=1)
    assert [c.model_type for c in candidates] == ['Property', 'SubmodelElementCollection']


def test_model_indices_are_validated_against_the_presented_candidates():
    provider = lambda *_: {'index': 99, 'reason': 'invalid index'}
    entity = Entity('Capacity', '1')
    from mcp_aas.semantic.match import Candidate
    candidates = [Candidate('Capacity', 'Capacity', 'Property', .6, None)]
    assert make_arbitrator(provider, None)(entity, candidates)[0] is None
    assert make_host_chooser(provider, None)(entity, candidates)[0] is None


def test_value_only_payloads_and_non_writable_rejection():
    assert typed_value({'modelType':'MultiLanguageProperty'}, [{'language':'en','text':'Hello'}]) == [{'en':'Hello'}]
    assert typed_value({'modelType':'Range'}, {'min':-10,'max':50}) == {'min':'-10','max':'50'}
    assert typed_value({'modelType':'Property'}, True) == 'true'
    with pytest.raises(ValueError):
        typed_value({'modelType':'Operation'}, 'start')


def test_population_rejects_cross_asset_and_external_targets():
    path = '/submodels/' + encode_id('urn:sm') + '/submodel-elements/Capacity'
    assert target_path({'API_path': path}, {'urn:sm'}) == path
    with pytest.raises(ValueError, match='not linked'):
        target_path({'API_path': path}, {'urn:other'})
    with pytest.raises(ValueError, match='relative'):
        target_path({'API_path': 'http://example.com'+path}, {'urn:sm'})
    with pytest.raises(ValueError, match='Invalid AAS'):
        target_path({'API_path': path + '/../../shells'}, {'urn:sm'})


@pytest.mark.asyncio
async def test_matching_preserves_unit_warnings_and_recomputes_edited_inputs(tmp_path, monkeypatch):
    from mcp_aas.tools import aas_match_inputs as tool
    monkeypatch.setattr(runtime, 'TEMP_DIR', str(tmp_path))
    monkeypatch.setattr(tool, 'TEMP_DIR', str(tmp_path))
    monkeypatch.setattr(tool, 'ATTACHMENTS_DIR', str(tmp_path))
    monkeypatch.setattr(runtime, 'profile', lambda *_: LLMSettings(api_type='openai', api_key='test'))
    raw = snapshot(tmp_path)
    raw['submodels'][0]['submodelElements'][0]['value'][0]['semanticId'] = {'keys':[{'value':'urn:capacity'}]}
    raw['conceptDescriptions'] = [{'id':'urn:capacity', 'embeddedDataSpecifications':[{
        'dataSpecificationContent':{'modelType':'DataSpecificationIec61360','unit':'Wh'}}]}]
    (tmp_path/'A.json').write_text(json.dumps(raw),encoding='utf-8')
    monkeypatch.setattr(runtime,'embed',lambda texts: np.tile([1.,0.,0.],(len(texts),1)))
    source = tmp_path/'entities.json'
    for value in ('1','2'):
        source.write_text(json.dumps([{'entity':'Capacity','value':value,'unit':'kWh','source':'test'}]),encoding='utf-8')
        result = await tool.AASMatchInputs().execute('entities.json','A',no_arbitration=True)
        assert result.error is None
        plan = json.loads((tmp_path/'entities_A_matching_result.json').read_text(encoding='utf-8'))
        assert plan[0]['decision']['value'] == value
        assert any('unit mismatch' in note for note in plan[0]['warnings'])


def test_named_azure_and_separate_embedding_profiles(monkeypatch):
    chat = LLMSettings(api_type='azure',api_key='test',model='chat')
    embedding = LLMSettings(api_type='openai',api_key='test',embedding_model='embedding')
    monkeypatch.setattr(runtime.config._config,'llm',{'azure':chat,'embedding':embedding})
    assert runtime.profile() is chat
    assert runtime.profile(True) is embedding


@pytest.mark.asyncio
async def test_write_http_failure_is_not_silently_reported_as_success():
    import httpx
    from mcp_aas.aas_utils.basyx_client import BasyxApiClient
    client = BasyxApiClient('http://test')
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(400)))
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await client.patch('/submodels/test/submodel-elements/P/$value', '"x"', reraise=True)
    finally:
        await client.client.aclose()
