"""Opt-in MCP + BaSyx integration, using a local deterministic model endpoint.

MCP_AAS_TEST_ENDPOINT=http://localhost:8081 pytest tests/test_mcp_live.py
Creates unique synthetic assets and removes them in a finally block.
No real model credentials or external model endpoint is used.
"""
import asyncio
import base64
import json
import os
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

pytestmark = pytest.mark.skipif(not os.getenv('MCP_AAS_TEST_ENDPOINT'), reason='Set MCP_AAS_TEST_ENDPOINT for live BaSyx testing')


def vector(text):
    text = text.lower()
    if 'ambiguous' in text:
        return [.6, .6, .529]
    if 'pressure' in text:
        return [-1., 0., 0.]
    if 'serial' in text:
        return [0., -1., 0.]
    if 'capacity' in text:
        return [1., 0., 0.]
    if 'label' in text:
        return [0., 1., 0.]
    return [0., 0., 1.]


@pytest.mark.asyncio
async def test_live_mcp_search_match_typed_population_and_cache(tmp_path):
    embedding_sizes = []
    chat_calls = []
    class ModelHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path.endswith('/embeddings'):
                texts = request['input']
                embedding_sizes.append(len(texts))
                result = {'object':'list', 'model':'test-embedding', 'data':[
                    {'object':'embedding','index':i,'embedding':vector(text)} for i,text in enumerate(texts)],
                    'usage':{'prompt_tokens':1,'total_tokens':1}}
            else:
                chat_calls.append(request)
                choice = {'index':-1,'reason':'no defensible match','modelType':None,'valueType':None,
                          'min':None,'max':None,'contentType':None}
                result = {'id':'test','object':'chat.completion','created':0,'model':'test-chat',
                    'choices':[{'index':0,'message':{'role':'assistant','content':json.dumps(choice)},'finish_reason':'stop'}]}
            body = json.dumps(result).encode()
            self.send_response(200)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *_):
            pass

    model = ThreadingHTTPServer(('127.0.0.1',0), ModelHandler)
    threading.Thread(target=model.serve_forever, daemon=True).start()
    endpoint = os.environ['MCP_AAS_TEST_ENDPOINT'].rstrip('/')
    label = 'MCPAlgorithmTest' + uuid.uuid4().hex[:10]
    aas_id, sm_id = 'urn:test:aas:'+label, 'urn:test:sm:'+label
    encode = lambda value: base64.urlsafe_b64encode(value.encode()).decode()
    sm_path, shell_path = '/submodels/'+encode(sm_id), '/shells/'+encode(aas_id)
    sm = {'modelType':'Submodel','id':sm_id,'idShort':'TechnicalData','kind':'Instance','submodelElements':[
        {'modelType':'Property','idShort':'Capacity','valueType':'xs:double','value':'10'},
        {'modelType':'MultiLanguageProperty','idShort':'Label','value':[{'language':'en','text':'Original'}]},
        {'modelType':'Range','idShort':'Temperature','valueType':'xs:double','min':'-10','max':'50'},
        {'modelType':'SubmodelElementCollection','idShort':'Specs','value':[]},
        {'modelType':'SubmodelElementList','idShort':'Records','typeValueListElement':'Property','valueTypeListElement':'xs:string','value':[]},
        {'modelType':'Operation','idShort':'Run','inputVariables':[],'outputVariables':[]},
    ]}
    shell = {'modelType':'AssetAdministrationShell','id':aas_id,'idShort':label,
             'assetInformation':{'assetKind':'Instance','globalAssetId':'urn:test:asset:'+label},
             'submodels':[{'type':'ModelReference','keys':[{'type':'Submodel','value':sm_id}]}]}
    config = tmp_path/'config.toml'
    config.write_text(f'[llm]\napi_type="openai"\napi_key="test"\nmodel="test-chat"\nembedding_model="test-embedding"\nbase_url="http://127.0.0.1:{model.server_port}/v1"\n', encoding='utf-8')
    data = tmp_path/'data'
    (data/'attachments').mkdir(parents=True)
    inputs = data/'attachments'/'entities.json'
    inputs.write_text(json.dumps([
        {'entity':'Capacity','value':'12','source':'synthetic test'},
        {'entity':'Pressure range','description':'Pressure interval','value':'5-10','unit':'bar','source':'synthetic test'},
    ]), encoding='utf-8')
    env = os.environ.copy()
    # Explicit test settings override inherited private runtime settings.
    for key in list(env):
        if key.startswith('MCP_AAS_'):
            env.pop(key)
    env.update(MCP_AAS_CONFIG_FILE=str(config), MCP_AAS_DATA_DIR=str(data),
               AAS_SERVER_ENDPOINT=endpoint, MCP_AAS_MANIFEST_POLL_INTERVAL='0')
    params = StdioServerParameters(command=sys.executable, args=['-m','mcp_aas'],
                                   cwd=str(Path(__file__).resolve().parents[1]), env=env)
    with httpx.Client(base_url=endpoint, timeout=15) as rest:
        try:
            rest.post('/submodels',json=sm).raise_for_status()
            rest.post('/shells',json=shell).raise_for_status()
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    names = {t.name for t in (await session.list_tools()).tools}
                    assert len(names) == 12
                    async def call(name, **args):
                        response = await asyncio.wait_for(session.call_tool(name,args), 60)
                        assert not response.isError, response
                        result = json.loads(response.content[0].text)
                        assert not result.get('error'), result
                        return result['output']
                    await call('aas_parse', aas_id=aas_id)
                    search = json.loads(await call('aas_search_property', aas_idShort=label,prop_name='capacity'))
                    assert search['results'][0]['idShort'] == 'Capacity'
                    assert search['results'][0]['similarity'] == pytest.approx(1.)
                    await call('aas_search_property',aas_idShort=label,prop_name='label')
                    assert embedding_sizes == [6,1,1]
                    await call('aas_match_inputs',aas_idShort=label,input_file_name='entities.json',
                               no_arbitration=True,create_in='Specs')
                    plan_path = data/'temp'/f'entities_{label}_matching_result.json'
                    decisions = json.loads(plan_path.read_text(encoding='utf-8'))
                    assert [d['decision']['action'] for d in decisions] == ['write','create']
                    assert decisions[1]['decision']['model_type'] == 'Range'
                    await call('aas_populate_inputs',aas_idShort=label,match_result_path=str(plan_path),dry_run=True)
                    assert rest.get(sm_path+'/submodel-elements/Capacity/$value').json() == '10'
                    await call('aas_populate_inputs',aas_idShort=label,match_result_path=str(plan_path))
                    assert rest.get(sm_path+'/submodel-elements/Capacity/$value').json() == '12'
                    pressure = rest.get(sm_path+'/submodel-elements/Specs.Pressure_range').json()
                    assert pressure['modelType'] == 'Range' and pressure['min'] == '5' and pressure['max'] == '10'
                    await call('aas_write_property',aas_idShort=label,updates=[
                        {'prop_idShort':'Label','value':[{'language':'en','text':'Updated'}]},
                        {'prop_idShort':'Temperature','value':{'min':-20,'max':60}},
                    ])
                    assert rest.get(sm_path+'/submodel-elements/Label').json()['value'] == [{'language':'en','text':'Updated'}]
                    assert rest.get(sm_path+'/submodel-elements/Temperature').json()['min'] == '-20'
                    inputs.write_text(json.dumps([{'entity':'Serial code','value':'007','source':'synthetic test'}]),encoding='utf-8')
                    await call('aas_match_inputs',aas_idShort=label,input_file_name='entities.json',
                               no_arbitration=True,create_in='Records')
                    await call('aas_populate_inputs',aas_idShort=label,match_result_path=str(plan_path))
                    child = rest.get(sm_path+'/submodel-elements/Records%5B0%5D').json()
                    assert child['value'] == '007' and not child.get('idShort')
                    inputs.write_text(json.dumps([{'entity':'Ambiguous measurement','value':'1','source':'synthetic test'}]),encoding='utf-8')
                    await call('aas_match_inputs',aas_idShort=label,input_file_name='entities.json')
                    assert json.loads(plan_path.read_text(encoding='utf-8'))[0]['decision']['action'] == 'skip'
                    assert len(chat_calls) == 2
                    assert all(request['response_format']['type'] == 'json_schema' for request in chat_calls)
        finally:
            rest.delete(shell_path)
            rest.delete(sm_path)
            model.shutdown()
            model.server_close()
            assert rest.get(shell_path).status_code == 404
            assert rest.get(sm_path).status_code == 404
