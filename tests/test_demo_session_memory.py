"""Execute the actual Demo JavaScript against a local, stubbed HTTP/DOM boundary."""
import shutil
import subprocess
from pathlib import Path

import pytest


def test_demo_reuses_session_on_reload_and_clears_only_after_server_success():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node.js unavailable for Demo JavaScript verification')
    script = r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
const html=fs.readFileSync(process.argv[1],'utf8');
const source=html.match(/<script>([\s\S]*?)<\/script>/)[1];
const storage=new Map(),requests=[];let failDelete=false,id=0;
function openTab(){
  const elements=new Map();
  function el(name){if(!elements.has(name))elements.set(name,{value:'test',files:[],hidden:false,disabled:false,classList:{add(){},remove(){}},addEventListener(event,handler){this[event]=handler},click(){return this.onclick()}});return elements.get(name)}
  const store={getItem:key=>storage.get(key)||null,setItem:(key,value)=>storage.set(key,value),removeItem:key=>storage.delete(key)};
  const context={document:{getElementById:el},localStorage:{getItem:()=>'',setItem(){}},sessionStorage:store,crypto:{getRandomValues:values=>{values.fill(++id);return values}},Uint8Array,
    fetch:async(path,opts)=>{if(path==='/health')return{json:async()=>({status:'ok',video_retrieval:{exact_ready:true}})};
      requests.push({path,opts});if(opts.method==='DELETE')return{ok:!failDelete,status:failDelete?503:204,text:async()=> 'storage unavailable'};
      return{ok:true,status:200,json:async()=>({data:{answer:'answer',route:'service',citations:[],manual_images:[],videos:[]}})};
    }};
  vm.runInNewContext(source,context);return el;
}
(async()=>{
  const tab=openTab();await tab('ask').onclick();const first=JSON.parse(requests.at(-1).opts.body).session_id;
  assert.ok(first);await tab('ask').onclick();assert.equal(JSON.parse(requests.at(-1).opts.body).session_id,first);
  const reopened=openTab();await reopened('ask').onclick();assert.equal(JSON.parse(requests.at(-1).opts.body).session_id,first);
  reopened('newSession').onclick();await reopened('ask').onclick();const second=JSON.parse(requests.at(-1).opts.body).session_id;assert.notEqual(second,first);
  failDelete=true;await reopened('clearSession').onclick();assert.equal(storage.get('rag_chat_session'),second);
  failDelete=false;await reopened('clearSession').onclick();assert.equal(requests.at(-1).path,'/v2/chat/sessions/'+second);assert.equal(storage.has('rag_chat_session'),false);
  assert.equal(reopened('ask').disabled,false);assert.equal(reopened('newSession').disabled,false);
  console.log('Demo session continuation, reload, new session, clear failure/success and non-TLS ID generation passed');
})().catch(error=>{console.error(error);process.exit(1)});
'''
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([node, '-e', script, str(root / 'web_demo/index.html')],
                            capture_output=True, text=True, timeout=15, check=False)
    assert result.returncode == 0, result.stderr
