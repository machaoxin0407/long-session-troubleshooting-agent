import copy
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from request_context import BUDGET_EVENTS, OPTIONAL_TEXTS, PROTECTED_TEXTS, ContextBudgetExceeded, prepare_body


@pytest.fixture
def limits(monkeypatch):
    monkeypatch.setenv('LLM_CONTEXT_WINDOW_TOKENS', '3000')
    monkeypatch.setenv('LLM_CONTEXT_SAFETY_TOKENS', '128')
    monkeypatch.setenv('LLM_IMAGE_TOKEN_RESERVE', '512')


def test_request_counts_system_tools_output_and_rejects_fixed_question(limits):
    with pytest.raises(ContextBudgetExceeded):
        prepare_body({'system': 'rule' * 700, 'messages': [{'role': 'user', 'content': 'question'}],
                      'tools': [{'name': 'search', 'description': 'rules' * 100}], 'max_tokens': 256})
    with pytest.raises(ContextBudgetExceeded):
        prepare_body({'messages': [{'role': 'user', 'content': '长问题' * 600}], 'max_tokens': 256})


def test_optional_history_is_removed_without_changing_current_question(limits):
    history = 'old history ' * 400
    question = 'Current ZX901 correction. Do not change this question.'
    original = {'messages': [{'role': 'user', 'content': history + question}], 'max_tokens': 256}
    old = copy.deepcopy(original)
    a, b, c = PROTECTED_TEXTS.set((question,)), OPTIONAL_TEXTS.set([(300, history)]), BUDGET_EVENTS.set([])
    try:
        prepared = prepare_body(original)
        assert question in prepared['messages'][0]['content']
        assert BUDGET_EVENTS.get()[-1]['within_budget']
        assert BUDGET_EVENTS.get()[-1]['removed_optional_parts'] == 1
        assert original == old
    finally:
        PROTECTED_TEXTS.reset(a)
        OPTIONAL_TEXTS.reset(b)
        BUDGET_EVENTS.reset(c)


@pytest.mark.parametrize('protocol', ['openai', 'anthropic'])
def test_old_tool_exchange_is_removed_as_a_unit_and_latest_source_retained(limits, protocol):
    messages = [{'role': 'user', 'content': 'fixed question'}]
    for name, text in [('old', 'obsolete evidence ' * 500), ('new', '[source=manual-7] latest evidence')]:
        if protocol == 'openai':
            messages.extend([{'role': 'assistant', 'tool_calls': [{'id': name, 'type': 'function',
                'function': {'name': 'search', 'arguments': '{}'}}], 'content': None},
                {'role': 'tool', 'tool_call_id': name, 'content': text}])
        else:
            messages.extend([{'role': 'assistant', 'content': [SimpleNamespace(type='tool_use', id=name,
                name='search', input={})]}, {'role': 'user', 'content': [
                {'type': 'tool_result', 'tool_use_id': name, 'content': text}]}])
    prepared = prepare_body({'messages': messages, 'max_tokens': 256})
    serialized = json.dumps(prepared)
    assert 'obsolete evidence' not in serialized
    assert 'manual-7' in serialized and 'latest evidence' in serialized
    assert 'fixed question' in serialized


def test_latest_tool_result_keeps_id_and_labelled_excerpt(limits):
    prepared = prepare_body({'messages': [{'role': 'user', 'content': 'fixed question'},
        {'role': 'assistant', 'tool_calls': [{'id': 'call-1', 'type': 'function',
            'function': {'name': 'search', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': 'call-1', 'content': '[source=manual-7]\n' + 'evidence ' * 900}],
        'max_tokens': 256})
    result = prepared['messages'][-1]
    assert result['tool_call_id'] == 'call-1'
    assert 'source=manual-7' in result['content']
    assert len(result['content']) < 7200


def test_image_payload_preserved_and_reserved_without_counting_base64_as_text(limits, monkeypatch):
    image = {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + 'A' * 10000}}
    body = {'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'question'}, image]}],
            'max_tokens': 256}
    assert prepare_body(body) == body
    monkeypatch.setenv('LLM_IMAGE_TOKEN_RESERVE', '3000')
    with pytest.raises(ContextBudgetExceeded):
        prepare_body(body)


def test_budget_boundary_is_inclusive(limits, monkeypatch):
    body = {'messages': [{'role': 'user', 'content': 'fixed question'}], 'max_tokens': 256}
    token = BUDGET_EVENTS.set([])
    try:
        prepare_body(body)
        event = BUDGET_EVENTS.get()[-1]
        exact = event['input_estimate'] + 256 + 128
        monkeypatch.setenv('LLM_CONTEXT_WINDOW_TOKENS', str(exact))
        assert prepare_body(body) == body
        monkeypatch.setenv('LLM_CONTEXT_WINDOW_TOKENS', str(exact - 1))
        with pytest.raises(ContextBudgetExceeded):
            prepare_body(body)
    finally:
        BUDGET_EVENTS.reset(token)


@pytest.mark.parametrize('endpoint', ['/chat', '/v2/chat'])
def test_oversized_api_question_is_413_without_models_or_saved_turn(tmp_path, monkeypatch, endpoint):
    from fastapi.testclient import TestClient
    import api_server as api
    from session_memory import SessionMemoryStore
    store = SessionMemoryStore(tmp_path / 'sessions.sqlite3')
    monkeypatch.setattr(api, '_SESSION_MEMORY', store)
    monkeypatch.setattr(api, 'EXPECTED_TOKEN', 'context-test')
    monkeypatch.setattr(api, '_write_api_error_trace', lambda **kw: None)
    monkeypatch.setattr(api, '_run_agent_impl', lambda *a: pytest.fail('must reject before models'))
    monkeypatch.setenv('LLM_CONTEXT_WINDOW_TOKENS', '9000')
    with TestClient(api.app) as client:
        response = client.post(endpoint, headers={'Authorization': 'Bearer context-test'},
                               json={'question': 'q' * 9000, 'session_id': 'oversize'})
    assert response.status_code == 413
    assert store.get('oversize') == []


def test_attempt_ledger_is_atomic_and_includes_failed_attempts(tmp_path, monkeypatch):
    from model_attempts import record_attempt
    ledger = tmp_path / 'attempts.json'
    ledger.write_text(json.dumps({'attempts': 0, 'limit': 7}))
    monkeypatch.setenv('MODEL_ATTEMPT_LEDGER', str(ledger))
    def attempt(_):
        try:
            record_attempt('test')
            return True
        except TimeoutError:
            return False
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(attempt, range(18))) == 7
    assert json.loads(ledger.read_text())['attempts'] == 7


def test_retrieval_trimming_preserves_sources_and_prefers_relevant_blocks():
    from request_context import _compact_retrieval
    text = '[SECTION_IDS] 1,2,3\n[1] 产品: ZX901 | 章节: 接线\nZX901 接线事实。\n'
    text += '[2] 产品: OTHER | 章节: 无关\n' + '无关背景。' * 200 + '\n'
    text += '[3] 产品: OTHER | 章节: 重复\n' + '无关背景。' * 200 + '\n'
    clipped, duplicates, omitted = _compact_retrieval(text, 500, ('ZX901 接线',))
    assert duplicates == 1 and omitted >= 1
    assert 'ZX901 接线事实。' in clipped
    assert all(f'[{n}] 产品:' in clipped for n in (1, 2, 3))
    assert '省略' in clipped and len(clipped.encode()) <= 500


@pytest.mark.parametrize('mode', ['ordinary', 'stream', 'no_retry', 'anthropic'])
def test_all_router_paths_reject_oversized_fixed_inputs_before_network(limits, monkeypatch, mode):
    import llm_router
    route = SimpleNamespace(base_url='http://127.0.0.1:1/v1', api_key='test', model='test', name='test',
                            protocol='anthropic' if mode == 'anthropic' else 'openai')
    monkeypatch.setattr(llm_router, '_active_routes', lambda: [route])
    monkeypatch.setattr(llm_router, '_get_openai_client', lambda *a: pytest.fail('must not start network'))
    common = {'system': 'fixed rule' * 600, 'messages': [{'role': 'user', 'content': 'question'}], 'max_tokens': 256}
    if mode == 'anthropic':
        import sys
        class Client:
            def __init__(self, **kw):
                self.messages = SimpleNamespace(create=lambda **kw: pytest.fail('must not send'))
        monkeypatch.setitem(sys.modules, 'anthropic', SimpleNamespace(Anthropic=Client))
    with pytest.raises(ContextBudgetExceeded):
        if mode == 'no_retry':
            llm_router.create_completion_no_retry(route=route, timeout=1, arguments=common)
        elif mode == 'stream':
            llm_router.create_message_streaming(**common)
        else:
            llm_router.create_message_with_fallback(**common)
