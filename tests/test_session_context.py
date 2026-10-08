from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from request_cancellation import CURRENT_CANCELLATION
from session_context import FIELDS, estimate_tokens, fit_text, validate_summary
from session_memory import SessionMemoryStore
from session_summarizer import ModelSessionSummarizer


def structured(sources):
    value = {field: [] for field in FIELDS}
    turn = sources[-1]
    value['constraints'] = [{
        'text': '用户说不能继续充电。',
        'sources': [{'turn_id': turn['id'], 'role': 'user', 'quote': '不能继续充电。'}],
    }]
    return value


def store_at(path, summarizer=None, **kw):
    return SessionMemoryStore(path, history_limit=2, context_tokens=2400,
                              recent_tokens=600, summary_tokens=650, recall_tokens=200,
                              summary_input_tokens=2000, summarizer=summarizer, **kw)


def fill(store, session='a', count=5):
    for n in range(count):
        store.append(session, f'型号 ZX900 第{n}轮。不能继续充电。' + '补充描述。' * 45,
                     '旧建议尚未核验。' + '普通描述。' * 20)


def test_short_context_keeps_original_even_if_legacy_excerpts_are_small(tmp_path):
    callback = Mock()
    store = store_at(tmp_path / 'a.sqlite3', callback, recent_chars=128)
    question = 'X' * 150 + '中间信息。' + 'Y' * 150
    store.append('a', question, 'answer')
    history = store.get('a')
    assert history[0]['content'] == question
    callback.assert_not_called()


def test_model_summary_persists_with_sources_and_cache_reopens(tmp_path):
    calls = []
    def summarize(payload, **kw):
        calls.append(json.loads(payload))
        return structured(calls[-1])
    path = tmp_path / 'a.sqlite3'
    store = store_at(path, summarize)
    fill(store)
    first = store.get('a')
    assert first[0]['compression'] == 'model_summary'
    assert '不能继续充电。' in first[0]['content'] and '[turn=' in first[0]['content']
    assert first[-2]['content'].startswith('型号 ZX900 第4轮。')
    assert len(calls) == 1 and estimate_tokens(json.dumps(calls[0], ensure_ascii=False)) <= 2000
    assert store_at(path, summarize).get('a') == first
    assert len(calls) == 1
    assert store.get('b') == []
    assert store.recall('b', 'ZX900') == []


@pytest.mark.parametrize('failure', ['timeout', 'invalid_json', 'wrong_role', 'fake_quote', 'empty'])
def test_summary_failure_falls_back_and_remains_recoverable(tmp_path, failure):
    def summarize(payload, **kw):
        if failure == 'timeout':
            raise TimeoutError('private endpoint and key')
        if failure == 'invalid_json':
            return 'invalid'
        value = structured(json.loads(payload))
        if failure == 'wrong_role':
            value['constraints'][0]['sources'][0]['role'] = 'assistant'
        elif failure == 'fake_quote':
            value['constraints'][0]['sources'][0]['quote'] = '不存在的原话'
        elif failure == 'empty':
            value = {field: [] for field in FIELDS}
        return value
    store = store_at(tmp_path / 'a.sqlite3', summarize)
    fill(store)
    history = store.get('a')
    assert history[0]['compression'] == 'rule_fallback'
    assert 'private endpoint' not in str(history)
    assert 'ZX900' in str(store.recall('a', 'ZX900', visible_history=history))
    assert estimate_tokens(json.dumps(history, ensure_ascii=False)) <= 2400 - 200 - 768


def test_source_role_and_turn_cannot_be_fabricated():
    sources = [{'id': 1, 'question': '不能继续充电。', 'answer': '建议重启。'}]
    value = structured(sources)
    value['constraints'][0]['sources'][0]['turn_id'] = 99
    with pytest.raises(ValueError):
        validate_summary(value, sources, 650)
    value = {field: [] for field in FIELDS}
    value['attempted_actions'] = [{'text': '用户已经重启', 'sources': [
        {'turn_id': 1, 'role': 'assistant', 'quote': '建议重启。'}]}]
    with pytest.raises(ValueError):
        validate_summary(value, sources, 650)


def test_summary_generation_does_not_lock_db_or_resurrect_cleared_session(tmp_path):
    path = tmp_path / 'a.sqlite3'
    def summarize(payload, **kw):
        SessionMemoryStore(path).clear('a')
        return structured(json.loads(payload))
    store = store_at(path, summarize)
    fill(store)
    assert store.get('a') == []
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT COUNT(*) FROM chat_memory_compactions').fetchone()[0] == 0


def test_concurrent_append_rejects_stale_summary_and_preserves_new_turn(tmp_path):
    path = tmp_path / 'a.sqlite3'
    calls = []
    def summarize(payload, **kw):
        calls.append(payload)
        SessionMemoryStore(path).append('a', '最新更正：型号 ZX901。', '按更正处理')
        return structured(json.loads(payload))
    store = store_at(path, summarize)
    fill(store)
    history = store.get('a')
    assert 'ZX901' in str(history) and history[0]['compression'] == 'rule_fallback'
    assert len(calls) == 1


def test_token_offload_recovers_long_latest_message_and_bounds_unicode(tmp_path):
    store = store_at(tmp_path / 'a.sqlite3')
    raw = '普通说明。' * 300 + '错误码 E17，型号 ZX900。' + '更多说明。' * 300
    store.append('a', raw, '等待证据')
    history = store.get('a')
    assert history[0]['compression'] == 'budget_offload'
    assert history[-2]['offloaded']
    recall = store.recall('a', 'ZX900 E17', visible_history=history)
    assert estimate_tokens(recall[0]['content']) <= 120
    assert 'ZX900' in str(recall)
    for text in ('中文。' * 1000, '🙂' * 1000, 'x' * 1000):
        assert estimate_tokens(fit_text(text, 250)) <= 250


def test_archive_pruning_invalidates_summary_and_clear_cascades(tmp_path):
    store = store_at(tmp_path / 'a.sqlite3', lambda payload, **kw: structured(json.loads(payload)),
                     archive_turns=5)
    fill(store)
    assert store.get('a')[0]['compression'] == 'model_summary'
    # The old first turn must disappear from the current summary too.
    store.append('a', '最新问题', '最新回答')
    assert '[turn=1;' not in str(store.get('a'))
    store.clear('a')
    assert store.get('a') == []
    with sqlite3.connect(store.path) as db:
        assert db.execute('SELECT COUNT(*) FROM chat_memory_compactions').fetchone()[0] == 0


def test_cancellation_is_not_swallowed_or_persisted(tmp_path):
    event = threading.Event()
    def summarize(payload, **kw):
        event.set()
        raise TimeoutError()
    store = store_at(tmp_path / 'a.sqlite3', summarize)
    fill(store)
    token = CURRENT_CANCELLATION.set(event)
    try:
        with pytest.raises(TimeoutError):
            store.get('a')
    finally:
        CURRENT_CANCELLATION.reset(token)
    with sqlite3.connect(store.path) as db:
        assert db.execute('SELECT COUNT(*) FROM chat_memory_compactions').fetchone()[0] == 0


@pytest.mark.parametrize('protocol', ['openai', 'anthropic'])
def test_adapter_uses_configured_route_and_accepts_only_complete_json(monkeypatch, protocol):
    import llm_router
    route = SimpleNamespace(base_url='http://127.0.0.1:1/v1', api_key='test',
                            model='configured-model', name='test', protocol=protocol)
    monkeypatch.setattr(llm_router, 'get_routes', lambda: [route])
    requests = []
    def respond(request):
        requests.append(json.loads(request.content))
        data = {'choices': [{'finish_reason': 'stop', 'message': {'content': '{}'}}]}
        if protocol == 'anthropic':
            data = {'stop_reason': 'end_turn', 'content': [{'type': 'text', 'text': '{}'}]}
        return httpx.Response(200, json=data)
    client_class = httpx.AsyncClient
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kw: client_class(
        **kw, transport=httpx.MockTransport(respond)))
    assert ModelSessionSummarizer()('[]', deadline_ts=time.time() + 20) == {}
    assert requests[0]['model'] == 'configured-model'


def test_adapter_enforces_total_deadline_without_network(monkeypatch):
    async def slow(request):
        await asyncio.sleep(1)
        return httpx.Response(200, json={})
    client_class = httpx.AsyncClient
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kw: client_class(
        **kw, transport=httpx.MockTransport(slow)))
    route = SimpleNamespace(base_url='http://127.0.0.1:1/v1', api_key='test',
                            model='test', protocol='openai')
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        asyncio.run(ModelSessionSummarizer()._request(route, '[]', time.time() + 0.05))
    assert time.monotonic() - started < 0.5


@pytest.mark.parametrize('finish_reason', ['length', 'content_filter'])
def test_adapter_rejects_incomplete_or_filtered_summary(monkeypatch, finish_reason):
    def respond(request):
        return httpx.Response(200, json={'choices': [
            {'finish_reason': finish_reason, 'message': {'content': '{}'}}]})
    client_class = httpx.AsyncClient
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kw: client_class(
        **kw, transport=httpx.MockTransport(respond)))
    route = SimpleNamespace(base_url='http://127.0.0.1:1/v1', api_key='test',
                            model='test', protocol='openai')
    with pytest.raises(ValueError, match='incomplete'):
        asyncio.run(ModelSessionSummarizer()._request(route, '[]', time.time() + 2))


def test_adapter_releases_route_slot_after_failure_and_skips_busy_route(monkeypatch):
    import llm_router
    route = SimpleNamespace(name='test', base_url='http://127.0.0.1:1/v1', api_key='test',
                            model='test', protocol='openai')
    semaphore = threading.BoundedSemaphore(1)
    monkeypatch.setattr(llm_router, 'get_routes', lambda: [route])
    monkeypatch.setattr(llm_router, '_get_route_semaphore', lambda _: semaphore)
    summarizer = ModelSessionSummarizer()
    async def fail(*args):
        raise ValueError('model failed')
    monkeypatch.setattr(summarizer, '_request', fail)
    with pytest.raises(ValueError):
        summarizer('[]')
    assert semaphore.acquire(blocking=False)
    try:
        with pytest.raises(TimeoutError, match='busy'):
            summarizer('[]')
    finally:
        semaphore.release()


def test_adapter_cancellation_closes_inflight_exchange(monkeypatch):
    event = threading.Event()
    ended = []
    async def respond(request):
        event.set()
        try:
            await asyncio.sleep(5)
        finally:
            ended.append(True)
    client_class = httpx.AsyncClient
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kw: client_class(
        **kw, transport=httpx.MockTransport(respond)))
    route = SimpleNamespace(base_url='http://127.0.0.1:1/v1', api_key='test',
                            model='test', protocol='openai')
    token = CURRENT_CANCELLATION.set(event)
    try:
        with pytest.raises(TimeoutError, match='cancelled'):
            asyncio.run(ModelSessionSummarizer()._request(route, '[]', time.time() + 3))
    finally:
        CURRENT_CANCELLATION.reset(token)
    assert ended == [True]


def test_summary_expiry_does_not_extend_ttl_on_read(tmp_path):
    clock = [100.0]
    store = store_at(tmp_path / 'a.sqlite3', lambda payload, **kw: structured(json.loads(payload)),
                     clock=lambda: clock[0], ttl_s=60)
    fill(store)
    assert store.get('a')[0]['compression'] == 'model_summary'
    clock[0] += 59
    assert store.get('a')
    clock[0] += 1
    assert store.get('a') == []
    with sqlite3.connect(store.path) as db:
        assert db.execute('SELECT COUNT(*) FROM chat_memory_compactions').fetchone()[0] == 0


def test_api_worker_receives_model_summary_with_deadline_and_trace(tmp_path, monkeypatch):
    import agent
    import api_server as api
    deadlines = []
    def summarize(payload, **kw):
        deadlines.append(kw['deadline_ts'])
        return structured(json.loads(payload))
    store = store_at(tmp_path / 'a.sqlite3', summarize)
    fill(store)
    monkeypatch.setattr(api, '_SESSION_MEMORY', store)
    monkeypatch.setattr(api, '_engine', SimpleNamespace())
    questions = []
    monkeypatch.setattr(api, '_classify_question', lambda q, **kw: (questions.append(q) or 'service', {}))
    monkeypatch.setattr(api, '_retrieve_video_evidence', lambda *_: ([], api.RetrievalStatusItem(
        requested_mode='bm25', effective_mode='bm25')))
    monkeypatch.setattr(agent, 'run_agent', lambda *_a, **kw: SimpleNamespace(answer='回答', pics=[], trace={}))
    deadline = time.time() + 20
    result = api._run_agent_sync('ZX900 现在怎么处理？', 'a', [], deadline)
    assert deadlines == [deadline]
    assert questions[0].endswith('用户当前问题: ZX900 现在怎么处理？')
    assert result[3]['session_memory_compression_method'] == 'model_summary'
    assert result[3]['session_memory_token_counter'] == 'utf8_byte_estimate'
    assert estimate_tokens(questions[0]) < 2400
