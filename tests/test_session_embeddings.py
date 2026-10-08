import asyncio
import json
import threading
import time
from types import SimpleNamespace

import httpx
import pytest

from request_cancellation import CURRENT_CANCELLATION
from session_embeddings import SessionEmbedder
from session_summarizer import ModelSessionSummarizer


def adapter(**kw):
    return SessionEmbedder(**{'base_url': 'http://127.0.0.1:1/v1', 'api_key': 'test', 'model': 'test', **kw})


def transport(monkeypatch, respond):
    client = httpx.AsyncClient
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kw: client(**kw, transport=httpx.MockTransport(respond)))


def test_batches_order_vectors_and_bound_long_unicode_text(monkeypatch):
    requests = []
    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        assert str(request.url).endswith('/v1/embeddings')
        assert request.headers['Authorization'] == 'Bearer test'
        assert all(len(text.encode()) <= 128 for text in body['input'])
        return httpx.Response(200, json={'data': [{'index': index, 'embedding': [float(index), 1.]}
                                                for index in reversed(range(len(body['input'])))]})
    transport(monkeypatch, respond)
    result = adapter(batch_size=2, text_bytes=128)(['长文本🙂' * 300, 'b', 'c'])
    assert result == [[0., 1.], [1., 1.], [0., 1.]]
    assert len(requests) == 2


@pytest.mark.parametrize('model', ['qwen3.7-text-embedding', 'qwen3.7-text-embedding-flash'])
def test_qwen_batch_limit_preserves_all_inputs_and_order(monkeypatch, model):
    batches = []

    def respond(request):
        body = json.loads(request.content)
        assert body['model'] == model
        assert len(body['input']) <= 20
        batches.append(body['input'])
        return httpx.Response(200, json={'data': [
            {'index': i, 'embedding': [float(text), 1.]}
            for i, text in reversed(list(enumerate(body['input'])))
        ]})

    transport(monkeypatch, respond)
    inputs = [str(i) for i in range(41)]
    assert adapter(model=model)(inputs) == [[float(i), 1.] for i in range(41)]
    assert [len(batch) for batch in batches] == [20, 20, 1]
    assert [text for batch in batches for text in batch] == inputs
    assert adapter(model=model, batch_size=8).batch_size == 8


@pytest.mark.parametrize('data', [
    [{'index': 0, 'embedding': [1.]}, {'index': 0, 'embedding': [2.]}],
    [{'index': True, 'embedding': [1.]}, {'index': 1, 'embedding': [2.]}],
    [{'index': 0, 'embedding': [1.]}],
    [{'index': 0, 'embedding': [1.]}, {'index': 1, 'embedding': [1., 2.]}],
])
def test_invalid_embedding_data_rejected(monkeypatch, data):
    transport(monkeypatch, lambda request: httpx.Response(200, json={'data': data}))
    with pytest.raises(ValueError):
        adapter()(['a', 'b'])


def test_timeout_cancels_transport_and_releases_slot(monkeypatch):
    ended = []
    async def respond(request):
        try:
            await asyncio.sleep(1)
        finally:
            ended.append(True)
    transport(monkeypatch, respond)
    with pytest.raises(TimeoutError):
        adapter(timeout_s=.02)(['a'])
    assert ended == [True]


def test_cancellation_propagates_and_closes_embedding_transport(monkeypatch):
    event, ended = threading.Event(), []
    async def respond(request):
        event.set()
        try:
            await asyncio.sleep(5)
        finally:
            ended.append(True)
    transport(monkeypatch, respond)
    token = CURRENT_CANCELLATION.set(event)
    try:
        with pytest.raises(TimeoutError, match='cancelled'):
            adapter()(['a'])
    finally:
        CURRENT_CANCELLATION.reset(token)
    assert ended == [True]


def test_budget_hook_stops_before_any_network_call(monkeypatch):
    calls = []
    transport(monkeypatch, lambda request: calls.append(request))
    def denied():
        raise TimeoutError('request limit')
    with pytest.raises(TimeoutError, match='request limit'):
        adapter(on_request=denied)(['a'])
    assert not calls


def test_summary_reserve_zero_and_usage_callbacks(monkeypatch):
    import llm_router
    route = SimpleNamespace(base_url='http://127.0.0.1:1/v1', api_key='test', model='test', protocol='openai', name='test')
    monkeypatch.setattr(llm_router, 'get_routes', lambda: [route])
    monkeypatch.setattr(llm_router, '_get_route_semaphore', lambda name: threading.BoundedSemaphore(1))
    requests, usage = [], []
    transport(monkeypatch, lambda request: httpx.Response(200, json={
        'choices': [{'finish_reason': 'stop', 'message': {'content': '{"nodes":[]}'}}],
        'usage': {'prompt_tokens': 12, 'completion_tokens': 4, 'total_tokens': 16, 'private': 'secret'}}))
    model = ModelSessionSummarizer(reserve_s=0, on_request=lambda: requests.append(True), on_usage=usage.append)
    assert model('[]', deadline_ts=time.time() + 2) == {'nodes': []}
    assert requests == [True] and usage == [{'prompt_tokens': 12, 'completion_tokens': 4, 'total_tokens': 16}]
    with pytest.raises(TimeoutError, match='no summary budget'):
        ModelSessionSummarizer()('[]', deadline_ts=time.time() + 2)


def test_embedding_signature_does_not_include_private_address_or_key():
    assert '127.0.0.1' not in adapter().signature
    assert adapter(api_key='different').signature == adapter().signature


def test_dedicated_settings_take_precedence_without_changing_manual_settings(monkeypatch):
    monkeypatch.setenv('EMBEDDING_BASE_URL', 'http://127.0.0.1:1/manual')
    monkeypatch.setenv('EMBEDDING_API_KEY', 'manual-test')
    monkeypatch.setenv('EMBEDDING_MODEL', 'manual-model')
    monkeypatch.setenv('CHAT_SESSION_TREE_EMBEDDING_BASE_URL', 'http://127.0.0.1:2/session')
    monkeypatch.setenv('CHAT_SESSION_TREE_EMBEDDING_API_KEY', 'session-test')
    monkeypatch.setenv('CHAT_SESSION_TREE_EMBEDDING_MODEL', 'qwen3.7-text-embedding')
    embedding = SessionEmbedder()
    assert embedding.model == 'qwen3.7-text-embedding' and embedding.batch_size == 20
    assert embedding.base_url.endswith('/session') and embedding.api_key == 'session-test'
    import os
    assert os.getenv('EMBEDDING_MODEL') == 'manual-model'
    monkeypatch.delenv('CHAT_SESSION_TREE_EMBEDDING_MODEL')
    assert SessionEmbedder().model == 'manual-model'


def test_duplicate_inputs_are_embedded_once_and_expanded_in_original_order(monkeypatch):
    batches = []
    def respond(request):
        body = json.loads(request.content)
        batches.append(body['input'])
        return httpx.Response(200, json={'data': [
            {'index': i, 'embedding': [float(i), 1.]} for i in range(len(body['input']))]})
    transport(monkeypatch, respond)
    assert adapter()(['a', 'a', 'b', 'a']) == [[0., 1.], [0., 1.], [1., 1.], [0., 1.]]
    assert batches == [['a', 'b']]


def test_connection_scope_reuses_one_client_and_closes_at_request_boundary(monkeypatch):
    client, created = httpx.AsyncClient, []
    def respond(request):
        return httpx.Response(200, json={'data': [{'index': 0, 'embedding': [1., 2.]}]})
    def factory(**kw):
        obj = client(**kw, transport=httpx.MockTransport(respond))
        created.append(obj)
        return obj
    monkeypatch.setattr(httpx, 'AsyncClient', factory)
    embedding = adapter()
    with embedding.connection_scope():
        assert embedding(['a']) == embedding(['b'])
        assert len(created) == 1 and not created[0].is_closed
    assert created[0].is_closed and embedding._local.loop is None
    with embedding.connection_scope():
        embedding(['a'])
    assert len(created) == 2 and created[1].is_closed


def test_json_summary_includes_json_instruction_and_disables_qwen_thinking(monkeypatch):
    import llm_router
    route = SimpleNamespace(base_url='http://127.0.0.1:1/v1', api_key='test', model='qwen-test',
                            protocol='openai', name='test')
    monkeypatch.setattr(llm_router, 'get_routes', lambda: [route])
    monkeypatch.setattr(llm_router, '_get_route_semaphore', lambda _: threading.BoundedSemaphore(1))
    def respond(request):
        body = json.loads(request.content)
        assert 'json' in body['messages'][0]['content'].lower()
        assert body['enable_thinking'] is False
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
                                        'message': {'content': '{"ok":true}'}}]})
    transport(monkeypatch, respond)
    assert ModelSessionSummarizer(prompt='中文摘要', reserve_s=0)('{}') == {'ok': True}
