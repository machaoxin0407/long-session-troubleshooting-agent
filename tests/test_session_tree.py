from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from session_context import _size
from request_cancellation import CURRENT_CANCELLATION
from session_memory import SessionMemoryStore
from session_tree import MemoryTree, TreeSettings


def records(count=16):
    return [{'id': n + 1, 'created_at': 100.,
             'question': ('ZX900 充电异常，不能开机。' if n % 2 else 'VR300 显示屏闪烁，已经重启。') + f'编号{n}。',
             'answer': '建议检查连接，尚未核验。'} for n in range(count)]


def model(payload, **_kw):
    groups = json.loads(payload)['groups']
    return {'nodes': [{'id': group['id'], 'text': '用户报告设备异常，客服建议尚未核验。',
                       'sources': [{'leaf_id': group['evidence'][0]['leaf_id'],
                                    'quote': group['evidence'][0]['quote']}]}
                      for group in groups]}


def store_at(path, **kw):
    return SessionMemoryStore(path, history_limit=2, context_tokens=2400, recent_tokens=600,
                              summary_tokens=650, recall_tokens=200, summary_input_tokens=2000,
                              tree_enabled=True, **kw)


def fill(store, count=8, session='a'):
    for record in records(count):
        store.append(session, record['question'] + '补充描述。' * 20, record['answer'])


def tree_from_store(store, session='a'):
    with sqlite3.connect(store.path) as db:
        payload = db.execute('SELECT payload FROM chat_memory_trees WHERE session_id=?', (session,)).fetchone()[0]
    return MemoryTree.load(payload)


def test_recursive_layers_partition_sources_and_feed_child_summaries():
    calls = []
    def summarize(payload, **kw):
        calls.append(json.loads(payload))
        return model(payload, **kw)
    tree = MemoryTree.build(records(), settings=TreeSettings(model_calls=8), summarizer=summarize)
    assert len(calls) >= 3
    assert max(node.level for node in tree.nodes.values()) >= 3
    assert len(tree.roots) == 1
    assert any(child['text'] == '用户报告设备异常，客服建议尚未核验。'
               for group in calls[1]['groups'] for child in group['children'])
    children = [child for node in tree.nodes.values() for child in node.children]
    assert len(children) == len(set(children)) == len(tree.nodes) - 1
    for node in tree.nodes.values():
        assert all(tree.nodes[child].level == node.level - 1 for child in node.children)
        assert all(source['quote'] in tree.nodes[source['leaf_id']].text for source in node.sources)
    restored = MemoryTree.load(tree.dump())
    assert restored.retrieve('ZX900') == tree.retrieve('ZX900')


def test_text_chunks_cover_every_character_even_at_leaf_limit():
    raw = records(2)
    raw[0]['question'] = '头🙂中间内容末尾。' * 1500
    tree = MemoryTree.build(raw, settings=TreeSettings(chunk_tokens=128, max_leaves=6))
    leaves = [node for node in tree.nodes.values() if node.level == 0]
    assert len(leaves) <= 6
    for record in raw:
        for role, key in (('user', 'question'), ('assistant', 'answer')):
            blocks = sorted((leaf for leaf in leaves if leaf.turn_id == record['id'] and leaf.role == role),
                            key=lambda leaf: leaf.start)
            assert ''.join(block.text for block in blocks) == record[key]
            assert all(record[key][block.start:block.end] == block.text for block in blocks)
    with pytest.raises(ValueError, match='all original messages'):
        MemoryTree.build(records(4), settings=TreeSettings(max_leaves=2))


def test_critical_middle_quote_survives_root_evidence_selection():
    from benchmark_session_memory import make_cases
    case = next(c for c in make_cases(40) if c['id'] == 'middle_of_long_turn')
    tree = MemoryTree.build(case['records'])
    root = tree.nodes[tree.roots[0]]
    assert any('ERR-742' in source['quote'] for source in root.sources)
    for source in root.sources:
        assert source['quote'] in tree.nodes[source['leaf_id']].text


@pytest.mark.parametrize('text', ['设备型号是ZX901。', '错误码ERR-742。', '用户未执行断电。', '用户恢复出厂设置。'])
def test_model_rejects_critical_fact_bound_to_unrelated_true_quote(text):
    def summarize(payload, **kw):
        output = model(payload, **kw)
        output['nodes'][0]['text'] = text
        return output
    raw = [{'id': n + 1, 'question': '确认预约信息，暂时没有补充。', 'answer': '收到。'} for n in range(4)]
    tree = MemoryTree.build(raw, summarizer=summarize)
    assert all(node.method != 'model_summary' for node in tree.nodes.values())
    assert any(row.get('reason') == 'tree critical fact absent from selected quotes' for row in tree.diagnostics)


def test_model_rejects_positive_execution_from_negative_source():
    def summarize(payload, **kw):
        output = model(payload, **kw)
        output['nodes'][0]['text'] = '用户已经执行断电。'
        return output
    raw = [{'id': n + 1, 'question': '我没有执行断电。', 'answer': '收到。'} for n in range(4)]
    tree = MemoryTree.build(raw, summarizer=summarize)
    assert all(node.method != 'model_summary' for node in tree.nodes.values())


def test_degenerate_vectors_terminate_and_layer_limit_returns_forest():
    raw = [{'id': n, 'question': '!', 'answer': '!'} for n in range(12)]
    tree = MemoryTree.build(raw, settings=TreeSettings(branch_size=2))
    assert len(tree.roots) == 1
    forest = MemoryTree.build(raw, settings=TreeSettings(branch_size=2, max_levels=1))
    assert len(forest.roots) == 12
    assert max(node.level for node in forest.nodes.values()) == 1


def test_collapsed_searches_all_layers_traversal_follows_selected_branches():
    # Injected vectors make the distinction observable without a network/model.
    def embed(texts, **kw):
        return np.array([[1., 0.] if 'ZX900' in text else [0., 1.] for text in texts])
    tree = MemoryTree.build(records(), embedder=embed, settings=TreeSettings(branch_size=2))
    collapsed = tree.retrieve('ZX900', mode='collapsed', top_k=3)
    assert all('ZX900' in node.text and score > .9 for node, score in collapsed)
    traversal = tree.retrieve('ZX900', mode='traversal', top_k=1)
    assert len(traversal) == max(node.level for node in tree.nodes.values()) + 1
    chosen = {node.id for node, _ in traversal}
    assert any(root in chosen for root in tree.roots)
    assert all(node.id in tree.roots or any(node.id in tree.nodes[parent].children for parent in chosen)
               for node, _ in traversal)
    with pytest.raises(ValueError):
        tree.retrieve('ZX900', mode='wrong')


@pytest.mark.parametrize('bad', ['timeout', 'quote', 'id', 'duplicate', 'oversize', 'empty'])
def test_invalid_model_summaries_fall_back_without_private_errors(bad):
    def summarize(payload, **kw):
        if bad == 'timeout':
            raise TimeoutError('secret endpoint')
        if bad == 'empty':
            return {'nodes': []}
        result = model(payload, **kw)
        item = result['nodes'][0]
        if bad == 'quote':
            item['sources'][0]['quote'] = '虚构引用'
        elif bad == 'id':
            item['id'] = 'foreign-node'
        elif bad == 'duplicate':
            result['nodes'].append(item)
        else:
            item['text'] = '长文本' * 1000
        return result
    tree = MemoryTree.build(records(4), summarizer=summarize)
    assert all(node.method == 'rule_fallback' for node in tree.nodes.values() if node.level)
    assert 'secret' not in str(tree.nodes)


def test_model_input_call_limit_and_expired_deadline():
    calls = []
    def summarize(payload, **kw):
        calls.append((payload, kw))
        return model(payload, **kw)
    settings = TreeSettings(model_calls=1, input_tokens=2000)
    MemoryTree.build(records(), summarizer=summarize, settings=settings)
    assert len(calls) == 1
    assert len(calls[0][0].encode()) <= 2000
    MemoryTree.build(records(), summarizer=summarize, deadline_ts=time.time() - 1)
    assert len(calls) == 1


def test_failed_semantic_embedding_rebuilds_consistent_lexical_tree():
    calls = []
    def embed(texts, **kw):
        calls.append(texts)
        if len(calls) > 1:
            raise RuntimeError('private endpoint')
        return np.ones((len(texts), 2))
    tree = MemoryTree.build(records(4), embedder=embed)
    assert tree.vector_method == 'lexical_tfidf'
    assert tree.vectors.shape[1] == 512
    assert tree.retrieve('ZX900')


def test_cache_persists_modes_share_tree_and_session_isolation(tmp_path):
    calls = []
    def summarize(payload, **kw):
        calls.append(payload)
        return model(payload, **kw)
    store = store_at(tmp_path / 'm.sqlite3', tree_summarizer=summarize)
    fill(store)
    first = store.get('a', question='ZX900', retrieval_mode='collapsed')
    assert first[0]['compression'] == 'tree_summary'
    assert first[0]['tree']['levels'] > 1
    assert _size(first) <= 2400 - 200 - 768
    assert store_at(store.path, tree_summarizer=summarize).get('a', question='ZX900') == first
    count = len(calls)
    other = store.get('a', question='VR300', retrieval_mode='traversal')
    assert len(calls) == count
    assert other[0]['tree']['retrieval'] == 'traversal'
    assert store.get('b', question='ZX900') == []
    assert store.recall('b', 'ZX900') == []


def test_clear_during_model_generation_does_not_lock_or_resurrect(tmp_path):
    path = tmp_path / 'm.sqlite3'
    def summarize(payload, **kw):
        SessionMemoryStore(path).clear('a')
        return model(payload, **kw)
    store = store_at(path, tree_summarizer=summarize)
    fill(store)
    assert store.get('a', question='ZX900') == []
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT COUNT(*) FROM chat_memory_trees').fetchone()[0] == 0


def test_append_during_generation_rejects_stale_tree(tmp_path):
    path = tmp_path / 'm.sqlite3'
    calls = []
    def summarize(payload, **kw):
        if not calls:
            SessionMemoryStore(path).append('a', '新追加事实 ZX901。', '旧回答')
        calls.append(payload)
        return model(payload, **kw)
    store = store_at(path, tree_summarizer=summarize)
    fill(store)
    history = store.get('a', question='ZX900')
    assert history[-2]['content'] == '新追加事实 ZX901。'
    assert tree_from_store(store).nodes


def test_expiry_clear_and_pruning_remove_tree_sources(tmp_path):
    clock = [100.]
    store = store_at(tmp_path / 'm.sqlite3', clock=lambda: clock[0], ttl_s=60, archive_turns=4)
    fill(store)
    store.get('a', question='ZX900')
    tree = tree_from_store(store)
    old_ids = {node.turn_id for node in tree.nodes.values() if node.level == 0}
    fill(store, count=4)
    store.get('a', question='ZX900')
    new_ids = {node.turn_id for node in tree_from_store(store).nodes.values() if node.level == 0}
    assert old_ids.isdisjoint(new_ids)
    clock[0] = 159
    assert store.get('a')
    clock[0] = 160
    assert store.get('a') == []
    with sqlite3.connect(store.path) as db:
        assert db.execute('SELECT COUNT(*) FROM chat_memory_trees').fetchone()[0] == 0


def test_short_history_and_raw_recall_of_offloaded_recent_turn(tmp_path):
    store = store_at(tmp_path / 'm.sqlite3')
    store.append('a', '型号 ZX900。', '好的。')
    assert all(item['role'] != 'memory' for item in store.get('a'))
    question = '无关。' * 400 + '错误码 UNIQUE123 必须断电。' + '末尾。' * 400
    store.append('a', question, '旧回答')
    history = store.get('a', question='UNIQUE123')
    assert any('UNIQUE123' in node.text for node in tree_from_store(store).nodes.values() if node.level == 0)
    assert 'UNIQUE123' in str(store.recall('a', 'UNIQUE123', visible_history=history))
    assert _size(history) <= 2400 - 200 - 768


def test_worker_query_mode_trace_and_request_validation(tmp_path, monkeypatch):
    import agent
    import api_server as api
    store = store_at(tmp_path / 'm.sqlite3')
    fill(store)
    monkeypatch.setattr(api, '_SESSION_MEMORY', store)
    monkeypatch.setattr(api, '_engine', SimpleNamespace())
    questions = []
    monkeypatch.setattr(api, '_classify_question', lambda q, **kw: (questions.append(q) or 'service', {}))
    monkeypatch.setattr(api, '_retrieve_video_evidence', lambda *_: ([], api.RetrievalStatusItem(
        requested_mode='bm25', effective_mode='bm25')))
    monkeypatch.setattr(agent, 'run_agent', lambda *_a, **kw: SimpleNamespace(answer='回答', pics=[], trace={}))
    result = api._run_agent_sync('ZX900 怎么处理？', 'a', [], time.time() + 20, 'traversal')
    assert result[3]['session_memory_tree']['retrieval'] == 'traversal'
    assert questions[0].endswith('用户当前问题: ZX900 怎么处理？')
    assert _size(store.get('a', question='ZX900')) < 2400
    assert api._SESSION_CONTEXT_QUERY.get() == ('', None)
    assert api.ChatRequest(question='测试', memory_retrieval='collapsed').memory_retrieval == 'collapsed'
    with pytest.raises(ValidationError):
        api.ChatRequest(question='测试', memory_retrieval='wrong')


def test_settings_reject_invalid_limits():
    with pytest.raises(ValueError):
        replace(TreeSettings(), branch_size=1)
    with pytest.raises(ValueError):
        replace(TreeSettings(), build_timeout_s=float('nan'))


def test_empty_tree_roundtrip_and_no_matching_terms_use_root_overview():
    tree = MemoryTree.load(MemoryTree.build([]).dump())
    assert tree.retrieve('ZX900') == []
    tree = MemoryTree.build(records())
    assert [node.id for node, _ in tree.retrieve('')] == tree.roots
    with pytest.raises(ValueError, match='duplicate'):
        MemoryTree.build(records(2) + records(2))


def test_inconsistent_embedding_dimensions_fall_back_and_semantic_load_needs_embedder():
    calls = []
    def embed(texts, **kw):
        calls.append(True)
        return np.ones((len(texts), len(calls) + 1))
    tree = MemoryTree.build(records(4), embedder=embed)
    assert tree.vector_method == 'lexical_tfidf'
    tree = MemoryTree.build(records(2), embedder=lambda texts, **kw: np.ones((len(texts), 2)))
    with pytest.raises(ValueError, match='original embedder'):
        MemoryTree.load(tree.dump()).retrieve('ZX900')


def test_cancellation_during_summary_propagates_without_persisting_tree(tmp_path):
    event = threading.Event()
    def summarize(payload, **kw):
        event.set()
        raise RuntimeError('private failure')
    store = store_at(tmp_path / 'm.sqlite3', tree_summarizer=summarize)
    fill(store)
    token = CURRENT_CANCELLATION.set(event)
    try:
        with pytest.raises(TimeoutError):
            store.get('a', question='ZX900')
    finally:
        CURRENT_CANCELLATION.reset(token)
    with sqlite3.connect(store.path) as db:
        assert db.execute('SELECT COUNT(*) FROM chat_memory_trees').fetchone()[0] == 0
        assert db.execute('SELECT COUNT(*) FROM chat_memory_turns').fetchone()[0] == 8


@pytest.mark.parametrize('endpoint', ['/chat', '/v2/chat'])
def test_http_mode_passed_to_worker_and_invalid_mode_rejected(tmp_path, monkeypatch, endpoint):
    from fastapi.testclient import TestClient
    import api_server as api
    calls = []
    def worker(*args):
        calls.append(args)
        return ('回答', [], 'service', {}, [], api.RetrievalStatusItem(
            requested_mode='bm25', effective_mode='bm25'), [], [])
    monkeypatch.setattr(api, '_SESSION_MEMORY', store_at(tmp_path / 'm.sqlite3'))
    monkeypatch.setattr(api, 'EXPECTED_TOKEN', 'offline-test-token')
    monkeypatch.setattr(api, '_run_agent_sync', worker)
    monkeypatch.setattr(api, '_write_api_success_trace', lambda **kw: None)
    with TestClient(api.app) as client:
        headers = {'Authorization': 'Bearer offline-test-token'}
        response = client.post(endpoint, headers=headers,
                               json={'question': 'ZX900', 'session_id': 'a', 'memory_retrieval': 'traversal'})
        assert response.status_code == 200
        assert calls[0][0] == 'ZX900' and calls[0][-1] == 'traversal'
        response = client.post(endpoint, headers=headers,
                               json={'question': 'ZX900', 'session_id': 'a', 'memory_retrieval': 'invalid'})
        assert response.status_code == 422 and len(calls) == 1
