from __future__ import annotations

import asyncio
import json
import sqlite3
import subprocess
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from session_memory import SessionMemoryStore, excerpt, observe


def test_reopen_and_separate_sessions(tmp_path):
    path = tmp_path / 'memory.sqlite3'
    store = SessionMemoryStore(path)
    store.append('a', '型号 DCB107，没有黄色灯。', '需要查看手册。')
    store.append('b', '型号 DCB112。', '不同会话。')
    restarted = SessionMemoryStore(path)
    assert restarted.get('a')[0]['content'] == '型号 DCB107，没有黄色灯。'
    assert restarted.get('unknown') == []
    assert 'DCB107' not in str(restarted.get('b'))


def test_actual_process_restart(tmp_path):
    path = tmp_path / 'memory.sqlite3'
    script = "from session_memory import SessionMemoryStore; import sys; SessionMemoryStore(sys.argv[1]).append('a', '重启后追问', '先前回答')"
    subprocess.run([sys.executable, '-c', script, str(path)], check=True,
                   cwd=Path(__file__).resolve().parents[1], capture_output=True)
    assert SessionMemoryStore(path).get('a')[0]['content'] == '重启后追问'


def test_compaction_preserves_middle_constraints_and_canonical_original(tmp_path):
    store = SessionMemoryStore(tmp_path / 'memory.sqlite3', history_limit=2,
                               recent_chars=400, excerpt_chars=240, summary_chars=1000)
    question = '普通描述。' * 80 + '更正：型号 DCB107，不是 DCB101，不能继续充电。' + '补充描述。' * 80
    store.append('a', question, '这是未核验的历史建议。')
    assert any('[中间内容已省略]' in item['content'] for item in store.get('a') if item['role'] == 'user')
    store.append('a', '后续问题', '后续回答')
    history = store.get('a')
    assert history[0]['role'] == 'memory'
    assert '型号 DCB107，不是 DCB101，不能继续充电。' in history[0]['content']
    assert '未重新核验' in history[0]['content']
    assert 'turn=' in history[0]['content']
    with sqlite3.connect(store.path) as db:
        payload = db.execute('SELECT payload FROM chat_memory_turns ORDER BY id LIMIT 1').fetchone()[0]
        assert json.loads(zlib.decompress(payload))['question'] == question
    assert 'DCB107' in str(store.recall('a', 'DCB107 充电'))
    assert store.recall('b', 'DCB107 充电') == []


def test_long_conversation_stays_bounded_and_can_recall_early_detail(tmp_path):
    store = SessionMemoryStore(tmp_path / 'memory.sqlite3', summary_chars=800,
                               recent_chars=600, excerpt_chars=150, recall_chars=800)
    store.append('a', 'Printer ZX900，错误码 E17，不能自动进纸。', '请根据本轮证据核验。')
    for n in range(20):
        store.append('a', f'第{n}轮其他讨论。' * 20, '普通回答。' * 30)
    history = store.get('a')
    recent = [item for item in history if item['role'] != 'memory']
    assert len(recent) == 6
    assert sum(len(item['content']) for item in recent) <= 600
    assert len(history[0]['content']) <= 800
    assert 'E17' in str(store.recall('a', 'ZX900 错误码 E17'))
    assert len(store.recall('a', 'ZX900 错误码 E17')[0]['content']) <= 800


def test_clear_and_expiry_delete_archive_without_resurrecting_it(tmp_path):
    now = [100.0]
    store = SessionMemoryStore(tmp_path / 'memory.sqlite3', ttl_s=60, clock=lambda: now[0])
    store.append('a', '错误码 E17', 'a')
    store.append('b', '保留另一个会话', 'b')
    store.clear('a')
    assert store.get('a') == []
    assert store.get('b')
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT COUNT(*) FROM chat_memory_turns WHERE session_id='a'").fetchone()[0] == 0
    now[0] += 60
    assert store.get('b') == []
    store.append('b', '新一轮', '新回答')
    assert '保留另一个会话' not in str(store.get('b'))


def test_archive_count_and_size_limits(tmp_path):
    store = SessionMemoryStore(tmp_path / 'memory.sqlite3', history_limit=2,
                               archive_turns=3, archive_chars=300)
    for n in range(6):
        store.append('a', f'问题{n}' * 10, f'回答{n}' * 10)
    with sqlite3.connect(store.path) as db:
        count, chars = db.execute('SELECT COUNT(*), SUM(chars) FROM chat_memory_turns').fetchone()
        assert count <= 3 and chars <= 300
    store.append('a', '很长的输入。' * 500, '很长的回答。' * 500)
    with sqlite3.connect(store.path) as db:
        assert db.execute('SELECT SUM(chars) FROM chat_memory_turns').fetchone()[0] <= 300


def test_long_recent_message_can_recall_clipped_middle_before_any_old_turn(tmp_path):
    store = SessionMemoryStore(tmp_path / 'memory.sqlite3', recent_chars=600)
    raw = '普通说明。' * 100 + '设备序列号 ZX900，错误码 E17。' + '更多说明。' * 100
    store.append('a', raw, '待确认')
    assert store.get('a')[0]['role'] == 'memory'
    assert 'ZX900' in str(store.recall('a', 'ZX900 E17'))


def test_concurrent_processes_merge_turns_without_lost_writes(tmp_path):
    path = tmp_path / 'memory.sqlite3'
    script = "from session_memory import SessionMemoryStore; import sys; SessionMemoryStore(sys.argv[1]).append('same', sys.argv[2], 'reply')"
    def append(n):
        subprocess.run([sys.executable, '-c', script, str(path), str(n)], check=True,
                       cwd=Path(__file__).resolve().parents[1], capture_output=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(append, range(8)))
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT COUNT(*) FROM chat_memory_turns').fetchone()[0] == 8
    assert len(SessionMemoryStore(path).get('same')) == 7


def test_sql_identifiers_are_data_and_newer_corrections_remain_ordered(tmp_path):
    store = SessionMemoryStore(tmp_path / 'memory.sqlite3', history_limit=2)
    key = "a'; DROP TABLE chat_memory;--"
    store.append(key, '型号 DCB101', '待确认')
    store.append(key, '更正，不是 DCB101，而是 DCB107。', '按更正核验')
    history = store.get(key)
    assert history[-2]['content'] == '更正，不是 DCB101，而是 DCB107。'
    assert store.get('a') == []


@pytest.mark.parametrize('text', ['长中文文本。' * 200, 'A long sentence. ' * 200, 'x' * 4000])
def test_compression_marks_omissions_and_obeys_budget(text):
    for compress in (excerpt, observe):
        assert len(compress(text, 150)) <= 150
        assert '[中间内容已省略]' in compress(text, 150)
    assert observe('型号 DCB107，不是 DCB101。', 100) == '型号 DCB107，不是 DCB101。'


@pytest.fixture
def api_memory(tmp_path, monkeypatch):
    import api_server as api
    monkeypatch.setattr(api, '_SESSION_MEMORY', SessionMemoryStore(tmp_path / 'api.sqlite3'))
    return api


def test_worker_receives_summary_recall_and_current_question(api_memory, monkeypatch):
    api = api_memory
    for n in range(5):
        api._append_session_turn('a', f'DCB107 充电故障 {n}', '未核验的旧建议。')
    captured = []
    monkeypatch.setattr(api, '_engine', SimpleNamespace(ensure_index=lambda: None))
    monkeypatch.setattr(api, '_classify_question', lambda q, **kw: (captured.append(q) or 'service', {}))
    monkeypatch.setattr(api, '_retrieve_video_evidence', lambda *_: ([], api.RetrievalStatusItem(requested_mode='bm25', effective_mode='bm25')))
    import agent
    monkeypatch.setattr(agent, 'run_agent', lambda q, *_a, **_kw: SimpleNamespace(answer='新回答', pics=[], trace={}))
    result = api._run_agent_sync('DCB107 现在怎么处理？', 'a', [])
    assert '较早对话压缩摘录' in captured[0]
    assert '当前会话相关原文回查' in captured[0]
    assert captured[0].endswith('用户当前问题: DCB107 现在怎么处理？')
    assert result[3]['session_history_turns'] == 3
    assert result[3]['session_memory_compressed'] and result[3]['session_memory_recalled']


def test_api_persists_only_successful_answers_and_clear_is_isolated(api_memory, monkeypatch):
    api = api_memory
    status = api.RetrievalStatusItem(requested_mode='bm25', effective_mode='bm25')
    result = ('回答', [], 'service', {}, [], status, [], [])
    monkeypatch.setattr(api, '_run_agent_sync', Mock(return_value=result))
    monkeypatch.setattr(api, '_format_answer', lambda answer, *_: answer)
    monkeypatch.setattr(api, '_write_api_success_trace', Mock())
    request = SimpleNamespace(headers={}, is_disconnected=Mock())
    async def connected():
        return False
    request.is_disconnected = connected
    asyncio.run(api.chat(api.ChatRequest(question='第一问', session_id='a'), request))
    assert api._get_session_history('a')[0]['content'] == '第一问'
    from response_integrity import IncompleteGenerationError
    monkeypatch.setattr(api, '_run_agent_sync', Mock(side_effect=IncompleteGenerationError('失败')))
    with pytest.raises(api.HTTPException) as caught:
        asyncio.run(api.chat(api.ChatRequest(question='失败问题', session_id='a'), request))
    assert caught.value.status_code == 502
    assert len(api._get_session_history('a')) == 2
    api._append_session_turn('b', '另一个会话', '回答')
    assert asyncio.run(api.clear_chat_session('a')).status_code == 204
    assert api._get_session_history('a') == []
    assert api._get_session_history('b')


@pytest.mark.parametrize('operation', ['get', 'append', 'recall', 'clear'])
def test_storage_failure_is_explicit_503(api_memory, monkeypatch, operation):
    api = api_memory
    monkeypatch.setattr(api._SESSION_MEMORY, operation, Mock(side_effect=sqlite3.OperationalError('private path')))
    functions = {'get': lambda: api._get_session_history('a'),
                 'append': lambda: api._append_session_turn('a', 'q', 'a'),
                 'recall': lambda: api._recall_session_history('a', 'q'),
                 'clear': lambda: asyncio.run(api.clear_chat_session('a'))}
    with pytest.raises(api.HTTPException) as caught:
        functions[operation]()
    assert caught.value.status_code == 503
    assert 'private path' not in caught.value.detail


def test_http_clear_requires_auth_and_preserves_other_sessions(api_memory, monkeypatch):
    from fastapi.testclient import TestClient
    api = api_memory
    monkeypatch.setattr(api, 'EXPECTED_TOKEN', 'offline-test-token')
    api._append_session_turn('a', 'q', 'a')
    api._append_session_turn('b', 'q', 'b')
    client = TestClient(api.app)
    try:
        assert client.delete('/v2/chat/sessions/a').status_code in {401, 403}
        assert api._get_session_history('a')
        response = client.delete('/v2/chat/sessions/a', headers={'Authorization': 'Bearer offline-test-token'})
        assert response.status_code == 204 and not response.content
        assert api._get_session_history('a') == []
        assert api._get_session_history('b')
    finally:
        client.close()
