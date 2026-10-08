import json

import pytest

from deploy import session_api_control as control


def test_stop_refuses_a_reused_or_foreign_process(tmp_path, monkeypatch):
    state_path = tmp_path / 'api_process.json'
    state_path.write_text(json.dumps({'pid': 123, 'start_ticks': 'old', 'cwd': '/project/release'}))
    monkeypatch.setattr(control, 'identity', lambda pid: {'pid': pid, 'start_ticks': 'new', 'cwd': '/other'})
    monkeypatch.setattr(control.os, 'kill', lambda *_: pytest.fail('must not signal foreign process'))
    with pytest.raises(RuntimeError, match='refusing to signal'):
        control.stop(state_path)
    assert state_path.exists()


def test_stop_signals_only_verified_process_and_clears_state(tmp_path, monkeypatch):
    path = tmp_path / 'api_process.json'
    path.write_text(json.dumps({'pid': 123, 'start_ticks': 'same', 'cwd': '/project/release'}))
    checks = iter([True, False])
    monkeypatch.setattr(control, 'owned', lambda state: next(checks))
    signals = []
    monkeypatch.setattr(control.os, 'kill', lambda *args: signals.append(args))
    assert control.stop(path) == {'stopped': True, 'was_running': True}
    assert signals == [(123, control.signal.SIGTERM)] and not path.exists()


def test_publish_selection_excludes_data_reports_and_credentials(monkeypatch):
    from deploy import publish_session_candidate as publish
    monkeypatch.setattr(publish.subprocess, 'check_output', lambda *a, **kw: (
        b'api_server.py\0session_embeddings.py\0.env\0.env.example\0data/index/dense.faiss\0'
        b'reports/private.py\0tests/test_session_release.py\0'))
    selected = publish.selected_files()
    assert 'api_server.py' in selected and 'session_embeddings.py' in selected
    assert 'tests/test_session_release.py' in selected
    assert not any(name.startswith(('.env', 'data/', 'reports/')) for name in selected)


def test_failed_promotion_preserves_config_database_and_release_pointer(tmp_path, monkeypatch):
    import sqlite3
    root = tmp_path / 'server'
    runtime = root / 'session_runtime'
    runtime.mkdir(parents=True)
    (root / '.env').write_text('UNCHANGED=configuration\n')
    (runtime / 'api.log').write_text('prior log\n')
    with sqlite3.connect(runtime / 'chat_sessions.sqlite3') as db:
        db.execute('CREATE TABLE evidence(value TEXT)')
        db.execute("INSERT INTO evidence VALUES ('keep')")
    def fail(*args):
        raise RuntimeError('startup failed')
    monkeypatch.setattr(control, 'start', fail)
    with pytest.raises(RuntimeError, match='startup failed'):
        control.promote(root, root / 'session_releases/new', runtime, 19495)
    assert (root / '.env').read_text() == 'UNCHANGED=configuration\n'
    assert not (root / 'session_current').exists()
    assert (runtime / 'api.log').read_text() == 'prior log\n'
    record = json.loads((runtime / 'release_backup.json').read_text())
    from pathlib import Path
    with sqlite3.connect(Path(record['backup']) / 'chat_sessions.sqlite3') as db:
        assert db.execute('SELECT value FROM evidence').fetchone() == ('keep',)


def test_rollback_refuses_an_unmanaged_pointer(tmp_path, monkeypatch):
    runtime = tmp_path / 'session_runtime'
    runtime.mkdir()
    (runtime / 'release_backup.json').write_text(json.dumps({'release': '/expected'}))
    (tmp_path / 'session_current').write_text('foreign file')
    monkeypatch.setattr(control, 'stop', lambda *args: pytest.fail('must not stop before identity check'))
    with pytest.raises(RuntimeError, match='pointer changed'):
        control.rollback(tmp_path, runtime)
