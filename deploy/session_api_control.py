"""Start/stop only a verified session API process; never operate the P1 stack."""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import shutil
import sqlite3
import subprocess
import time
from uuid import uuid4
from pathlib import Path

import requests
from dotenv import dotenv_values


def identity(pid):
    proc = Path('/proc') / str(pid)
    # stat's comm field can contain spaces; fields after ')' start at state.
    fields = proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'start_ticks': fields[19], 'cwd': str(proc.joinpath('cwd').resolve())}


def owned(state):
    try:
        actual = identity(state['pid'])
        return actual == {key: state[key] for key in ['pid', 'start_ticks', 'cwd']}
    except (OSError, KeyError, ValueError):
        return False


def stop(state_path):
    if not state_path.exists():
        return {'stopped': True, 'was_running': False}
    state = json.loads(state_path.read_text())
    if not owned(state):
        raise RuntimeError('process identity does not match; refusing to signal')
    os.kill(state['pid'], signal.SIGTERM)
    for _ in range(150):
        if not owned(state):
            state_path.unlink()
            return {'stopped': True, 'was_running': True}
        time.sleep(.1)
    raise RuntimeError('owned API did not exit within stop deadline')


def start(root, release, runtime, port, ledger=None):
    root, release, runtime = root.resolve(), release.resolve(), runtime.resolve()
    if root / 'session_releases' not in release.parents:
        raise RuntimeError('release must be within the project session releases directory')
    if runtime != root / 'session_runtime' and root / 'session_validation' not in runtime.parents:
        raise RuntimeError('runtime must be a dedicated session API directory')
    manifest = json.loads((release / 'release_manifest.json').read_text())
    import hashlib
    for name, digest in manifest['files'].items():
        path = release / name
        if release not in path.resolve().parents or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError('release file validation failed')
    runtime.mkdir(parents=True, exist_ok=True, mode=0o700)
    state_path = runtime / 'api_process.json'
    if state_path.exists():
        raise RuntimeError('process state already exists; use verified stop before start')
    for required in ['data/index/dense.faiss', 'data/index/retrieval_index.pkl']:
        if not (root / required).is_file():
            raise RuntimeError('existing manual index missing; automatic rebuilding is disabled')
    environment = {**os.environ, **{k: v for k, v in dotenv_values(root / '.env').items() if v is not None}}
    # These are the existing P1 defaults, not new manual model credentials.
    environment['MANUAL_DENSE_ENABLED'] = environment.get('MANUAL_DENSE_ENABLED', environment.get('P1_MANUAL_DENSE_ENABLED', '0'))
    environment['RERANK_ENABLED'] = environment.get('RERANK_ENABLED', environment.get('P1_RERANK_ENABLED', '0'))
    environment.update({'CHAT_SESSION_TREE_ENABLED': '1', 'CHAT_SESSION_MODEL_SUMMARY': '1',
        'CHAT_SESSION_TREE_EMBEDDING': 'semantic', 'CHAT_SESSION_TREE_RETRIEVAL': 'collapsed',
        'CHAT_SESSION_TTL_S': '3600',
        'CHAT_SESSION_SUMMARY_MODEL': environment.get('CHAT_SESSION_SUMMARY_MODEL') or 'qwen3.7-flash',
        'CHAT_SESSION_DB_PATH': str(runtime / 'chat_sessions.sqlite3'),
        'CHAT_API_RAW_PATH': str(runtime / 'api_trace.jsonl'),
        'CHAT_API_TRACE_PATH': str(runtime / 'agent_trace.jsonl'),
        'LLM_CONTEXT_WINDOW_TOKENS': '32768', 'LLM_CONTEXT_SAFETY_TOKENS': '1024',
        'LLM_IMAGE_TOKEN_RESERVE': '8192'})
    if ledger:
        environment['MODEL_ATTEMPT_LEDGER'] = str(ledger.resolve())
    else:
        environment.pop('MODEL_ATTEMPT_LEDGER', None)
    # A socket reservation detects a conflicting listener. A racing bind is
    # handled by the startup gate; no existing process is stopped.
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', port))
    with (runtime / 'api.log').open('ab') as log:
        process = subprocess.Popen([str(root / '.venv/bin/python'), '-m', 'uvicorn', 'api_server:app',
            '--host', '127.0.0.1', '--port', str(port), '--workers', '1',
            '--timeout-graceful-shutdown', '10'], cwd=release, env=environment,
            stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    state = {**identity(process.pid), 'port': port}
    state_path.write_text(json.dumps(state))
    os.chmod(state_path, 0o600)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if process.poll() is not None:
            state_path.unlink(missing_ok=True)
            raise RuntimeError('API exited during startup; inspect private runtime log')
        try:
            health = requests.get(f'http://127.0.0.1:{port}/health', timeout=2).json()
            if health.get('status') == 'ok' and health.get('engine_ready') and health.get('auth_configured'):
                return {'started': True, 'engine_ready': True, 'auth_configured': True,
                        'loopback_only': True, 'port': port,
                        'manual_dense_enabled': environment['MANUAL_DENSE_ENABLED'] in {'1', 'true'},
                        'video_exact_ready': health.get('video_retrieval', {}).get('exact_ready', False)}
        except (requests.RequestException, ValueError):
            pass
        time.sleep(.3)
    stop(state_path)
    raise RuntimeError('API readiness gate failed; new process stopped')


def promote(root, release, runtime, port):
    """Back up this instance, start it, then atomically publish its release link."""
    root, release, runtime = root.resolve(), release.resolve(), runtime.resolve()
    if runtime != root / 'session_runtime':
        raise RuntimeError('formal promotion requires the dedicated formal runtime')
    pointer = root / 'session_current'
    if pointer.exists() and not pointer.is_symlink():
        raise RuntimeError('refusing to replace an unmanaged release pointer')
    if (runtime / 'api_process.json').exists():
        raise RuntimeError('an instance state already exists; refusing to replace it')
    prior = os.readlink(pointer) if pointer.is_symlink() else None
    backup = root / '.env_backups' / ('session_release_' + uuid4().hex)
    backup.mkdir(parents=True, mode=0o700)
    shutil.copy2(root / '.env', backup / 'configuration.env.bak')
    os.chmod(backup / 'configuration.env.bak', 0o600)
    database = runtime / 'chat_sessions.sqlite3'
    if database.exists():
        with sqlite3.connect(database) as source, sqlite3.connect(backup / 'chat_sessions.sqlite3') as target:
            source.backup(target)
    runtime.mkdir(parents=True, exist_ok=True, mode=0o700)
    record = {'previous_pointer': prior, 'release': str(release), 'backup': str(backup),
              'configuration_modified': False}
    (runtime / 'release_backup.json').write_text(json.dumps(record))
    result = start(root, release, runtime, port)
    pending = root / ('session_current_' + uuid4().hex)
    try:
        pending.symlink_to(release, target_is_directory=True)
        os.replace(pending, pointer)
    except Exception:
        pending.unlink(missing_ok=True)
        stop(runtime / 'api_process.json')
        raise
    return {**result, 'promoted': True, 'configuration_backed_up': True,
            'prior_database_backed_up': database.exists() and (backup / 'chat_sessions.sqlite3').exists()}


def rollback(root, runtime):
    root, runtime = root.resolve(), runtime.resolve()
    if runtime != root / 'session_runtime':
        raise RuntimeError('rollback requires the dedicated formal runtime')
    record = json.loads((runtime / 'release_backup.json').read_text())
    pointer = root / 'session_current'
    if not pointer.is_symlink() or pointer.resolve() != Path(record['release']).resolve():
        raise RuntimeError('release pointer changed; refusing to overwrite it')
    result = stop(runtime / 'api_process.json')
    if record['previous_pointer'] is None:
        pointer.unlink()
    else:
        pending = root / ('session_rollback_' + uuid4().hex)
        pending.symlink_to(record['previous_pointer'], target_is_directory=True)
        os.replace(pending, pointer)
    # start/promote do not edit .env. Do not overwrite later unrelated edits.
    return {**result, 'rolled_back': True, 'configuration_modified': False, 'logs_preserved': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'stop', 'status', 'promote', 'rollback'])
    parser.add_argument('--root', type=Path, default=Path.home() / 'rag3d-video')
    parser.add_argument('--release', type=Path)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--port', type=int, default=19495)
    parser.add_argument('--ledger', type=Path)
    args = parser.parse_args()
    try:
        if args.action in {'start', 'promote'}:
            if args.release is None:
                parser.error('--release is required for start')
            if args.action == 'promote':
                result = promote(args.root, args.release, args.runtime, args.port)
            else:
                result = start(args.root, args.release, args.runtime, args.port, args.ledger)
        elif args.action == 'rollback':
            result = rollback(args.root, args.runtime)
        elif args.action == 'stop':
            result = stop(args.runtime / 'api_process.json')
        else:
            state_path = args.runtime / 'api_process.json'
            result = {'running': state_path.exists() and owned(json.loads(state_path.read_text()))}
    except Exception as exc:
        print(json.dumps({'ok': False, 'error_type': type(exc).__name__, 'reason': str(exc)}))
        return 2
    print(json.dumps({'ok': True, **result}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
