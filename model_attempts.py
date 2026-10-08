"""Optional cross-process attempt ledger for explicitly bounded live validation."""
from __future__ import annotations

import json
import os
from pathlib import Path


def record_attempt(kind):
    filename = os.getenv('MODEL_ATTEMPT_LEDGER', '')
    if not filename:
        return
    lock_path = Path(filename + '.lock')
    with lock_path.open('a+b') as lock:
        if os.name == 'nt':
            import msvcrt
            lock.seek(0)
            lock.write(b'0')
            lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            path = Path(filename)
            state = json.loads(path.read_text(encoding='utf-8'))
            if state['attempts'] >= state['limit']:
                raise TimeoutError('live validation attempt budget exhausted')
            state['attempts'] += 1
            state.setdefault('kinds', {})[kind] = state.get('kinds', {}).get(kind, 0) + 1
            temporary = path.with_suffix('.pending')
            temporary.write_text(json.dumps(state), encoding='utf-8')
            os.replace(temporary, path)
        finally:
            if os.name == 'nt':
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
