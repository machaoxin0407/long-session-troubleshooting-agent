"""Bounded local experiments; SSH secrets exist only in the child environment."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deploy.run_session_live import credentials  # noqa: E402
from session_experiment import CAPS, verify_ledgers  # noqa: E402
from session_experiment_data import require_review  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=['development', 'freeze', 'test', 'supplement'], required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    require_review(output, 'development' if args.phase == 'development' else None)
    verify_ledgers(output)
    if args.phase != 'development' and args.phase != 'freeze' and not (output / 'freeze.json').exists():
        raise ValueError('frozen experiment required')
    config = credentials()
    environment = {**os.environ, **config, 'SILICONFLOW_ONLY': '1',
                   'CHAT_SESSION_SUMMARY_MODEL': 'qwen3.7-flash',
                   'LLM_CONTEXT_WINDOW_TOKENS': '32768', 'LLM_CONTEXT_SAFETY_TOKENS': '1024'}
    if args.phase != 'freeze':
        environment['MODEL_ATTEMPT_LEDGER'] = str(output / f'ledger_{args.phase}.json')
    command = [sys.executable, 'benchmark_session_memory.py', '--experiment', '--online',
               '--phase', args.phase, '--output', str(output)]
    process = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True,
                             text=True, encoding='utf-8', errors='replace', timeout=21600)
    for kind, content in [('stdout', process.stdout), ('stderr', process.stderr)]:
        for name, value in config.items():
            if name.endswith(('BASE_URL', 'API_KEY')) and value:
                content = content.replace(value, '[private configuration]')
        # Never overwrite earlier execution logs, including failed launches.
        n = 0
        while (output / f'launch_{args.phase}_{n}_{kind}.log').exists():
            n += 1
        (output / f'launch_{args.phase}_{n}_{kind}.log').write_text(content, encoding='utf-8')
    print(json.dumps({'phase': args.phase, 'exit_code': process.returncode,
                      'attempts': verify_ledgers(output), 'phase_cap': CAPS.get(args.phase)}, ensure_ascii=False))
    return process.returncode


if __name__ == '__main__':
    raise SystemExit(main())
