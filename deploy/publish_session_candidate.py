"""Package current code with hashes; stage without overwriting the server checkout."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shlex
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
SSH = r'C:\Windows\System32\OpenSSH\ssh.exe' if os.name == 'nt' else 'ssh'

BOOTSTRAP = r'''
import sys,tarfile,io,json,hashlib,os
from pathlib import Path,PurePosixPath
root=Path.home()/'rag3d-video'
release_id=sys.argv[1]
if not release_id.replace('_','').isalnum():raise ValueError('invalid release id')
release=root/'session_releases'/release_id
release.mkdir(parents=True,exist_ok=False)
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as archive:
 for member in archive.getmembers():
  name=PurePosixPath(member.name)
  if name.is_absolute() or '..' in name.parts or not member.isfile():raise ValueError('unsafe archive member')
  path=release.joinpath(*name.parts)
  path.parent.mkdir(parents=True,exist_ok=True)
  with archive.extractfile(member) as source,path.open('xb') as target:target.write(source.read())
manifest=json.loads((release/'release_manifest.json').read_text())
for name,digest in manifest['files'].items():
 if hashlib.sha256((release/name).read_bytes()).hexdigest()!=digest:raise ValueError('hash mismatch')
for name in ['data','data_video']:
 if (root/name).is_dir():(release/name).symlink_to(root/name,target_is_directory=True)
(release/'.env').symlink_to(root/'.env')
print(json.dumps({'staged':True,'release_id':release_id,'files':len(manifest['files']),'checkout_overwritten':False,'data_copied':False}))
'''


def selected_files():
    paths = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
                                    cwd=ROOT).decode('utf-8').split('\0')
    selected = []
    for name in sorted(set(paths)):
        path = PurePosixPath(name)
        if (not name or path.parts[0] in {'data', 'data_video', 'reports', 'paper', '.git', '.venv'}
                or path.name.startswith('.env') or not (ROOT / name).is_file()):
            continue
        if path.suffix in {'.py', '.sh', '.json', '.html', '.css', '.js', '.md', '.txt'}:
            selected.append(name)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--release-id', required=True)
    args = parser.parse_args()
    if not args.release_id.replace('_', '').isalnum():
        parser.error('invalid release id')
    names = selected_files()
    manifest = {'release_id': args.release_id, 'files': {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}}
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode='w:gz') as archive:
        for name in names:
            data = (ROOT / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != manifest['files'][name]:
                raise RuntimeError('source changed during packaging')
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o755 if name.endswith('.sh') else 0o644
            archive.addfile(info, io.BytesIO(data))
        data = json.dumps(manifest, indent=2).encode()
        info = tarfile.TarInfo('release_manifest.json')
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    args.output.mkdir(parents=True, exist_ok=True)
    archive_path = args.output / (args.release_id + '.tar.gz')
    with archive_path.open('xb') as file:
        file.write(payload.getvalue())
    args.output.joinpath('release_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    command = '~/rag3d-video/.venv/bin/python -c ' + shlex.quote(BOOTSTRAP) + ' ' + shlex.quote(args.release_id)
    process = subprocess.run([SSH, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
        '-o', 'StrictHostKeyChecking=yes', 'newpulse-server', command], input=payload.getvalue(),
        capture_output=True, timeout=60)
    if process.returncode:
        print(json.dumps({'staged': False, 'details_suppressed': True}))
        return 2
    result = json.loads(process.stdout)
    args.output.joinpath('stage_result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
