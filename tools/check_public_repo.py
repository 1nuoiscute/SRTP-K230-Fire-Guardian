"""Validate tracked public source, small artifacts and top-level navigation."""
import ast
import json
import re
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT)
files = [ROOT / name for name in tracked.decode('utf-8').split('\0') if name]
if not files:
    raise SystemExit('No tracked/staged public files: git add the explicit public allowlist first')
errors=[]
for path in files:
    rel=path.relative_to(ROOT).as_posix()
    if path.stat().st_size>5_000_000: errors.append(f'Unexpected large public file: {rel}')
    if path.suffix in {'.pt','.elf','.kmodel','.mp4','.webm','.ogv','.onnx','.img','.zip','.o'}: errors.append(f'Local artifact tracked: {rel}')
    if path.suffix=='.py':
        try: ast.parse(path.read_text(encoding='utf-8-sig'),filename=rel)
        except SyntaxError as error: errors.append(f'Python syntax: {rel}: {error}')
    if path.suffix=='.json':
        try: json.loads(path.read_text(encoding='utf-8-sig'))
        except ValueError as error: errors.append(f'JSON invalid: {rel}: {error}')
    if path.suffix in {'.py','.md','.sh','.c','.h','.json','.yaml','.yml'}:
        text=path.read_text(encoding='utf-8-sig')
        for pattern in [r'gh[pousr]_[A-Za-z0-9]{30,}',r'github_pat_[A-Za-z0-9_]{40,}',r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',r'sk-[A-Za-z0-9_-]{32,}']:
            if re.search(pattern,text): errors.append(f'Credential-shaped content: {rel}')
for target in re.findall(r'\]\(([^)]+)\)',(ROOT/'README.md').read_text(encoding='utf-8')):
    if '://' not in target and not (ROOT/target.split('#')[0]).is_file(): errors.append(f'Missing README target: {target}')
if errors: raise SystemExit('\n'.join(errors))
print(f'Public source check passed: {len(files)} files; syntax, JSON, artifacts, credential patterns, README navigation')
