"""Read-only verification of the local board handoff snapshot; no device access."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / 'docs/board_handoff_artifacts_20260930.json').read_text(encoding='utf-8'))
    failures = []
    for entry in manifest['entries']:
        path = (root / entry['local_path']).resolve()
        if not path.is_relative_to(root):
            failures.append('Path outside workspace: ' + entry['id'])
            continue
        if not path.is_file():
            failures.append('Missing local artifact: ' + entry['id'])
            continue
        if path.stat().st_size != entry['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            failures.append('Changed local artifact: ' + entry['id'])
    if failures:
        raise SystemExit('\n'.join(failures))
    print(f"Local handoff identities verified: {len(manifest['entries'])} files; no live board acceptance implied")


if __name__ == '__main__':
    main()
