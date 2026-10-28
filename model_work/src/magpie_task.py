"""Run a bounded, auditable analysis/code task through the user's Magpie gateway.

Only explicitly supplied project files are sent. No tools or shell execution are
delegated. Outputs are proposals that require local verification by the owner.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import urllib.error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--file', type=Path, action='append', default=[])
    parser.add_argument('--image', type=Path, action='append', default=[])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=180)
    args = parser.parse_args()
    if not args.model.startswith(('copilot/', 'workbuddy/')):
        raise ValueError('Explicit Copilot or WorkBuddy route required')
    args.out.mkdir(parents=True, exist_ok=False)
    task = args.task.read_text(encoding='utf-8')
    files = []
    content = task
    for path in args.file:
        payload = path.read_bytes()
        files.append({'path': str(path), 'sha256': hashlib.sha256(payload).hexdigest()})
        content += '\n\nFILE: ' + str(path) + '\n' + payload.decode('utf-8')
    user_content = content
    if args.image:
        user_content = [{'type': 'text', 'text': content}]
        for path in args.image:
            payload = path.read_bytes()
            mime = 'image/png' if path.suffix.lower() == '.png' else 'image/jpeg'
            files.append({'path': str(path), 'sha256': hashlib.sha256(payload).hexdigest(), 'media_type': mime})
            user_content.extend([{'type': 'text', 'text': 'REVIEW IMAGE: ' + path.name},
                                 {'type': 'image_url', 'image_url': {'url': 'data:' + mime + ';base64,' + base64.b64encode(payload).decode('ascii')}}])
    request = {'model': args.model, 'messages': [
        {'role': 'system', 'content': 'You are an SRTP computer vision collaborator. Follow the supplied task. Distinguish evidence from inference. You cannot execute code or inspect files other than the supplied text. Never claim experiments were run. Return concrete, reviewable work.'},
        {'role': 'user', 'content': user_content}], 'stream': False}
    raw = json.dumps(request, ensure_ascii=False).encode('utf-8')
    (args.out / 'request.json').write_bytes(raw)
    started = time.time()
    meta = {'requested_model': args.model, 'files': files, 'started_unix': started,
            'request_sha256': hashlib.sha256(raw).hexdigest(), 'executed_tools': False}
    try:
        req = urllib.request.Request('http://127.0.0.1:3425/v1/chat/completions', data=raw,
                                     headers={'Authorization': 'Bearer magpie', 'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=args.timeout) as response:
            result = json.load(response)
        (args.out / 'response.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        text = result['choices'][0]['message'].get('content') or ''
        if not isinstance(text, str) or not text.strip():
            raise ValueError('No usable answer content')
        (args.out / 'answer.md').write_text(text, encoding='utf-8')
        meta.update(status='completed', returned_model=result.get('model'), usage=result.get('usage'),
                    response_sha256=hashlib.sha256((args.out / 'response.json').read_bytes()).hexdigest())
        print(json.dumps({'status': 'completed', 'model': result.get('model'), 'out': str(args.out), 'chars': len(text)}))
    except Exception as error:
        meta.update(status='failed', error_type=type(error).__name__, error=str(error))
        if isinstance(error, urllib.error.HTTPError):
            (args.out / 'error_body.txt').write_bytes(error.read())
        raise
    finally:
        meta['elapsed_seconds'] = time.time() - started
        (args.out / 'task_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
