"""Render every shared video sample with differing prediction counts for manual review."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from data_integrity import sha256


def prepare(comparison, out):
    if out.exists(): raise ValueError('Refusing overwrite')
    plan = json.loads((comparison / 'plan.json').read_text(encoding='utf-8'))
    complete = json.loads((comparison / 'comparison_complete.json').read_text(encoding='utf-8'))
    if complete['plan_sha256'] != sha256(comparison / 'plan.json'): raise ValueError('Plan changed')
    rows = [json.loads(x) for x in (comparison / 'per_frame.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(rows) != sum(c['sampled_frames'] for c in complete['clips']): raise ValueError('Incomplete samples')
    selected = [r for r in rows if len({len(p) for p in r['predictions'].values()}) > 1]
    out.mkdir(parents=True)
    rendered = []
    for vi, source in enumerate(plan['videos'], 1):
        wanted = {r['frame_index']: r for r in selected if r['video_index'] == vi}
        if not wanted: continue
        video = Path(source['path'])
        if sha256(video) != source['sha256']: raise ValueError('Video changed')
        cap = cv2.VideoCapture(str(video)); index = 0; found = set()
        try:
            while True:
                ok, im = cap.read()
                if not ok: break
                if index in wanted:
                    r = wanted[index]; h, w = im.shape[:2]
                    if (w, h) != (r['width'], r['height']) or hashlib.sha256(np.ascontiguousarray(im).tobytes()).hexdigest() != r['source_pixel_sha256']:
                        raise ValueError('Decoded pixels changed')
                    tiles = []; ratio = min(640/w, 420/h)
                    resized = cv2.resize(im, (round(w*ratio), round(h*ratio)))
                    rh, rw = resized.shape[:2]; top, left = 40+(420-rh)//2, (640-rw)//2
                    for model, boxes in r['predictions'].items():
                        tile = np.full((460, 640, 3), 25, np.uint8)
                        tile[top:top+rh, left:left+rw] = resized
                        for b in boxes:
                            x1,y1,x2,y2=b['xyxy']
                            cv2.rectangle(tile,(round(x1*ratio+left),round(y1*ratio+top)),(round(x2*ratio+left),round(y2*ratio+top)),(255,90,0),2)
                        cv2.putText(tile,f"{model} v{vi} sample={r['sample_index']} boxes={len(boxes)}",(8,26),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,240,240),1)
                        tiles.append(tile)
                    rendered.append((r, np.concatenate(tiles,1))); found.add(index)
                index += 1
        finally: cap.release()
        if found != set(wanted): raise ValueError('Missing selected frames')
        if index != complete['clips'][vi-1]['decoded_frames']: raise ValueError('Decode count changed')
        if sha256(video) != source['sha256']: raise ValueError('Video changed during decode')
    if len(rendered) != len(selected): raise ValueError('Incomplete review')
    sheets=[]
    for start in range(0,len(rendered),3):
        page=[r[1] for r in rendered[start:start+3]]
        while len(page)<3: page.append(np.zeros_like(page[0]))
        path=out/f'sheet_{start//3+1:02}.jpg'
        if not cv2.imwrite(str(path),np.concatenate(page,0)): raise ValueError('Encoding failed')
        sheets.append(dict(file=path.name,sha256=sha256(path)))
    manifest=dict(role='All count-different samples; no count-based correctness inference',
        comparison_sha256=sha256(comparison/'comparison_complete.json'),per_frame_sha256=sha256(comparison/'per_frame.jsonl'),
        script_sha256=sha256(Path(__file__)),selected_samples=len(selected),samples=[r[0] for r in rendered],sheets=sheets)
    (out/'review_inputs.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'{len(selected)} count-different samples / {len(sheets)} pages')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); prepare(a.comparison,a.out)
