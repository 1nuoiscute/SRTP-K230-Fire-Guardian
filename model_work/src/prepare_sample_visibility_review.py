"""Decode all samples of one registered development clip for visibility-only review."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from data_integrity import sha256


def prepare(comparison,video_index,out):
    if out.exists(): raise ValueError('Existing sample visibility review')
    plan=json.loads((comparison/'plan.json').read_text(encoding='utf-8'))
    complete=json.loads((comparison/'comparison_complete.json').read_text(encoding='utf-8'))
    records=[json.loads(s) for s in (comparison/'per_frame.jsonl').read_text(encoding='utf-8').splitlines()]
    selected=[r for r in records if r['video_index']==video_index]
    video=Path(plan['videos'][video_index-1]['path']); expected=plan['videos'][video_index-1]['sha256']
    if sha256(video)!=expected or not selected:
        raise ValueError('Wrong registered source')
    requested={r['frame_index']:r for r in selected}; cap=cv2.VideoCapture(str(video))
    out.mkdir(parents=True); tiles=[]; rows=[]; index=0
    try:
        while True:
            ok,im=cap.read()
            if not ok: break
            if index in requested:
                r=requested[index]
                digest=hashlib.sha256(np.ascontiguousarray(im).tobytes()).hexdigest()
                h,w=im.shape[:2]
                if digest!=r['source_pixel_sha256'] or (w,h)!=(r['width'],r['height']):
                    raise ValueError('Decoded pixels differ from compared sample')
                path=out/f"sample_{r['sample_index']:04}.png"
                if not cv2.imwrite(str(path),im): raise ValueError('Sample encoding failed')
                tile=np.full((466,640,3),25,np.uint8)
                overview=cv2.resize(im,(240,426)); tile[40:,0:240]=overview
                # Fixed bottom half shows pot/burner without enhancing or hiding originals.
                crop=im[h//2:]; ch,cw=crop.shape[:2]; gain=min(400/cw,426/ch)
                detail=cv2.resize(crop,(round(cw*gain),round(ch*gain)))
                dh,dw=detail.shape[:2]; top,left=40+(426-dh)//2,240+(400-dw)//2
                tile[top:top+dh,left:left+dw]=detail
                cv2.putText(tile,f"Sample {r['sample_index']:02} / t={r['time_seconds']:.2f}s / raw + lower half",(8,26),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,240,240),1)
                rows.append(dict(sample_index=r['sample_index'],frame_index=index,time_seconds=r['time_seconds'],size=[w,h],
                    pixel_sha256=digest,file=path.name,file_sha256=sha256(path),crop_xyxy=[0,h//2,w,h]))
                tiles.append(tile)
            index+=1
    finally: cap.release()
    if len(rows)!=len(selected) or index!=complete['clips'][video_index-1]['decoded_frames']:
        raise ValueError('Incomplete matching decode')
    sheets=[]
    for start in range(0,len(tiles),4):
        page=tiles[start:start+4]
        while len(page)<4: page.append(np.zeros((466,640,3),np.uint8))
        path=out/f'sheet_{start//4+1:02}.jpg'
        if not cv2.imwrite(str(path),np.concatenate([np.concatenate(page[:2],1),np.concatenate(page[2:],1)],0)): raise ValueError('Sheet failed')
        sheets.append(dict(file=path.name,sha256=sha256(path)))
    meta=dict(scope='Visibility review of every already-exposed sampled frame; no predictions shown, no dense video/event truth',
        source_video_sha256=expected,comparison_plan_sha256=sha256(comparison/'plan.json'),per_frame_sha256=sha256(comparison/'per_frame.jsonl'),
        script_sha256=sha256(Path(__file__)),sample_fps=plan['requested_sample_fps'],independent_kitchen=False,
        all_sample_pixels_match=True,decoded_frames=index,samples=len(rows),images=rows,sheets=sheets)
    (out/'review_inputs.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    print(f'{len(rows)} identical samples / {len(sheets)} visibility sheets',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--comparison',type=Path,required=True)
    p.add_argument('--video-index',type=int,required=True); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); prepare(a.comparison,a.video_index,a.out)
