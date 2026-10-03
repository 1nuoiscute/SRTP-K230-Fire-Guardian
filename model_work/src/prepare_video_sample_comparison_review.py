"""Render explicit sampled frames, including errors hidden by equal box counts."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from data_integrity import sha256


def prepare(comparison, video_index, samples, out):
    if out.exists():raise ValueError('Existing review')
    plan=json.loads((comparison/'plan.json').read_text(encoding='utf-8'))
    complete=json.loads((comparison/'comparison_complete.json').read_text(encoding='utf-8'))
    if sha256(comparison/'plan.json')!=complete['plan_sha256'] or sha256(comparison/'per_frame.jsonl')!=complete['per_frame_sha256']:raise ValueError('Completed comparison changed')
    if not 1<=video_index<=len(plan['videos']) or not samples or len(samples)!=len(set(samples)):raise ValueError('Invalid selection')
    records=[json.loads(s) for s in (comparison/'per_frame.jsonl').read_text(encoding='utf-8').splitlines()]
    wanted={r['frame_index']:r for r in records if r['video_index']==video_index and r['sample_index'] in samples}
    if {r['sample_index'] for r in wanted.values()}!=set(samples):raise ValueError('Missing sample')
    source=plan['videos'][video_index-1];video=Path(source['path']);labels=[m['label'] for m in plan['models']]
    if sha256(video)!=source['sha256']:raise ValueError('Video changed')
    cap=cv2.VideoCapture(str(video));index=0;rows=[];inputs=[]
    if not cap.isOpened():raise ValueError('Could not decode source')
    out.mkdir(parents=True)
    try:
        while True:
            ok,im=cap.read()
            if not ok:break
            if index in wanted:
                r=wanted[index];h,w=im.shape[:2]
                if hashlib.sha256(np.ascontiguousarray(im).tobytes()).hexdigest()!=r['source_pixel_sha256'] or [w,h]!=[r['width'],r['height']]:raise ValueError('Compared pixels differ')
                tiles=[];scale=min(640/w,420/h);resized=cv2.resize(im,(round(w*scale),round(h*scale)));rh,rw=resized.shape[:2]
                for label in labels:
                    tile=np.full((460,640,3),24,np.uint8);left=(640-rw)//2;top=40+(420-rh)//2;tile[top:top+rh,left:left+rw]=resized
                    for b in r['predictions'][label]:
                        x1,y1,x2,y2=b['xyxy'];cv2.rectangle(tile,(round(x1*scale+left),round(y1*scale+top)),(round(x2*scale+left),round(y2*scale+top)),(255,90,0),2)
                    cv2.putText(tile,f"{label} v{video_index} sample={r['sample_index']}",(8,26),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,240,240),1)
                    tiles.append(tile)
                rows.append(np.concatenate(tiles,axis=1));inputs.append(r)
            index+=1
    finally:cap.release()
    if index!=complete['clips'][video_index-1]['decoded_frames'] or len(rows)!=len(samples) or sha256(video)!=source['sha256']:raise ValueError('Decode inventory changed')
    path=out/'selected_samples.jpg'
    if not cv2.imwrite(str(path),np.concatenate(rows,axis=0)):raise ValueError('Encoding failed')
    (out/'review_inputs.json').write_text(json.dumps(dict(role='Explicit sampled frames; equal counts do not prove correctness',
        script_sha256=sha256(Path(__file__)),comparison_sha256=sha256(comparison/'comparison_complete.json'),
        per_frame_sha256=sha256(comparison/'per_frame.jsonl'),selection=samples,video_index=video_index,
        samples=inputs,pages=[dict(file=path.name,sha256=sha256(path))]),indent=2)+'\n',encoding='utf-8')
    print(f'{len(samples)} identity-matched samples prepared')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--comparison',type=Path,required=True)
    p.add_argument('--video-index',type=int,required=True);p.add_argument('--samples',type=int,nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();prepare(a.comparison,a.video_index,a.samples,a.out)
