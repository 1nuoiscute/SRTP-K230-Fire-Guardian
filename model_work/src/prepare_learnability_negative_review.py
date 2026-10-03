"""Prepare negative-source pages from completed train-positive diagnostics."""
import argparse
import json
from pathlib import Path
from data_integrity import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['run','data','out']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('Existing review output')
    import cv2
    import numpy as np
    complete=json.loads((a.run/'complete.json').read_text(encoding='utf-8'))
    plan=json.loads((a.run/'plan.json').read_text(encoding='utf-8'))
    if sha256(a.run/'plan.json')!=complete['plan_sha256'] or sha256(a.data/'build_manifest.json')!=plan['dataset']['manifest_sha256']:
        raise ValueError('Diagnostic identity changed')
    names=[r['name'] for r in plan['arms']]
    if len(names)!=4 or any(complete['completed_arms'][n]['actual_optimizer_steps']!=80 for n in names):
        raise ValueError('Incomplete paired diagnostic')
    for n in names:
        for file,digest in complete['completed_arms'][n]['artifacts'].items():
            if sha256(a.run/n/file)!=digest:raise ValueError('Changed arm artifact')
    rows=[r for r in plan['inputs'] if not r['boxes']]
    frames={n:{r['image']:r for r in complete['completed_arms'][n]['final']['evaluations']['full_rect']['frames']} for n in names}
    rows=[r for r in rows if any(frames[n][r['image']]['predictions'] for n in names)]
    a.out.mkdir(parents=True);pages=[]
    for row in rows:
        path=a.data/'images/train'/row['image']
        if sha256(path)!=row['sha256']:raise ValueError('Changed source image')
        raw=cv2.imread(str(path));tiles=[]
        for n in names:
            im=raw.copy();preds=frames[n][row['image']]['predictions']
            for pred in preds:
                x1,y1,x2,y2=map(round,pred['xyxy']);cv2.rectangle(im,(x1,y1),(x2,y2),(255,0,0),2)
            h,w=im.shape[:2];scale=min(510/w,360/h);im=cv2.resize(im,(round(w*scale),round(h*scale)))
            ih,iw=im.shape[:2];tile=np.full((420,520,3),24,np.uint8)
            tile[50:50+ih,(520-iw)//2:(520-iw)//2+iw]=im
            cv2.putText(tile,n+' boxes '+str(len(preds)),(8,23),cv2.FONT_HERSHEY_SIMPLEX,.55,(240,240,240),1)
            tiles.append(tile)
        target=a.out/row['image']
        if not cv2.imwrite(str(target),np.concatenate(tiles,axis=1)):raise RuntimeError('Encoding failed')
        pages.append(dict(image=row['image'],source_sha256=row['sha256'],file=target.name,sha256=sha256(target)))
    (a.out/'review_inputs.json').write_text(json.dumps(dict(role='Review inputs; no automatic human-review claim',
        complete_sha256=sha256(a.run/'complete.json'),script_sha256=sha256(Path(__file__)),pages=pages),indent=2)+'\n',encoding='utf-8')
    print('Negative-source review pages:',len(pages))


if __name__=='__main__':main()
