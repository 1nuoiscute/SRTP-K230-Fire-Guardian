"""Zoom all four v20 train-positive envelopes and re-decode CPU preview."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from data_integrity import sha256


def prepare(data, images, runtime, out):
    if out.exists():raise ValueError('Existing detail review')
    meta=json.loads((data/'build_manifest.json').read_text(encoding='utf-8'))
    comparison=json.loads((images/'comparison_complete.json').read_text(encoding='utf-8'))
    if sha256(data/'build_manifest.json')!=comparison['plan']['data_manifest_sha256']:raise ValueError('Changed data')
    rows=[r for r in meta['lineage'] if r.get('v20_new_source') and r['boxes_xyxy']]
    if len(rows)!=4 or sum(len(r['boxes_xyxy']) for r in rows)!=4:raise ValueError('Wrong positive scope')
    out.mkdir(parents=True); pages=[]
    def save(name, tiles, inputs):
        path=out/name
        if not cv2.imwrite(str(path),np.concatenate(tiles,axis=1)):raise ValueError('Encoding failed')
        pages.append(dict(file=name,sha256=sha256(path),inputs=inputs))
    for row in rows:
        raw=data/'images/train'/row['image']
        if sha256(raw)!=row['sha256']:raise ValueError('Source changed')
        h,w=cv2.imread(str(raw)).shape[:2];x1,y1,x2,y2=map(int,row['boxes_xyxy'][0])
        crop=[max(0,x1-60),max(0,y1-60),min(w,x2+60),min(h,y2+60)]
        tiles=[];inputs=[]
        for label in ['v16_best','v20_best','v20_last']:
            p=images/(label+'_new_sources')/row['image'];im=cv2.imread(str(p))
            a,b,c,d=crop;im=im[b:d,a:c];scale=min(630/im.shape[1],370/im.shape[0])
            im=cv2.resize(im,(round(im.shape[1]*scale),round(im.shape[0]*scale)))
            ih,iw=im.shape[:2];tile=np.full((420,640,3),24,np.uint8);left=(640-iw)//2;top=40+(370-ih)//2
            tile[top:top+ih,left:left+iw]=im
            cv2.putText(tile,label+' '+row['image'],(8,26),cv2.FONT_HERSHEY_SIMPLEX,.55,(240,240,240),1)
            tiles.append(tile);inputs.append(dict(path=str(p),sha256=sha256(p),crop_xyxy=crop))
        save(row['image'].replace('.jpg','_positive_zoom.jpg'),tiles,inputs)
    summary=json.loads((runtime/'summary.json').read_text(encoding='utf-8'));video=runtime/'boxed.mp4'
    digest=sha256(video);cap=cv2.VideoCapture(str(video));count=0;tiles=[]
    if not cap.isOpened() or summary['sampled_frames']!=27:raise ValueError('Wrong CPU preview scope')
    while True:
        ok,im=cap.read()
        if not ok:break
        if count in [0,13,26]:
            h,w=im.shape[:2];scale=min(630/w,370/h);im=cv2.resize(im,(round(w*scale),round(h*scale)))
            ih,iw=im.shape[:2];tile=np.full((420,640,3),24,np.uint8);left=(640-iw)//2;top=40+(370-ih)//2
            tile[top:top+ih,left:left+iw]=im
            cv2.putText(tile,'Independent CPU preview sample '+str(count),(8,26),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,240,240),1)
            tiles.append(tile)
        count+=1
    cap.release()
    if count!=27 or len(tiles)!=3 or sha256(video)!=digest:raise ValueError('CPU preview decode differs')
    save('cpu_video_three_samples.jpg',tiles,dict(path=str(video),sha256=digest,samples=[0,13,26]))
    (out/'review_inputs.json').write_text(json.dumps(dict(role='Detail pages; actual visual review required',
        script_sha256=sha256(Path(__file__)),image_comparison_sha256=sha256(images/'comparison_complete.json'),
        cpu_summary_sha256=sha256(runtime/'summary.json'),cpu_preview_decoded_frames=count,pages=pages),indent=2)+'\n',encoding='utf-8')
    print('Four positive envelope zooms / CPU preview independently decoded 27 frames')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data','images','runtime','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();prepare(a.data,a.images,a.runtime,a.out)
