"""Prepare identity-bound v20 review pages; this does not assert human review."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from data_integrity import sha256


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def prepare(images, videos, previous_videos, data, out):
    if out.exists(): raise ValueError('Existing review output')
    current=read(videos/'comparison_complete.json'); prior=read(previous_videos/'comparison_complete.json')
    for folder, complete in [(videos,current),(previous_videos,prior)]:
        for key,name in [('plan_sha256','plan.json'),('per_frame_sha256','per_frame.jsonl'),('review_manifest_sha256','review_manifest.json')]:
            if sha256(folder/name)!=complete[key]: raise ValueError('Completed video identity changed')
    old=[json.loads(s) for s in (previous_videos/'per_frame.jsonl').read_text(encoding='utf-8').splitlines()]
    new=[json.loads(s) for s in (videos/'per_frame.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(old)!=307 or len(new)!=307 or current['total_samples']!=307: raise ValueError('Wrong paired scope')
    identity=['video_index','video_sha256','frame_index','sample_index','time_seconds','width','height','source_pixel_sha256']
    controls=['v16_best','v18_last','v19_last']
    for a,b in zip(old,new):
        if any(a[k]!=b[k] for k in identity): raise ValueError('Source frame identity differs')
        if any(a['predictions'][k]!=b['predictions'][k] for k in controls): raise ValueError('Reference prediction differs')
    expected=controls+['v20_best','v20_last']
    if [m['label'] for m in read(videos/'plan.json')['models']]!=expected: raise ValueError('Unexpected sheet rows')
    complete=read(images/'comparison_complete.json')
    if sha256(data/'build_manifest.json')!=complete['plan']['data_manifest_sha256']: raise ValueError('Data changed')
    selected=[r for r in read(data/'build_manifest.json')['lineage'] if r.get('v20_new_source')]
    if len(selected)!=14 or sum(len(r['boxes_xyxy']) for r in selected)!=4: raise ValueError('Wrong development source scope')
    labels=['v16_best','v20_best','v20_last']; source_inputs=[]; pages=[]
    for label in labels:
        summary=read(images/(label+'_new_sources')/'summary.json')
        if summary!=complete['models'][label]['new_training_sources']: raise ValueError('Source summary differs')
        if {r['image'] for r in summary['frames']}!={r['image'] for r in selected}: raise ValueError('Source inventory differs')
    out.mkdir(parents=True)
    def save(name, pixels, origin):
        path=out/name
        if not cv2.imwrite(str(path),pixels): raise ValueError('Page encoding failed')
        pages.append(dict(file=name,sha256=sha256(path),source=origin))
    rows=[]
    for source in sorted(selected,key=lambda r:r['image']):
        raw=data/'images/train'/source['image']
        if sha256(raw)!=source['sha256']: raise ValueError('Raw source changed')
        tiles=[]
        for label in labels:
            p=images/(label+'_new_sources')/source['image']; im=cv2.imread(str(p))
            if im is None: raise ValueError('Missing annotated source')
            h,w=im.shape[:2]; scale=min(630/w,370/h)
            resized=cv2.resize(im,(round(w*scale),round(h*scale)))
            rh,rw=resized.shape[:2];tile=np.full((420,640,3),24,np.uint8)
            left=(640-rw)//2;top=40+(370-rh)//2;tile[top:top+rh,left:left+rw]=resized
            cv2.putText(tile,label+' '+source['image'],(8,26),cv2.FONT_HERSHEY_SIMPLEX,.55,(240,240,240),1)
            tiles.append(tile);source_inputs.append(dict(path=str(p),sha256=sha256(p)))
        rows.append(np.concatenate(tiles,axis=1))
    for start in range(0,len(rows),4):
        page=rows[start:start+4]
        while len(page)<4:page.append(np.zeros_like(rows[0]))
        save(f'new_sources_{start//4+1:02}.jpg',np.concatenate(page,axis=0),'All 14 training-exposed sources, sorted names; green GT / blue prediction')
    manifest=read(videos/'review_manifest.json')
    for item in manifest['videos']:
        path=videos/item['sheet']
        if sha256(path)!=item['sheet_sha256']: raise ValueError('Fixed video sheet changed')
        im=cv2.imread(str(path))
        if im.shape!=(2300,1920,3): raise ValueError('Unexpected five-model page geometry')
        save(f"video_{item['video_index']:02}_three_models.jpg",np.concatenate([im[k*460:(k+1)*460] for k in [0,3,4]],axis=0),dict(file=str(path),sha256=sha256(path),rows_zero_based=[0,3,4]))
    write(out/'review_inputs.json',dict(role='Prepared review pages; actual visual inspection remains required',
        script_sha256=sha256(Path(__file__)),image_comparison_sha256=sha256(images/'comparison_complete.json'),
        video_comparison_sha256=sha256(videos/'comparison_complete.json'),previous_video_comparison_sha256=sha256(previous_videos/'comparison_complete.json'),
        paired_reference_audit=dict(shared_frames=307,exact_reference_predictions=921,controls=controls,identity_fields=identity),
        new_sources=14,new_sources_are_training_exposed=True,reserved_sources_predicted=False,
        annotated_source_inputs=source_inputs,pages=pages))
    print('307 exact shared frames / 921 reference predictions; 14 sources and 9 fixed video pages prepared')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['images','videos','previous-videos','data','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();prepare(a.images,a.videos,a.previous_videos,a.data,a.out)
