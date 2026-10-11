"""Mine train-only full contexts for actual flame/negative review and build v19."""
import argparse
from collections import Counter
from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import shutil
import cv2
import numpy as np
import yaml
from data_integrity import sha256, validate_yolo
from extend_flame_review_v17 import read, write, verify as verify_parent, quantiles
from fire_label_revision_dataset import validation_snapshot

PARENT_SHA='8e70dea69f39beafa0f17b3a6d4b483c537622fdf52271755fd8f3e2447bd54d'
MINER_SHA='2534a8c727157d0333bf7e0b785cd09e950f5e6e0192831947ba26cfe865b3ce'


def csv_rows(path):
    with path.open(encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def check_inventory(meta,rows):
    raw=csv_rows(Path(meta['base'])/'manifest.csv')
    train={r['file']:r for r in raw if r['split']=='train'}
    forbidden={r['sha256'] for r in raw if r['split']!='train'}
    if len(rows)!=40 or len({r['image_sha256'] for r in rows})!=40:
        raise ValueError('Incomplete or repeated review pixels')
    for r in rows:
        name=Path(r['image']).name; source=train.get(name)
        if source is None or source['sha256']!=r['image_sha256'] or r['image_sha256'] in forbidden:
            raise ValueError('Non-train or overlapping pixels')
        if source['provenance']!=r['provenance'] or r['split']!='train':
            raise ValueError('Source identity differs')
        if sha256(Path(r['image']))!=r['image_sha256'] or sha256(Path(r['label']))!=r['label_sha256']:
            raise ValueError('Source image or label changed')
        text=Path(r['label']).read_text(encoding='utf-8'); validate_yolo(text)
        if r['kind']=='negative' and (text.strip() or source['fire_boxes']!='0'):
            raise ValueError('Negative source has labels')
    return raw


def sheets(folder,rows,prefix,proposal=False):
    tiles=[]; result=[]
    for r in rows:
        im=cv2.imread(r['image']); h,w=im.shape[:2]
        if [w,h]!=r['size']: raise ValueError('Image dimensions changed')
        if proposal:
            for b in r.get('boxes_xyxy',[]): cv2.rectangle(im,tuple(map(round,b[:2])),tuple(map(round,b[2:])),(0,255,0),2)
        ratio=min(640/w,640/h); view=cv2.resize(im,(round(w*ratio),round(h*ratio)))
        tile=np.full((680,640,3),25,np.uint8); rh,rw=view.shape[:2]
        tile[40+(640-rh)//2:40+(640-rh)//2+rh,(640-rw)//2:(640-rw)//2+rw]=view
        caption=f"{r['index']:02d} {r['provenance']} {r.get('status','pending')}"
        cv2.putText(tile,caption,(8,25),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,240,240),1); tiles.append(tile)
    for start in range(0,len(tiles),4):
        page=tiles[start:start+4]; image=np.concatenate([np.concatenate(page[:2],1),np.concatenate(page[2:],1)],0)
        p=folder/f'{prefix}_{start//4+1:02d}.jpg'
        if p.exists() or not cv2.imwrite(str(p),image): raise ValueError('Existing or failed sheet')
        result.append(dict(file=p.name,sha256=sha256(p)))
    return result


def prepare(parent,weights,out):
    if out.exists(): raise ValueError('Existing review')
    checked=verify_parent(parent,PARENT_SHA)
    if sha256(weights)!=MINER_SHA: raise ValueError('Wrong mining checkpoint')
    meta=read(parent/'build_manifest.json'); base=Path(meta['base'])
    raw=csv_rows(base/'manifest.csv')
    present={r['image']:r for r in meta['lineage']}; excluded={r['image'] for r in meta['excluded_lineage']}
    earlier=set()
    for review in [Path(meta['review']),Path(read(Path(meta['parent'])/'build_manifest.json')['review'])]:
        earlier.update(Path(r['image']).name for r in read(review/'review_inputs.json')['images'])
    train=[r for r in raw if r['split']=='train' and r['file'] not in earlier]
    negatives=[r for r in train if r['provenance'] in ('object_pan','nofire_real_indoor') and r['fire_boxes']=='0' and r['file'] in present]
    from ultralytics import YOLO
    model=YOLO(str(weights)); mined=[]
    for n,r in enumerate(negatives,1):
        image=base/'images/train'/r['file']; label=base/'labels/train'/(image.stem+'.txt')
        if sha256(image)!=r['sha256'] or label.read_text(encoding='utf-8').strip(): raise ValueError('Mining source changed')
        pred=model.predict(str(image),imgsz=640,conf=.25,iou=.6,device=0,half=False,rect=True,verbose=False)[0]
        boxes=[dict(confidence=float(b.conf.item()),xyxy=b.xyxy[0].cpu().tolist()) for b in pred.boxes]
        mined.append(dict(file=r['file'],image_sha256=r['sha256'],provenance=r['provenance'],predictions=boxes,max_confidence=max((b['confidence'] for b in boxes),default=0)))
        if n%50==0: print(f'Mined {n}/{len(negatives)} original train negatives',flush=True)
    selected=[]
    for prov in ('object_pan','nofire_real_indoor'):
        ranking=sorted([r for r in mined if r['provenance']==prov],key=lambda r:(-r['max_confidence'],r['image_sha256']))
        if len(ranking)<12: raise ValueError('Insufficient unreviewed negatives')
        byname={r['file']:r for r in train}; selected += [(byname[r['file']],'negative') for r in ranking[:12]]
    selected += [(r,'flame') for r in quantiles([r for r in train if r['provenance']=='kitchen_stove_fire' and r['file'] in excluded],8)]
    selected += [(r,'flame') for r in quantiles([r for r in train if r['provenance']=='ks_flame' and r['file'] in present],8)]
    rows=[]
    for n,(r,kind) in enumerate(selected,1):
        im=base/'images/train'/r['file']; lp=base/'labels/train'/(im.stem+'.txt'); pixels=cv2.imread(str(im))
        if pixels is None: raise ValueError('Unreadable original')
        h,w=pixels.shape[:2]; boxes=[]
        for line in lp.read_text(encoding='utf-8').splitlines():
            _,x,y,bw,bh=map(float,line.split()); boxes.append([(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h])
        rows.append(dict(index=n,kind=kind,image=str(im.resolve()),label=str(lp.resolve()),image_sha256=sha256(im),label_sha256=sha256(lp),size=[w,h],split='train',provenance=r['provenance'],source_group=r['group_id'],license=r['license'],original_boxes_xyxy=boxes))
    check_inventory(meta,rows)
    if sha256(weights)!=MINER_SHA: raise ValueError('Miner weights changed')
    out.mkdir(parents=True); write(out/'mining_predictions.json',dict(role='Training feedback only; prediction is not truth',weights_sha256=MINER_SHA,configuration=dict(imgsz=640,conf=.25,iou=.6,device=0,half=False,rect=True),images=mined))
    write(out/'review_inputs.json',dict(parent=str(parent.resolve()),parent_identity=checked,parent_manifest_sha256=PARENT_SHA,source_manifest_sha256=sha256(base/'manifest.csv'),script_sha256=sha256(Path(__file__)),mining_predictions_sha256=sha256(out/'mining_predictions.json'),role='40 original train contexts, no predictions drawn; not independent acceptance',independent_kitchen=False,images=rows,sheets=sheets(out,rows,'original')))
    print('Prepared 24 mined negatives + 16 flame originals / 10 pages',flush=True)


def validate_decisions(inputs,decisions):
    if [r['index'] for r in decisions]!=list(range(1,41)) or [r['index'] for r in inputs['images']]!=list(range(1,41)): raise ValueError('Incomplete ordered decisions')
    rows=[]
    for source,decision in zip(inputs['images'],decisions):
        status=decision['status']; boxes=decision.get('boxes_xyxy',[])
        if status not in ('flame','negative','held') or (status!='held' and status!=source['kind']): raise ValueError('Unapproved source/status conversion')
        if not isinstance(decision.get('note'),str) or not decision['note'].strip(): raise ValueError('Missing actual review note')
        if (status=='flame')!=bool(boxes): raise ValueError('Wrong box/status contract')
        w,h=source['size']
        for box in boxes:
            if len(box)!=4 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in box) or not (0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h): raise ValueError('Invalid physical geometry')
        rows.append(dict(source,status=status,boxes_xyxy=boxes,note=decision['note']))
    return rows


def render(review):
    inputs=read(review/'review_inputs.json'); meta=read(Path(inputs['parent'])/'build_manifest.json'); check_inventory(meta,inputs['images'])
    rows=validate_decisions(inputs,read(review/'decisions.json'))
    write(review/'proposal_manifest.json',dict(decisions_sha256=sha256(review/'decisions.json'),sheets=sheets(review,rows,'proposal',True)))


def finalize(review,seen):
    if (review/'review_complete.json').exists(): raise ValueError('Already finalized')
    inputs=read(review/'review_inputs.json'); parent=Path(inputs['parent']); verify_parent(parent,PARENT_SHA)
    meta=read(parent/'build_manifest.json'); check_inventory(meta,inputs['images'])
    if inputs['script_sha256']!=sha256(Path(__file__)) or inputs['source_manifest_sha256']!=sha256(Path(meta['base'])/'manifest.csv') or inputs['mining_predictions_sha256']!=sha256(review/'mining_predictions.json'): raise ValueError('Review source/mining changed')
    proposal=read(review/'proposal_manifest.json')
    if proposal['decisions_sha256']!=sha256(review/'decisions.json'): raise ValueError('Decisions changed after rendering')
    expected={r['sha256'] for r in inputs['sheets']+proposal['sheets']}
    if set(seen)!=expected or len(seen)!=len(expected): raise ValueError('Every original/proposal page must actually be viewed')
    for item in inputs['sheets']+proposal['sheets']:
        if sha256(review/item['file'])!=item['sha256']: raise ValueError('Viewed sheet changed')
    rows=validate_decisions(inputs,read(review/'decisions.json'))
    write(review/'review_complete.json',dict(role='Actual full-frame train review; no event/independent truth',review_inputs_sha256=sha256(review/'review_inputs.json'),decisions_sha256=sha256(review/'decisions.json'),proposal_manifest_sha256=sha256(review/'proposal_manifest.json'),seen_sheet_sha256=seen,counts=dict(Counter(r['status'] for r in rows)),images=rows))


def label_text(row):
    w,h=row['size']; return ''.join(f'0 {(b[0]+b[2])/(2*w):.9f} {(b[1]+b[3])/(2*h):.9f} {(b[2]-b[0])/w:.9f} {(b[3]-b[1])/h:.9f}\n' for b in row['boxes_xyxy'])


def planned(parent,review):
    verify_parent(parent,PARENT_SHA); meta=read(parent/'build_manifest.json'); inp=read(review/'review_inputs.json'); complete=read(review/'review_complete.json')
    if Path(inp['parent']).resolve()!=parent.resolve() or complete['review_inputs_sha256']!=sha256(review/'review_inputs.json') or complete['decisions_sha256']!=sha256(review/'decisions.json') or complete['proposal_manifest_sha256']!=sha256(review/'proposal_manifest.json'): raise ValueError('Approval identity changed')
    decisions=validate_decisions(inp,read(review/'decisions.json'))
    if complete['images']!=decisions: raise ValueError('Approval differs from decisions')
    check_inventory(meta,decisions); rows=deepcopy(meta['lineage']); byname={r['image']:r for r in rows}; excluded={r['image']:deepcopy(r) for r in meta['excluded_lineage']}; changes={}
    raw={r['file']:r for r in csv_rows(Path(meta['base'])/'manifest.csv')}
    for r in decisions:
        if r['status']=='held': continue
        name=Path(r['image']).name
        if name not in byname:
            if r['status']!='flame' or name not in excluded: raise ValueError('Unlisted addition')
            item=excluded.pop(name); rows.append(item); byname[name]=item
        item=byname[name]
        if item['sha256']!=r['image_sha256'] or raw[name]['split']!='train': raise ValueError('Wrong approved source')
        if r['status']=='negative' and (parent/'labels/train'/(Path(name).stem+'.txt')).read_text(encoding='utf-8').strip(): raise ValueError('Negative inherited labels differ')
        item.update(weight=8,v19_context_review_status=r['status'],v19_original_label_sha256=r['label_sha256'])
        if r['status']=='flame': item.update(boxes_xyxy=r['boxes_xyxy'],visible_flame_train_revision=True,teacher_preserve=False)
        changes[name]=r
    return meta,rows,list(excluded.values()),changes,complete


def build(parent,review,out):
    if out.exists(): raise ValueError('Existing v19 dataset')
    meta,rows,excluded,changes,complete=planned(parent,review)
    (out/'images/train').mkdir(parents=True); (out/'labels/train').mkdir(parents=True); entries=[]
    for r in rows:
        name=r['image']; change=changes.get(name); image=Path(change['image']) if change else parent/'images/train'/name
        target=out/'images/train'/name; label=out/'labels/train'/(target.stem+'.txt'); shutil.copyfile(image,target)
        if change and change['status']=='flame': label.write_text(label_text(change),encoding='utf-8',newline='\n'); r['label_sha256']=sha256(label)
        else: shutil.copyfile(parent/'labels/train'/label.name,label)
        entries.extend([str(target.resolve())]*r['weight'])
    (out/'train.txt').write_text('\n'.join(entries)+'\n',encoding='utf-8',newline='\n'); base=Path(meta['base'])
    config=dict(path=str(out.resolve()),train=str((out/'train.txt').resolve()),val=str(base/'images/val'),nc=1,names={0:'fire'})
    (out/'data.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8',newline='\n')
    write(out/'build_manifest.json',dict(role='v19_reviewed_train_full_context_extension',parent=str(parent.resolve()),parent_sha256=PARENT_SHA,base=str(base),review=str(review.resolve()),review_sha256=sha256(review/'review_complete.json'),builder_sha256=sha256(Path(__file__)),validation=validation_snapshot(base),independent_kitchen=False,smoke_absence_unlabelled=True,unique_paths=len(rows),train_entries=len(entries),review_counts=complete['counts'],lineage=rows,excluded_lineage=excluded))
    print(json.dumps(verify(out,sha256(out/'build_manifest.json')),indent=2),flush=True)


def verify(folder,expected):
    if sha256(folder/'build_manifest.json')!=expected: raise ValueError('V19 manifest changed')
    meta=read(folder/'build_manifest.json')
    if meta['parent_sha256']!=PARENT_SHA or meta['builder_sha256']!=sha256(Path(__file__)) or meta['independent_kitchen'] or not meta['smoke_absence_unlabelled']: raise ValueError('Wrong dataset scope')
    parent=Path(meta['parent']); review=Path(meta['review']); source,rows,excluded,changes,complete=planned(parent,review)
    if meta['review_sha256']!=sha256(review/'review_complete.json') or meta['validation']!=validation_snapshot(Path(source['base'])) or meta['excluded_lineage']!=excluded: raise ValueError('Review/validation/quarantine changed')
    counter=Counter()
    for r in rows:
        image=folder/'images/train'/r['image']; lp=folder/'labels/train'/(image.stem+'.txt'); change=changes.get(r['image'])
        if change and change['status']=='flame':
            if lp.read_text(encoding='utf-8')!=label_text(change): raise ValueError('Unapproved flame label')
            r['label_sha256']=sha256(lp)
        if sha256(image)!=r['sha256'] or sha256(lp)!=r['label_sha256']: raise ValueError('Image/label changed')
        validate_yolo(lp.read_text(encoding='utf-8')); counter[image.resolve()]=r['weight']
    config=dict(path=str(folder.resolve()),train=str((folder/'train.txt').resolve()),val=str(Path(source['base'])/'images/val'),nc=1,names={0:'fire'})
    if rows!=meta['lineage'] or len(rows)!=meta['unique_paths'] or sum(counter.values())!=meta['train_entries'] or Counter(Path(p).resolve() for p in (folder/'train.txt').read_text(encoding='utf-8').splitlines())!=counter or yaml.safe_load((folder/'data.yaml').read_text(encoding='utf-8'))!=config or meta['review_counts']!=complete['counts']: raise ValueError('Membership/sampling/configuration changed')
    if {p.name for p in (folder/'images/train').iterdir()}!={r['image'] for r in rows} or {p.name for p in (folder/'labels/train').iterdir()}!={Path(r['image']).stem+'.txt' for r in rows}: raise ValueError('Unlisted output')
    return dict(manifest_sha256=expected,images_verified=len(rows),entries_verified=sum(counter.values()),approved_contexts=len(changes),review_counts=complete['counts'],validation_images=len(meta['validation']),independent_kitchen=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('action',choices=['prepare','render','finalize','build','verify']); p.add_argument('--parent',type=Path); p.add_argument('--weights',type=Path); p.add_argument('--review',type=Path); p.add_argument('--out',type=Path); p.add_argument('--expected-sha256'); p.add_argument('--seen-sheet',action='append',default=[]); a=p.parse_args()
    if a.action=='prepare': prepare(a.parent.resolve(),a.weights.resolve(),a.out.resolve())
    elif a.action=='render': render(a.review.resolve())
    elif a.action=='finalize': finalize(a.review.resolve(),a.seen_sheet)
    elif a.action=='build': build(a.parent.resolve(),a.review.resolve(),a.out.resolve())
    else: print(json.dumps(verify(a.out.resolve(),a.expected_sha256),indent=2))
