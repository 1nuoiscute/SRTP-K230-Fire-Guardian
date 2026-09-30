"""Build development-only v5 from reviewed flame frames and verified negative crops.
All teammate videos are development-exposed. Derived crops/weighted entries are
not independent scenes. Refuses overwrite; never mutates input labels/media.
"""
import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from data_integrity import sha256, iou, validate_yolo

ROOT=Path(__file__).resolve().parents[2]

def yolo(box,w,h):
    x1,y1,x2,y2=box
    text=f'0 {(x1+x2)/2/w:.8f} {(y1+y2)/2/h:.8f} {(x2-x1)/w:.8f} {(y2-y1)/h:.8f}\n'
    validate_yolo(text)
    return text

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); out=args.out.resolve();base=args.base.resolve()
    if out.exists(): raise SystemExit(f'Refusing overwrite: {out}')
    rows=list(csv.DictReader((base/'manifest.csv').open(encoding='utf-8-sig')))
    selected=[r for r in rows if r['split']=='train']
    excluded={r['sha256'] for r in rows if r['split']!='train'}
    manual=ROOT/'model_work/out/teammate_videos_20260928/manual_frames'
    truth=json.loads((manual/'flame_gt_review.json').read_text(encoding='utf-8'))
    metadata={r['image']:r for r in json.loads((manual/'manifest.json').read_text(encoding='utf-8'))}
    if len(truth)!=10: raise SystemExit('Reviewed ten-frame set changed')
    (out/'images/train').mkdir(parents=True)
    (out/'labels/train').mkdir(parents=True)
    train=[]; lineage=[]; reviewed=[]; seen=set(); counts=Counter()
    for row in selected:
        src=base/'images/train'/row['file']; label=base/'labels/train'/(src.stem+'.txt')
        digest=sha256(src)
        if digest!=row['sha256'] or digest in excluded: raise SystemExit('Base hash mismatch/overlap')
        if digest in seen: continue
        seen.add(digest); validate_yolo(label.read_text(encoding='utf-8'))
        train.append(src.as_posix());counts[row['provenance']]+=1
        lineage.append({'image':src.name,'sha256':digest,'origin':'legacy_train','source_group':row['group_id'],'weight':1})
    def add(stem,frame,boxes,origin,group,source_hash,weight):
        h,w=frame.shape[:2]; p=out/'images/train'/(stem+'.jpg'); label=out/'labels/train'/(stem+'.txt')
        if not cv2.imwrite(str(p),frame): raise RuntimeError('Cannot write image')
        text=''.join(yolo(b,w,h) for b in boxes);label.write_text(text,encoding='utf-8')
        digest=sha256(p)
        if digest in excluded: raise SystemExit('Derived image overlaps legacy diagnostic split')
        train.extend([p.as_posix()]*weight)
        lineage.append({'image':p.name,'sha256':digest,'origin':origin,'source_group':group,
                        'parent_sha256':source_hash,'boxes_xyxy':boxes,'weight':weight})
        thumb=frame.copy()
        for b in boxes: cv2.rectangle(thumb,tuple(b[:2]),tuple(b[2:]),(0,255,0),3)
        thumb=cv2.resize(thumb,(180,320)); reviewed.append((stem,thumb))
    for row in truth:
        src=manual/row['image']; frame=cv2.imread(str(src));gt=row['flame_xyxy'];meta=metadata[row['image']]
        if frame is None: raise SystemExit(f'Missing {src}')
        digest=sha256(src); stem=src.stem
        add('teammate_'+stem,frame,[gt],'reviewed_flame',meta['source'],digest,12)
        # Portrait cooking frames except the open-ring clip: the bottom third
        # lies below every reviewed visible flame, and contains the blue sticker.
        if not stem.startswith('219bd59'):
            h,w=frame.shape[:2];crop_box=[0,980,w,h]
            if gt[3]>=crop_box[1]: raise SystemExit('Negative crop intersects reviewed flame')
            crop=frame[crop_box[1]:h,:].copy()
            add('negative_bottom_'+stem,crop,[],'sticker_hard_negative_crop',meta['source'],digest,8)
            lineage[-1]['crop_xyxy']=crop_box
            # Tight wrong-object crop plus context, only when spatially disjoint.
            for i,pred in enumerate(meta['predictions']):
                b=pred['xyxy']
                if iou(gt,b)>0.05 or b[1]<980: continue
                c=[max(0,int(b[0])-35),max(980,int(b[1])-35),min(w,int(b[2])+35),min(h,int(b[3])+35)]
                if c[0]>=c[2] or c[1]>=c[3]: continue
                add(f'negative_sticker_{stem}_{i}',frame[c[1]:c[3],c[0]:c[2]],[],
                    'sticker_hard_negative_crop',meta['source'],digest,8)
                lineage[-1]['crop_xyxy']=c
    # Preserve the reviewed 24 first-kitchen frames, deduplicating repeated paths.
    old=ROOT/'model_work/data/user_video_adaptation_20260927/build_manifest.json'
    oldmeta=json.loads(old.read_text(encoding='utf-8'))
    for row in oldmeta['samples']:
        p=Path(row['source']);f=cv2.imread(str(p))
        if f is None or sha256(p)!=row['sha256']: raise SystemExit('Old reviewed frame identity changed')
        add('old_'+p.stem,f,row['boxes_xyxy'],'reviewed_old_video',row['video'],row['sha256'],3)
    (out/'train.txt').write_text('\n'.join(train)+'\n',encoding='utf-8')
    (out/'data.yaml').write_text(f'path: {out.as_posix()}\ntrain: {(out/"train.txt").as_posix()}\nval: {(base/"images/val").as_posix()}\nnc: 1\nnames:\n  0: fire\n',encoding='utf-8')
    cols=6; sheet=np.full((((len(reviewed)+cols-1)//cols)*350,cols*200,3),25,np.uint8)
    for index,(stem,frame) in enumerate(reviewed):
        x=index%cols*200;y=index//cols*350;sheet[y:y+320,x:x+180]=frame
        cv2.putText(sheet,stem[:24],(x+2,y+338),cv2.FONT_HERSHEY_SIMPLEX,.36,(255,255,255),1)
    cv2.imwrite(str(out/'label_review_sheet.jpg'),sheet)
    result={'role':'development_only','base':str(base),'legacy_train':dict(counts),'legacy_unique_images':len(seen),
            'train_entries':len(train),'unique_paths':len(set(train)),'derived_images':len(reviewed),
            'review_required':'Derived negatives must be visually checked before training.',
            'limitations':['reviewed labels are approximate rectangles','all nine videos are development-exposed',
                          'crop weighting is not independent sample growth','legacy val is a checkpoint selector, not a fresh test'],
            'lineage':lineage}
    (out/'build_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='lineage'},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
