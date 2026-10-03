"""Localization on the fourteen new training-exposed sources, never reserved queues."""
import argparse
import json
from pathlib import Path
import cv2
from ultralytics import YOLO
from data_integrity import sha256,matches

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='0')
    args=parser.parse_args();out=args.out
    if out.exists(): raise SystemExit(f'Refusing overwrite: {out}')
    meta=json.loads((args.data/'build_manifest.json').read_text(encoding='utf-8'))
    if sha256(args.data/'build_manifest.json')!='15a509075be0d2ff062894cd7de1160699bbdc7b657d8199bfba77cb80bf36fa':raise ValueError('Wrong v20 diagnostic data')
    selected=[r for r in meta['lineage'] if r.get('v20_new_source')]
    if len(selected)!=14 or sum(len(r['boxes_xyxy']) for r in selected)!=4:raise ValueError('Incomplete new-source diagnostic membership')
    model=YOLO(str(args.weights));rows=[]
    for row in selected:
        p=args.data/'images/train'/row['image']
        if sha256(p)!=row['sha256']: raise SystemExit('Image hash mismatch')
        result=model.predict(str(p),imgsz=640,conf=0.25,iou=0.6,device=args.device,half=False,rect=True,verbose=False)[0]
        boxes=[{'confidence':round(float(b.conf.item()),4),'xyxy':[float(v) for v in b.xyxy[0].tolist()]} for b in result.boxes]
        score=matches(row['boxes_xyxy'],[b['xyxy'] for b in boxes])
        rows.append({'image':row['image'],'origin':row['origin'],'source_group':row['source_group'],**score,'predictions':boxes})
        image=cv2.imread(str(p))
        for box in row['boxes_xyxy']:cv2.rectangle(image,tuple(map(round,box[:2])),tuple(map(round,box[2:])),(0,255,0),2)
        for b in boxes:
            a=[int(v) for v in b['xyxy']];cv2.rectangle(image,tuple(a[:2]),tuple(a[2:]),(255,0,0),2)
        out.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(out/row['image']),image)
    by_origin={}
    for kind in sorted({r['origin'] for r in rows}):
        group=[r for r in rows if r['origin']==kind]
        by_origin[kind]={'images':len(group),'images_with_fp':sum(r['fp']>0 for r in group),
                          **{k:sum(r[k] for r in group) for k in ['tp','fp','fn']}}
    result={'weights_sha256':sha256(args.weights),'config':{'imgsz':640,'conf':0.25,'nms_iou':0.6,'match_iou':0.5},
            'role':'Fourteen source-extension train images; development feedback, not independent acceptance','by_origin':by_origin,'frames':rows}
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='frames'},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__': main()
