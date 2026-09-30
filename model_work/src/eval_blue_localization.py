"""Primary-burner localization on five previously inspected diagnostic photos.
Annotations are approximate and not exhaustive. No overall accuracy or FP count
is asserted. These photos are diagnostic even if the model has not trained on them.
"""
import argparse,json
from pathlib import Path
import cv2
from ultralytics import YOLO
from data_integrity import sha256,iou
ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='0')
    args=p.parse_args()
    if args.out.exists(): raise SystemExit('Refusing overwrite')
    rows=json.loads((ROOT/'docs/blue_diagnostic_primary_boxes_20260930.json').read_text(encoding='utf-8'))
    model=YOLO(str(args.weights));result=[];args.out.mkdir(parents=True)
    for row in rows:
        image=ROOT/'model_work/data/commons_blue_review_20260927'/(row['slug']+'.jpg')
        if sha256(image)!=row['sha256']: raise SystemExit('Photo identity mismatch')
        pred=model.predict(str(image),imgsz=640,conf=0.25,iou=0.6,device=args.device,verbose=False)[0]
        boxes=[{'confidence':round(float(b.conf.item()),4),'xyxy':[float(v) for v in b.xyxy[0].tolist()]} for b in pred.boxes]
        overlap=max((iou(row['primary_flame_xyxy'],b['xyxy']) for b in boxes),default=0)
        result.append({'slug':row['slug'],'detected_anywhere':bool(boxes),'primary_best_iou':overlap,
                       'primary_localized_iou50':overlap>=0.5,'predictions':boxes})
        frame=cv2.imread(str(image));gt=row['primary_flame_xyxy']
        cv2.rectangle(frame,tuple(gt[:2]),tuple(gt[2:]),(0,255,0),3)
        for b in boxes:
            c=[int(v) for v in b['xyxy']];cv2.rectangle(frame,tuple(c[:2]),tuple(c[2:]),(255,0,0),3)
        cv2.imwrite(str(args.out/(row['slug']+'.jpg')),frame)
    summary={'weights_sha256':sha256(args.weights),'imgsz':640,'conf':0.25,'nms_iou':0.6,'images':len(result),
             'detected_anywhere':sum(r['detected_anywhere'] for r in result),
             'primary_localized_iou50':sum(r['primary_localized_iou50'] for r in result),
             'role':'selected development diagnostics; approximate primary boxes; not accuracy', 'per_image':result}
    (args.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='per_image'},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__': main()
