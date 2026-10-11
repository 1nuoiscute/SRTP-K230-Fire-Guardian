"""Paired two-class pilot and known-fire-only regressions; no event acceptance."""
import argparse
import csv
import json
from collections import Counter,defaultdict
from pathlib import Path
import cv2
import yaml
from ultralytics import YOLO
from class_detection_metrics import scores_by_known_class
from data_integrity import sha256,iou
from joint_pilot_dataset import verify_dataset


ROOT=Path(__file__).resolve().parents[2]


def load(path):return json.loads(path.read_text(encoding='utf-8'))
def dump(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def input_record(image,label,image_sha,classes):
    if sha256(image)!=image_sha:raise ValueError('Image identity changed')
    im=cv2.imread(str(image))
    if im is None:raise ValueError('Unreadable image')
    h,w=im.shape[:2]; truth={c:[] for c in classes}
    for line in label.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():continue
        cls,x,y,bw,bh=map(float,line.split())
        if cls!=int(cls) or int(cls) not in truth:raise ValueError('Unexpected target class')
        truth[int(cls)].append([(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h])
    return {'image':str(image),'label':str(label),'image_sha256':image_sha,'label_sha256':sha256(label),'truth':truth}


def predict(model,row,device):
    if sha256(Path(row['image']))!=row['image_sha256'] or ('label' in row and sha256(Path(row['label']))!=row['label_sha256']):
        raise ValueError('Diagnostic input changed')
    pred=model.predict(row['image'],imgsz=640,conf=.25,iou=.6,device=device,classes=None,verbose=False)[0]
    return [{'class':int(b.cls.item()),'confidence':float(b.conf.item()),'xyxy':[float(v) for v in b.xyxy[0].tolist()]} for b in pred.boxes]


def aggregate(rows,classes):
    result={}
    for cls in classes:
        counts={key:sum(r['scores'][cls][key] for r in rows) for key in ('tp','fp','fn')}
        denominator=2*counts['tp']+counts['fp']+counts['fn']
        result[cls]={**counts,'f1':2*counts['tp']/denominator if denominator else None}
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pilot-data',type=Path,required=True)
    p.add_argument('--expected-pilot-manifest-sha256',required=True)
    p.add_argument('--legacy-data',type=Path,required=True)
    p.add_argument('--models',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='0')
    a=p.parse_args()
    if a.out.exists():raise SystemExit('Refusing overwrite')
    check=verify_dataset(a.pilot_data,yaml.safe_load((a.pilot_data/'data.yaml').read_text()),a.expected_pilot_manifest_sha256)
    pilot_manifest=load(a.pilot_data/'build_manifest.json'); pilot=[]
    for row in pilot_manifest['images']:
        if row['split']=='val':
            image=a.pilot_data/'images/val'/row['file']; label=a.pilot_data/'labels/val'/(image.stem+'.txt')
            pilot.append(input_record(image,label,row['image_sha256'],(0,1)))
    legacy=[]
    for row in csv.DictReader((a.legacy_data/'manifest.csv').open(encoding='utf-8-sig')):
        if row['split']!='test':continue
        image=a.legacy_data/'images/test'/row['file']; label=a.legacy_data/'labels/test'/(image.stem+'.txt')
        record=input_record(image,label,row['sha256'],(0,)); record['provenance']=row['provenance']
        if len(record['truth'][0])!=int(row['fire_boxes']):raise SystemExit('Legacy box count changed')
        legacy.append(record)
    if len(legacy)!=286 or sum(r['provenance']=='nofire_real_indoor' for r in legacy)!=100:
        raise SystemExit('Legacy diagnostic coverage changed')
    blue=[]
    for row in load(ROOT/'docs/blue_diagnostic_primary_boxes_20260930.json'):
        blue.append({'image':str(ROOT/'model_work/data/commons_blue_review_20260927'/(row['slug']+'.jpg')),
                     'image_sha256':row['sha256'],'truth':{0:[row['primary_flame_xyxy']]}})
    models=load(a.models)
    if len({m['label'] for m in models})!=len(models):raise SystemExit('Duplicate model label')
    for m in models:
        if not m['label'].replace('_','').isalnum() or sha256(Path(m['weights']))!=m['expected_sha256']:
            raise SystemExit('Model identity/label invalid')
    a.out.mkdir(parents=True); dump(a.out/'inputs.json',{'pilot':pilot,'legacy':legacy,'blue':blue})
    result={'role':'exposed two-class development pilot and repeated known-fire regression; no independent kitchen/event acceptance',
            'script_sha256':sha256(Path(__file__)),'input_records_sha256':sha256(a.out/'inputs.json'),
            'model_list_sha256':sha256(a.models),'pilot_dataset':check,'models':{},
            'fixed_configuration':{'imgsz':640,'conf':.25,'nms_iou':.6,'match_iou':.5,'device':a.device},
            'scope':{'pilot_val_images':len(pilot),'pilot_val_classes':{0:'fire',1:'smoke'},'legacy_images':286,'legacy_known_classes':[0],
                'blue_primary_images':len(blue),'unknown_legacy_smoke':'predictions counted but no TP/FP/FN or F1 assigned'}}
    for entry in models:
        label=entry['label']; model=YOLO(entry['weights'])
        names={k:v.lower() for k,v in model.names.items()}
        if names not in ({0:'fire'},{0:'fire',1:'smoke'}):raise SystemExit('Model class mapping unexpected')
        joint_rows=[]; legacy_rows=[]; blue_rows=[]
        images_dir=a.out/(label+'_pilot_overlays');images_dir.mkdir()
        for i,row in enumerate(pilot,1):
            predictions=predict(model,row,a.device); scores=scores_by_known_class(row['truth'],predictions)
            joint_rows.append({'index':i,'scores':scores,'predictions':predictions})
            im=cv2.imread(row['image'])
            for cls,boxes in row['truth'].items():
                for b in boxes:cv2.rectangle(im,tuple(map(int,b[:2])),tuple(map(int,b[2:])),(0,255,0),2)
            for b in predictions:
                cv2.rectangle(im,tuple(map(int,b['xyxy'][:2])),tuple(map(int,b['xyxy'][2:])),(255,0,0) if b['class']==0 else (0,0,255),2)
            cv2.imwrite(str(images_dir/f'{i:02d}.jpg'),im)
        for row in legacy:
            predictions=predict(model,row,a.device); scores=scores_by_known_class(row['truth'],predictions)
            legacy_rows.append({'provenance':row['provenance'],'scores':scores,'unscored_smoke_boxes':sum(p['class']==1 for p in predictions)})
        for row in blue:
            predictions=predict(model,row,a.device)
            overlap=max((iou(row['truth'][0][0],b['xyxy']) for b in predictions if b['class']==0),default=0)
            blue_rows.append({'image_sha256':row['image_sha256'],'primary_best_iou':overlap,'localized_iou50':overlap>=.5})
        grouped=defaultdict(list)
        for row in legacy_rows:grouped[row['provenance']].append(row)
        by_source={source:{'images':len(rows),**aggregate(rows,(0,))[0],
                          'unscored_smoke_boxes':sum(r['unscored_smoke_boxes'] for r in rows)} for source,rows in grouped.items()}
        metrics=model.val(data=str((a.legacy_data/'data.yaml').resolve()),split='test',classes=[0],single_cls=False,
                          imgsz=640,batch=4,device=a.device,workers=0,plots=False,
                          project=str(a.out),name=label+'_legacy_map',exist_ok=False)
        if list(metrics.box.ap_class_index)!=[0]:raise SystemExit('Legacy mAP includes an unscored class')
        model_result={'weights_sha256':entry['expected_sha256'],'names':model.names,
                      'pilot_val_fixed':aggregate(joint_rows,(0,1)),
                      'legacy_map':{'precision':float(metrics.box.mp),'recall':float(metrics.box.mr),'map50':float(metrics.box.map50),'map50_95':float(metrics.box.map)},
                      'legacy_fixed_by_source':by_source,'blue_primary_localized_iou50':sum(r['localized_iou50'] for r in blue_rows)}
        dump(a.out/(label+'_local_results.json'),{'summary':model_result,'pilot':joint_rows,'legacy':legacy_rows,'blue':blue_rows})
        result['models'][label]=model_result; dump(a.out/'comparison_partial.json',result)
        print(json.dumps({'model':label,**model_result},indent=2),flush=True)
    dump(a.out/'comparison_complete.json',result)


if __name__=='__main__':main()
