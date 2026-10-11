"""Prepare identity-bound two-class start and test actual raw/fire-NMS invariance."""
import argparse
import copy
import json
import platform
from datetime import datetime,timezone
from pathlib import Path
import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.nn.tasks import DetectionModel
from data_integrity import sha256
from joint_head_init import transfer_fire_and_add_smoke


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--start',type=Path,required=True)
    p.add_argument('--expected-sha256',required=True)
    p.add_argument('--review',type=Path,required=True)
    p.add_argument('--expected-review-sha256',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists(): raise SystemExit('Refusing overwrite')
    if sha256(a.start)!=a.expected_sha256 or sha256(a.review)!=a.expected_review_sha256:
        raise SystemExit('Source checkpoint/review identity changed')
    rows=[r for r in json.loads(a.review.read_text(encoding='utf-8'))['images'] if r['approved_for_training']]
    rows=sorted(rows,key=lambda r:r['sha256'])[:8]
    if len(rows)!=8: raise SystemExit('Need eight reviewed comparison images')
    torch.set_num_threads(2); torch.manual_seed(20261002)
    old=YOLO(str(a.start)); source=old.model.float().eval()
    if source.names.get(0,'').lower()!='fire': raise SystemExit('Starting class is not verified fire')
    target=DetectionModel(copy.deepcopy(source.yaml),ch=3,nc=2,verbose=False).float().eval()
    audit=transfer_fire_and_add_smoke(source,target)
    target.names={0:'fire',1:'smoke'}; target.args=copy.deepcopy(source.args)
    raw=[]
    with torch.inference_mode():
        for row in rows:
            if sha256(Path(row['image']))!=row['sha256']: raise SystemExit('Sample changed')
            image=cv2.imread(row['image'])
            if image is None: raise SystemExit('Sample decode failed')
            square=LetterBox(new_shape=(640,640),auto=False,stride=32)(image=image)
            tensor=torch.from_numpy(np.ascontiguousarray(square[:,:,::-1].transpose(2,0,1))).float().unsqueeze(0)/255
            before=source(tensor)[0]; after=target(tensor)[0]
            if before.shape[1]!=5 or after.shape[1]!=6: raise SystemExit('Unexpected legacy output layout')
            difference=float((before-after[:,:5]).abs().max())
            torch.testing.assert_close(before,after[:,:5],atol=1e-5,rtol=1e-6)
            if not bool(torch.all(after[:,5]<.01)): raise SystemExit('Smoke start not low')
            raw.append({'image_sha256':row['sha256'],'raw_fire_and_geometry_max_abs':difference,
                        'smoke_max':float(after[:,5].max())})
    a.out.mkdir(parents=True); checkpoint=a.out/'start.pt'
    # Match Ultralytics checkpoint half storage, then verify the saved/reloaded wrapper.
    torch.save({'model':copy.deepcopy(target).half(),'epoch':-1,'optimizer':None,
                'train_args':copy.deepcopy(old.ckpt.get('train_args',{})),
                'date':datetime.now(timezone.utc).isoformat(),'version':ultralytics.__version__},checkpoint)
    new=YOLO(str(checkpoint)); wrappers=[]
    for row in rows:
        kwargs=dict(imgsz=640,conf=.25,iou=.6,device='cpu',rect=False,verbose=False)
        baseline=old.predict(row['image'],**kwargs)[0].boxes.data.cpu().numpy()
        candidate=new.predict(row['image'],**kwargs)[0].boxes.data.cpu().numpy()
        if np.any(candidate[:,5]!=0): raise SystemExit('Untrained smoke crosses comparison threshold')
        if baseline.shape!=candidate.shape: raise SystemExit('Reloaded NMS counts differ')
        np.testing.assert_allclose(baseline,candidate,atol=1e-3,rtol=1e-6)
        wrappers.append({'image_sha256':row['sha256'],'fire_boxes':len(baseline),
                         'max_abs':float(np.abs(baseline-candidate).max()) if len(baseline) else 0})
    if sha256(a.start)!=a.expected_sha256: raise SystemExit('Original checkpoint changed')
    record={'role':'untrained two-class development start; not smoke capability or model promotion',
            'source_sha256':a.expected_sha256,'start_sha256':sha256(checkpoint),
            'source_review_sha256':sha256(a.review),'script_sha256':sha256(Path(__file__)),
            'transfer_script_sha256':sha256(Path(__file__).with_name('joint_head_init.py')),
            'transfer':audit,'raw_square_cpu_comparison':raw,'saved_reload_wrapper_comparison':wrappers,
            'comparison_configuration':{'images':8,'imgsz':640,'rect':False,'device':'cpu','conf':.25,'nms_iou':.6,
                 'raw_atol':1e-5,'raw_rtol':1e-6,'wrapper_atol':1e-3,'wrapper_rtol':1e-6},
            'environment':{'python':platform.python_version(),'torch':torch.__version__,'ultralytics':ultralytics.__version__},
            'limitations':['Eight exposed images, not exhaustive numerical/independent performance acceptance.',
                'Smoke output is deliberately untrained and low, not a trained negative classifier.',
                'Training may change fire behaviour; complete candidate diagnostics still required.']}
    (a.out/'initialization.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps({'start_sha256':record['start_sha256'],'transfer':audit,
                      'raw_max':max(r['raw_fire_and_geometry_max_abs'] for r in raw),'wrapper_max':max(r['max_abs'] for r in wrappers)},indent=2))


if __name__=='__main__':main()
