"""Bounded reproducible v5 experiment. Never replaces board or old checkpoints."""
import argparse
import json
import platform
from pathlib import Path
import torch
import ultralytics
from ultralytics import YOLO
from data_integrity import sha256

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--start',type=Path,required=True)
    parser.add_argument('--expected-sha256',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=20)
    args=parser.parse_args();out=args.out.resolve();start=args.start.resolve();data=args.data.resolve()
    if out.exists(): raise SystemExit(f'Refusing overwrite: {out}')
    if not torch.cuda.is_available(): raise SystemExit('CUDA unavailable')
    digest=sha256(start)
    if digest!=args.expected_sha256.lower(): raise SystemExit('Starting checkpoint identity mismatch')
    reviewed=json.loads((data.parent/'review_approval.json').read_text(encoding='utf-8'))
    if not reviewed.get('approved') or reviewed['manifest_sha256']!=sha256(data.parent/'build_manifest.json'):
        raise SystemExit('Dataset has not passed visual review')
    config=dict(data=str(data),epochs=args.epochs,patience=8,imgsz=640,batch=4,device=0,workers=0,
                seed=20260930,optimizer='AdamW',lr0=0.00015,lrf=0.1,weight_decay=0.0005,
                hsv_h=0.015,hsv_s=0.3,hsv_v=0.25,degrees=3.0,translate=0.05,scale=0.2,
                fliplr=0.5,mosaic=0.2,mixup=0.0,close_mosaic=5,
                project=str(out.parent),name=out.name,exist_ok=False,save=True,plots=True,val=True)
    identity={'start_sha256':digest,'start':str(start),'dataset_manifest_sha256':sha256(data.parent/'build_manifest.json'),
              'config':config,'environment':{'python':platform.python_version(),'torch':torch.__version__,
                'ultralytics':ultralytics.__version__,'gpu':torch.cuda.get_device_name(0)},
              'role':'development candidate; all teammate videos now exposed; no board deployment'}
    # Run identity is written before training; completed metadata is separate.
    marker=out.parent/(out.name+'_preflight.json')
    if marker.exists(): raise SystemExit('Existing preflight; inspect previous run before restarting')
    marker.write_text(json.dumps(identity,ensure_ascii=False,indent=2),encoding='utf-8')
    YOLO(str(start)).train(**config)
    for name in ['best.pt','last.pt']: identity[name+'_sha256']=sha256(out/'weights'/name)
    (out/'experiment_meta.json').write_text(json.dumps(identity,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(identity,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__': main()
