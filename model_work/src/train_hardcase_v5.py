"""Bounded reproducible hard-case experiment. Never replaces board or old checkpoints."""
import argparse
import json
import platform
from pathlib import Path
import torch
import ultralytics
from ultralytics import YOLO
from data_integrity import sha256
from dataset_preflight import verify_reviewed_dataset
import yaml
from backbone_freeze_audit import FrozenBackboneAudit

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--start',type=Path,required=True)
    parser.add_argument('--expected-sha256',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--lr0',type=float,default=0.00015)
    parser.add_argument('--warmup-bias-lr',type=float,default=0.1)
    parser.add_argument('--freeze-backbone',action='store_true',help='Freeze the actual checkpoint backbone and audit live parameters/BN buffers')
    args=parser.parse_args();out=args.out.resolve();start=args.start.resolve();data=args.data.resolve()
    if out.exists(): raise SystemExit(f'Refusing overwrite: {out}')
    if not 0 < args.lr0 <= 1 or not 0 <= args.warmup_bias_lr <= 1: raise SystemExit('Invalid learning rate')
    if not torch.cuda.is_available(): raise SystemExit('CUDA unavailable')
    digest=sha256(start)
    if digest!=args.expected_sha256.lower(): raise SystemExit('Starting checkpoint identity mismatch')
    try:
        dataset_check=verify_reviewed_dataset(data.parent,yaml.safe_load(data.read_text(encoding='utf-8')))
    except (ValueError,KeyError,OSError) as error:
        raise SystemExit(f'Dataset preflight failed: {error}') from error
    config=dict(data=str(data),epochs=args.epochs,patience=8,imgsz=640,batch=4,device=0,workers=0,
                seed=20260930,optimizer='AdamW',lr0=args.lr0,warmup_bias_lr=args.warmup_bias_lr,lrf=0.1,weight_decay=0.0005,
                hsv_h=0.015,hsv_s=0.3,hsv_v=0.25,degrees=3.0,translate=0.05,scale=0.2,
                fliplr=0.5,mosaic=0.2,mixup=0.0,close_mosaic=5,
                project=str(out.parent),name=out.name,exist_ok=False,save=True,plots=True,val=True)
    model=YOLO(str(start))
    freeze_audit=None
    if args.freeze_backbone:
        backbone=model.model.yaml.get('backbone')
        if not isinstance(backbone,list) or not 0 < len(backbone) < len(model.model.model):
            raise SystemExit('Cannot identify a nonempty checkpoint backbone and trainable neck/head')
        config['freeze']=len(backbone)
        freeze_audit=FrozenBackboneAudit(model.model,len(backbone),out/'backbone_freeze_audit.json')
        model.add_callback('on_train_start',freeze_audit.on_start)
        model.add_callback('on_train_batch_start',freeze_audit.on_batch_start)
        model.add_callback('on_train_epoch_end',freeze_audit.on_epoch_end)
    identity={'training_script_sha256':sha256(Path(__file__)), 'dataset_config_sha256':sha256(data), 'dataset_verification':dataset_check, 'start_sha256':digest,'start':str(start),'dataset_manifest_sha256':sha256(data.parent/'build_manifest.json'),
              'config':config,'environment':{'python':platform.python_version(),'torch':torch.__version__,
                'ultralytics':ultralytics.__version__,'gpu':torch.cuda.get_device_name(0)},
              'role':'development candidate; all teammate videos now exposed; no board deployment'}
    identity['local_dependency_sha256']={name:sha256(Path(__file__).with_name(name)) for name in ('data_integrity.py','dataset_preflight.py','backbone_freeze_audit.py')}
    if freeze_audit:
        framework_trainer=Path(ultralytics.__file__).parent/'engine/trainer.py'
        identity['backbone_freeze']={**freeze_audit.plan,'audit_script_sha256':sha256(Path(__file__).with_name('backbone_freeze_audit.py')),'framework_trainer_sha256':sha256(framework_trainer)}
    # Run identity is written before training; completed metadata is separate.
    marker=out.parent/(out.name+'_preflight.json')
    if marker.exists(): raise SystemExit('Existing preflight; inspect previous run before restarting')
    marker.write_text(json.dumps(identity,ensure_ascii=False,indent=2),encoding='utf-8')
    model.train(**config)
    if freeze_audit:
        freeze_audit.verify_state(model.trainer,'train_complete')
        identity['backbone_freeze']['completed_audit_sha256']=sha256(out/'backbone_freeze_audit.json')
    for name in ['best.pt','last.pt']: identity[name+'_sha256']=sha256(out/'weights'/name)
    (out/'experiment_meta.json').write_text(json.dumps(identity,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(identity,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__': main()
