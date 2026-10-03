"""Fixed small-data schedule package; fresh start, unchanged pilot and screening."""
import argparse
import json
import platform
from pathlib import Path
import torch
import ultralytics
import yaml
from ultralytics import YOLO
from backbone_freeze_audit import FrozenBackboneAudit
from data_integrity import sha256
from joint_pilot_dataset import verify_dataset
from optimizer_cadence_audit import OptimizerCadenceAudit


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--expected-manifest-sha256',required=True)
    p.add_argument('--start',type=Path,required=True)
    p.add_argument('--expected-start-sha256',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); out=a.out.resolve(); marker=out.parent/(out.name+'_preflight.json')
    if out.exists() or marker.exists(): raise SystemExit('Existing run/preflight; no restart')
    if not torch.cuda.is_available(): raise SystemExit('CUDA unavailable')
    if sha256(a.start)!=a.expected_start_sha256: raise SystemExit('Starting checkpoint changed')
    config=yaml.safe_load(a.data.read_text(encoding='utf-8'))
    verified=verify_dataset(a.data.parent,config,a.expected_manifest_sha256)
    model=YOLO(str(a.start)); count=len(model.model.yaml['backbone'])
    if model.names!={0:'fire',1:'smoke'}: raise SystemExit('Starting class names changed')
    initial={k:v.detach().cpu().clone() for k,v in model.model.state_dict().items()}
    freeze=FrozenBackboneAudit(model.model,count,out/'backbone_freeze_audit.json')
    cadence=OptimizerCadenceAudit(out/'optimizer_cadence_audit.json')
    def initial_check(trainer):
        state=trainer.model.state_dict()
        if state.keys()!=initial.keys() or any(not torch.equal(value.detach().cpu(),initial[key]) for key,value in state.items()):
            raise RuntimeError('Framework starting state differs from verified start')
        freeze.on_start(trainer)
        if trainer.accumulate!=1: raise RuntimeError('Expected one update per batch without warmup')
        cadence.attach(trainer.optimizer)
    model.add_callback('on_train_start',initial_check)
    model.add_callback('on_train_batch_start',freeze.on_batch_start)
    model.add_callback('on_train_batch_end',cadence.on_batch_end)
    model.add_callback('on_train_epoch_end',freeze.on_epoch_end)
    model.add_callback('on_train_epoch_end',cadence.on_epoch_end)
    schedule=dict(data=str(a.data.resolve()),epochs=40,patience=0,imgsz=640,batch=4,nbs=4,device=0,workers=0,
                  seed=20261002,optimizer='AdamW',lr0=.0005,lrf=.1,weight_decay=.0005,
                  warmup_epochs=0,warmup_bias_lr=.001,freeze=count,hsv_h=.015,hsv_s=.3,hsv_v=.25,
                  degrees=3,translate=.05,scale=.2,fliplr=.5,mosaic=.2,mixup=0,close_mosaic=5,
                  project=str(out.parent),name=out.name,exist_ok=False,save=True,plots=True,val=True)
    identity={'role':'small-data schedule package v2; exposed development only, no causal attribution to one knob',
              'script_sha256':sha256(Path(__file__)),'data_config_sha256':sha256(a.data),
              'dataset':verified,'start_sha256':sha256(a.start),'config':schedule,
              'environment':{'python':platform.python_version(),'torch':torch.__version__,'ultralytics':ultralytics.__version__,
                             'gpu':torch.cuda.get_device_name(0)},'backbone_freeze':freeze.plan,
              'local_dependency_sha256':{name:sha256(Path(__file__).with_name(name)) for name in
                  ('joint_pilot_dataset.py','backbone_freeze_audit.py','data_integrity.py','optimizer_cadence_audit.py')}}
    marker.write_text(json.dumps(identity,indent=2),encoding='utf-8')
    try:
        model.train(**schedule)
        freeze.verify_state(model.trainer,'train_complete')
    finally:
        cadence.close()
    if not cadence.steps: raise RuntimeError('No observed optimizer updates')
    for name in ('best.pt','last.pt'):identity[name+'_sha256']=sha256(out/'weights'/name)
    identity['completed_backbone_audit_sha256']=sha256(out/'backbone_freeze_audit.json')
    identity['completed_optimizer_cadence_sha256']=sha256(out/'optimizer_cadence_audit.json')
    identity['actual_optimizer_steps']=len(cadence.steps); identity['actual_batches']=cadence.batches
    (out/'experiment_meta.json').write_text(json.dumps(identity,indent=2),encoding='utf-8')
    print(json.dumps(identity,indent=2))


if __name__=='__main__':main()
