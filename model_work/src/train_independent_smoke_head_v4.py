"""Fixed shared-feature smoke head training, with source state and EMA checks."""
import argparse
import json
import platform
from pathlib import Path
import torch
import ultralytics
from ultralytics import YOLO
from backbone_freeze_audit import FrozenBackboneAudit
from data_integrity import sha256
from smoke_head_dataset import verify_smoke_dataset
from optimizer_cadence_audit import OptimizerCadenceAudit


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True)
    p.add_argument('--expected-manifest-sha256',required=True); p.add_argument('--start',type=Path,required=True)
    p.add_argument('--expected-start-sha256',required=True); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); out=a.out.resolve(); marker=out.parent/(out.name+'_preflight.json')
    if out.exists() or marker.exists(): raise SystemExit('Existing run/preflight; no restart')
    if not torch.cuda.is_available() or sha256(a.start)!=a.expected_start_sha256: raise SystemExit('CUDA/start identity unavailable')
    verified=verify_smoke_dataset(a.data.parent,a.expected_manifest_sha256)
    model=YOLO(str(a.start))
    if model.names!={0:'smoke'}: raise SystemExit('Single-class smoke names changed')
    initial={k:v.detach().cpu().clone() for k,v in model.model.state_dict().items()}
    blocks=len(model.model.model)-1
    freeze=FrozenBackboneAudit(model.model,blocks,out/'shared_feature_audit.json')
    cadence=OptimizerCadenceAudit(out/'optimizer_cadence_audit.json'); ema_checks=[]
    def initial_check(trainer):
        state=trainer.model.state_dict()
        if state.keys()!=initial.keys() or any(not torch.equal(value.detach().cpu(),initial[key]) for key,value in state.items()):
            raise RuntimeError('Actual initial state differs')
        freeze.on_start(trainer)
        if trainer.accumulate!=1: raise RuntimeError('Unexpected accumulation')
        cadence.attach(trainer.optimizer)
    def epoch_end(trainer):
        freeze.on_epoch_end(trainer)
        state=trainer.ema.ema.state_dict()
        with torch.no_grad():
            for key,value in freeze.reference.items():
                state[key].copy_(value.to(device=state[key].device,dtype=state[key].dtype))
                if not torch.equal(state[key].detach().cpu(),value): raise RuntimeError('EMA shared feature differs')
        ema_checks.append({'epoch':trainer.epoch+1,'shared_features_equal':True})
        (out/'shared_ema_audit.json').write_text(json.dumps(ema_checks,indent=2)+'\n',encoding='utf-8')
        cadence.on_epoch_end(trainer)
    model.add_callback('on_train_start',initial_check)
    model.add_callback('on_train_batch_start',freeze.on_batch_start)
    model.add_callback('on_train_batch_end',cadence.on_batch_end)
    model.add_callback('on_train_epoch_end',epoch_end)
    schedule=dict(data=str(a.data.resolve()),epochs=40,patience=0,imgsz=640,batch=4,nbs=4,device=0,workers=0,
                  seed=20261002,optimizer='AdamW',lr0=.0005,lrf=.1,weight_decay=.0005,
                  warmup_epochs=0,warmup_bias_lr=.001,freeze=blocks,hsv_h=.015,hsv_s=.3,hsv_v=.25,
                  degrees=3,translate=.05,scale=.2,fliplr=.5,mosaic=.2,mixup=0,close_mosaic=5,
                  project=str(out.parent),name=out.name,exist_ok=False,save=True,plots=True,val=True)
    identity={'role':'Independent smoke head on fixed original v3 features; exposed development only',
              'script_sha256':sha256(Path(__file__)),'data_config_sha256':sha256(a.data),
              'dataset':verified,'start_sha256':sha256(a.start),'config':schedule,'shared_feature_plan':freeze.plan,
              'environment':{'python':platform.python_version(),'torch':torch.__version__,'ultralytics':ultralytics.__version__,
                             'gpu':torch.cuda.get_device_name(0)},'local_dependency_sha256':{name:sha256(Path(__file__).with_name(name)) for name in
                  ('smoke_head_dataset.py','joint_pilot_dataset.py','backbone_freeze_audit.py','optimizer_cadence_audit.py','data_integrity.py')}}
    marker.write_text(json.dumps(identity,indent=2),encoding='utf-8')
    try:
        model.train(**schedule); freeze.verify_state(model.trainer,'train_complete')
    finally: cadence.close()
    for name in ('best.pt','last.pt'): identity[name+'_sha256']=sha256(out/'weights'/name)
    identity['completed_shared_audit_sha256']=sha256(out/'shared_feature_audit.json')
    identity['completed_shared_ema_sha256']=sha256(out/'shared_ema_audit.json')
    identity['completed_optimizer_cadence_sha256']=sha256(out/'optimizer_cadence_audit.json')
    identity['actual_optimizer_steps']=len(cadence.steps); identity['actual_batches']=cadence.batches
    (out/'experiment_meta.json').write_text(json.dumps(identity,indent=2),encoding='utf-8')
    print(json.dumps(identity,indent=2))


if __name__=='__main__':main()
