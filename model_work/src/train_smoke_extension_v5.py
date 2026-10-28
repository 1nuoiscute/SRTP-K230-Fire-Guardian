"""Fixed shared-feature smoke head training on an extension manifest, with source state and EMA checks.

The manifest's train entries extend the v4 train set; the nine original val
members are kept exactly as-is. New samples go to train only. Only visible smoke
appearance is trained; unknown cooking plumes are not treated as negatives and
no fire-event truth is claimed.
"""
import argparse
import json
import platform
import shutil
from pathlib import Path
import torch
import ultralytics
from ultralytics import YOLO
from backbone_freeze_audit import FrozenBackboneAudit
from data_integrity import sha256
from optimizer_cadence_audit import OptimizerCadenceAudit
import yaml


def verify_extension(data, manifest_path):
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    folder=data.resolve().parent
    expected=dict(path=str(folder),train='images/train',val='images/val',nc=1,names={0:'smoke'})
    if yaml.safe_load(data.read_text(encoding='utf-8'))!=expected:
        raise ValueError('Data config redirects or changes smoke task')
    parent=Path(manifest['parent'])
    if sha256(parent/'build_manifest.json')!=manifest['parent_manifest_sha256']:
        raise ValueError('Parent manifest changed')
    if sha256(Path(manifest['source_review']))!=manifest['source_review_sha256']:
        raise ValueError('Extension approval changed')
    approved=json.loads(Path(manifest['source_review']).read_text(encoding='utf-8'))
    additions={r['sha256']:r for r in approved['images'] if r['approved_for_training']}
    old=json.loads((parent/'build_manifest.json').read_text(encoding='utf-8'))['images']
    inherited={(r['split'],r['file'],r['image_sha256'],r['label_sha256']) for r in old}
    actual_inherited=set();actual_additions=set();groups={};seen=set();paths=set();labels=set()
    for row in manifest['images']:
        image,label=Path(row['file']),Path(row['label'])
        split=row['split']; digest=row['image_sha256']
        if split not in ('train','val') or row['review_status']!='approved':raise ValueError('Unapproved split/member')
        if image!=folder/'images'/split/image.name or label!=folder/'labels'/split/(image.stem+'.txt'):
            raise ValueError('Member redirects outside dataset')
        if digest in seen or image in paths or label in labels:raise ValueError('Duplicate member')
        seen.add(digest);paths.add(image);labels.add(label)
        if sha256(image)!=digest or sha256(label)!=row['label_sha256']:raise ValueError('Member bytes changed')
        if groups.setdefault(row['group'],split)!=split:raise ValueError('Group crosses splits')
        if row['origin']=='v4_parent':actual_inherited.add((split,image.name,digest,row['label_sha256']))
        elif row['origin']=='extension':
            if split!='train' or digest not in additions:raise ValueError('Unapproved extension or new val')
            actual_additions.add(digest)
        else:raise ValueError('Unknown origin')
        for line in label.read_text(encoding='utf-8').splitlines():
            v=list(map(float,line.split()))
            if len(v)!=5 or v[0]!=0 or not all(0<=x<=1 for x in v[1:]) or min(v[3:])<=0:
                raise ValueError('Invalid smoke label')
    if actual_inherited!=inherited or len([r for r in old if r['split']=='val'])!=9:
        raise ValueError('Original train/val members changed')
    if actual_additions!=additions.keys():raise ValueError('Approved extension membership changed')
    if paths!={p for s in ('train','val') for p in (folder/'images'/s).iterdir() if p.is_file()}:
        raise ValueError('Unlisted image')
    if labels!={p for s in ('train','val') for p in (folder/'labels'/s).iterdir() if p.suffix=='.txt'}:
        raise ValueError('Unlisted label')
    return manifest


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True); p.add_argument('--start',type=Path,required=True)
    p.add_argument('--expected-start-sha256',required=True); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); out=a.out.resolve(); marker=out.parent/(out.name+'_preflight.json')
    if out.exists() or marker.exists(): raise SystemExit('Existing run/preflight; no restart')
    if not torch.cuda.is_available() or sha256(a.start)!=a.expected_start_sha256: raise SystemExit('CUDA/start identity unavailable')
    manifest=verify_extension(a.data,a.manifest)
    group_split={}; image_count=0
    for row in manifest['images']:
        if row['split'] not in ('train','val'): raise SystemExit('Non train/val split: '+row['file'])
        if sha256(Path(row['file']))!=row['image_sha256']: raise SystemExit('Image identity changed: '+row['file'])
        if sha256(Path(row['label']))!=row['label_sha256']: raise SystemExit('Label identity changed: '+row['file'])
        if group_split.setdefault(row['group'],row['split'])!=row['split']:
            raise SystemExit('Group crosses splits: '+row['group'])
        image_count+=1
    dataset={'manifest_sha256':sha256(a.manifest),'image_count':image_count,
             'manifest_role':'custom extension manifest; v4 val members unchanged, new samples train-only'}
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
    def capture_sources(trainer):
        if Path(trainer.save_dir).resolve()!=out:raise ValueError('Trainer changed output directory')
        snapshot=out/'source_snapshot';snapshot.mkdir()
        for name in [Path(__file__).name,'backbone_freeze_audit.py','optimizer_cadence_audit.py','data_integrity.py']:
            shutil.copyfile(Path(__file__).with_name(name),snapshot/name)
    model.add_callback('on_train_start',capture_sources)
    schedule=dict(data=str(a.data.resolve()),epochs=40,patience=0,imgsz=640,batch=4,nbs=4,device=0,workers=0,
                  seed=20261002,optimizer='AdamW',lr0=.0005,lrf=.1,weight_decay=.0005,
                  warmup_epochs=0,warmup_bias_lr=.001,freeze=blocks,hsv_h=.015,hsv_s=.3,hsv_v=.25,
                  degrees=3,translate=.05,scale=.2,fliplr=.5,mosaic=.2,mixup=0,close_mosaic=5,
                  project=str(out.parent),name=out.name,exist_ok=False,save=True,plots=True,val=True)
    identity={'role':'Smoke head on fixed original features with extension train samples; exposed development only',
              'script_sha256':sha256(Path(__file__)),'data_config_sha256':sha256(a.data),
              'manifest_sha256':sha256(a.manifest),'dataset':dataset,
              'start_sha256':sha256(a.start),'config':schedule,'shared_feature_plan':freeze.plan,
              'labels_note':'Only visible smoke appearance; unknown cooking plumes not assigned negatives; no fire-event truth claimed.',
              'environment':{'python':platform.python_version(),'torch':torch.__version__,'ultralytics':ultralytics.__version__,
                             'gpu':torch.cuda.get_device_name(0)},'local_dependency_sha256':{name:sha256(Path(__file__).with_name(name)) for name in
                  ('backbone_freeze_audit.py','optimizer_cadence_audit.py','data_integrity.py')}}
    marker.write_text(json.dumps(identity,indent=2),encoding='utf-8')
    try:
        model.train(**schedule); freeze.verify_state(model.trainer,'train_complete')
    finally: cadence.close()
    for name in ('best.pt','last.pt'): identity[name+'_sha256']=sha256(out/'weights'/name)
    for name in ('best.pt','last.pt'):
        state=YOLO(str(out/'weights'/name)).model.state_dict()
        if any(not torch.equal(state[k].detach().cpu(),v) for k,v in freeze.reference.items()):
            raise RuntimeError('Saved shared features changed')
    identity['saved_shared_features_equal']=True
    identity['completed_shared_audit_sha256']=sha256(out/'shared_feature_audit.json')
    identity['completed_shared_ema_sha256']=sha256(out/'shared_ema_audit.json')
    identity['completed_optimizer_cadence_sha256']=sha256(out/'optimizer_cadence_audit.json')
    identity['actual_optimizer_steps']=len(cadence.steps); identity['actual_batches']=cadence.batches
    (out/'experiment_meta.json').write_text(json.dumps(identity,indent=2),encoding='utf-8')
    print(json.dumps(identity,indent=2))


if __name__=='__main__':main()
