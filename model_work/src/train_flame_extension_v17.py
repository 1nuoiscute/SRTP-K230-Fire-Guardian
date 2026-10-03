"""Bounded full-network v17 fine-tuning from selected v16 best on audited extended labels."""
import argparse
from copy import deepcopy
from pathlib import Path
import platform
from types import SimpleNamespace
import torch
import ultralytics
from ultralytics import YOLO
from backbone_freeze_audit import state_digest
from data_integrity import sha256
from extend_flame_review_v17 import read, write, verify
from train_scratch_semantic_v16 import FullNetworkAudit, verify_trainer_reconstruction, schedule as prior_schedule

START_SHA='e770cf68d12cd43b4494a1439dc9c65f14d1d5fc570b2e2e379cc084384418fd'
PRIMARY=['best.pt','last.pt']
DEPENDENCIES=['extend_flame_review_v17.py','train_scratch_semantic_v16.py','backbone_freeze_audit.py','optimizer_cadence_audit.py','data_integrity.py']


def schedule(data,out,start):
    config=prior_schedule(data,out,start)
    config.update(lr0=.00005,seed=20261006)
    return config


class ExtensionAudit(FullNetworkAudit):
    def on_start(self,trainer):
        actual=trainer.model.state_dict()
        if actual.keys()!=self.initial.keys() or any(not torch.equal(v.detach().cpu(),self.initial[k]) for k,v in actual.items()):
            raise RuntimeError('Actual v17 start differs from selected v16 best')
        frozen=[n for n,p in trainer.model.named_parameters() if not p.requires_grad]
        if frozen!=[f'model.{self.blocks}.dfl.conv.weight'] or trainer.accumulate!=1:
            raise RuntimeError('Unexpected frozen scope/accumulation')
        self.cadence.attach(trainer.optimizer)
        write(self.out/'actual_initialization_audit.json',dict(exact_v16_best_state_equal=True,source_sha256=START_SHA,
            initial_state_sha256=state_digest(self.initial),frozen_parameters=frozen,
            trainable_parameters=sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)))
        print('V17_INIT_AUDIT: exact v16 best state; full learned network trainable',flush=True)


def rehearse(model,data,audit,config,out):
    from ultralytics.cfg import get_cfg
    from ultralytics.data.dataset import YOLODataset
    rows=read(data.parent/'build_manifest.json')['lineage']
    groups=[([r for r in rows if r.get('v17_new_flame_revision')],2),
            ([r for r in rows if r.get('v17_reviewed_negative')],2),
            ([r for r in rows if r['visible_flame_train_revision'] and not r.get('v17_new_flame_revision')],2),
            ([r for r in rows if r['origin']=='legacy_train' and r['teacher_preserve'] and not r.get('v17_reviewed_negative')],1),
            ([r for r in rows if r['origin']!='legacy_train' and r.get('boxes_xyxy')],1)]
    selected=[]
    for candidates,count in groups:
        if len(candidates)<count:
            raise ValueError('Missing actual rehearsal category')
        selected.extend(sorted(candidates,key=lambda r:r['image'])[:count])
    paths=out/'rehearsal_images.txt'; out.mkdir(parents=True)
    paths.write_text('\n'.join(str(data.parent/'images/train'/r['image']) for r in selected)+'\n',encoding='utf-8')
    ds=YOLODataset(img_path=str(paths),data={'nc':1,'names':{0:'fire'}},imgsz=640,batch_size=8,augment=False,rect=False,cache=False)
    batch=ds.collate_fn([ds[i] for i in range(len(ds))]); batch['img']=batch['img'].float()/255
    if len(batch['im_file'])!=8:
        raise RuntimeError('Wrong rehearsal batch')
    torch.set_num_threads(2); core=model.model.cpu().train(); core.args=get_cfg(overrides=config)
    for name,p in core.named_parameters(): p.requires_grad_('.dfl.' not in name)
    optimizer=torch.optim.AdamW([p for p in core.parameters() if p.requires_grad],lr=config['lr0'],weight_decay=config['weight_decay'])
    trainer=SimpleNamespace(model=core,optimizer=optimizer,accumulate=1,epoch=0)
    try:
        audit.on_start(trainer)
        loss,items=core.loss(batch)
        if not torch.isfinite(loss).all(): raise RuntimeError('Nonfinite rehearsal loss')
        loss.sum().backward()
        gradients=[p.grad for p in core.parameters() if p.requires_grad]
        if not all(g is not None and torch.isfinite(g).all() for g in gradients):
            raise RuntimeError('Missing/nonfinite learned gradient')
        optimizer.step(); audit.cadence.on_batch_end(trainer); audit.epoch_end(trainer)
        if len(audit.cadence.steps)!=1: raise RuntimeError('Wrong actual rehearsal update')
        write(out/'cpu_rehearsal_complete.json',dict(candidate_training=False,actual_optimizer_steps=1,images=8,
            new_flame_images=2,new_negative_images=2,augmented=False,loss=[float(v) for v in items.detach()],
            all_learned_gradients_finite=True,full_network_audit=audit.epochs[0],
            source_inputs=[dict(image_sha256=r['sha256'],label_sha256=r['label_sha256']) for r in selected]))
    finally: audit.cadence.close()


def validate_completion(meta,epochs,cadence):
    if not meta['candidate_training'] or meta['source_sha256']!=START_SHA or meta['config']['epochs']!=12 or meta['checkpoint_screen']!=PRIMARY or set(meta['checkpoints'])!=set(PRIMARY):
        raise ValueError('Wrong completed v17 route')
    if len(epochs)!=12 or [r['epoch'] for r in epochs]!=list(range(1,13)) or not all(r['changed_feature_weight_tensors']>0 and r['changed_head_weight_tensors']>0 for r in epochs):
        raise ValueError('Incomplete full-network fixed schedule')
    if cadence['actual_optimizer_steps']!=meta['actual_optimizer_steps'] or cadence['batches']!=meta['actual_batches'] or meta['actual_optimizer_steps']<=0:
        raise ValueError('Update audit differs')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--start',type=Path,required=True); p.add_argument('--data',type=Path,required=True)
    p.add_argument('--expected-data-sha256',required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--rehearse-cpu',action='store_true'); a=p.parse_args()
    out=a.out.resolve(); marker=out.parent/(out.name+'_preflight.json')
    if out.exists() or marker.exists(): raise ValueError('Existing run/preflight; no overwrite')
    if sha256(a.start)!=START_SHA: raise ValueError('Selected v16 source changed')
    checked=verify(a.data.resolve().parent,a.expected_data_sha256)
    parent_meta=read(a.start.resolve().parents[1]/'experiment_meta.json')
    if parent_meta['checkpoints']['best.pt']!=START_SHA or parent_meta['actual_optimizer_steps']!=2837 or parent_meta['config']['epochs']!=12:
        raise ValueError('v16 completion identity differs')
    if not a.rehearse_cpu and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    model=YOLO(str(a.start)); reconstructed=verify_trainer_reconstruction(model,a.start)
    if model.names!={0:'fire'}: raise ValueError('Expected single fire class')
    audit=ExtensionAudit(model.model,out); config=schedule(a.data,out,a.start)
    identity=dict(role='Full-network label/negative extension from v16 best; not independent acceptance or a single-factor ablation',
        source_sha256=START_SHA,dataset=checked,config=config,candidate_training=not a.rehearse_cpu,
        checkpoint_screen=PRIMARY,teacher_loss=False,actual_trainer_reconstruction=reconstructed,
        parent_experiment_sha256=sha256(a.start.resolve().parents[1]/'experiment_meta.json'),script_sha256=sha256(Path(__file__)),
        dependencies={n:sha256(Path(__file__).with_name(n)) for n in DEPENDENCIES},
        environment=dict(python=platform.python_version(),torch=torch.__version__,ultralytics=ultralytics.__version__))
    if a.rehearse_cpu:
        rehearse(model,a.data.resolve(),audit,config,out); write(out/'cpu_rehearsal_identity.json',identity); return
    marker.parent.mkdir(parents=True,exist_ok=True); write(marker,identity)
    model.add_callback('on_train_start',audit.on_start)
    model.add_callback('on_train_batch_end',audit.cadence.on_batch_end)
    model.add_callback('on_train_epoch_end',audit.epoch_end)
    try: model.train(**config)
    except Exception as error:
        out.mkdir(parents=True,exist_ok=True); write(out/'failed.json',dict(error_type=type(error).__name__,message=str(error))); raise
    finally: audit.cadence.close()
    identity.update(checkpoints={p.name:sha256(p) for p in (out/'weights').glob('*.pt')},
        actual_optimizer_steps=len(audit.cadence.steps),actual_batches=audit.cadence.batches,
        completed_audits={n:sha256(out/n) for n in ['actual_initialization_audit.json','full_network_update_audit.json','optimizer_cadence_audit.json']})
    validate_completion(identity,audit.epochs,read(out/'optimizer_cadence_audit.json'))
    write(out/'experiment_meta.json',identity)
    print('V17 completed; best/last both require project image/video comparison',flush=True)


if __name__=='__main__': main()
