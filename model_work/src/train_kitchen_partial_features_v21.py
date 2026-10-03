"""V21 full-context partial-feature adaptation, with fixed feature BN statistics.

Same v20 data/start/schedule; freeze modules0-9 instead of0-22. Learn modules
10-22 and head23, preserving all feature BN running buffers and fixed DFL.
Both saved checkpoints require unchanged project image/video comparisons.
"""
import argparse
from pathlib import Path
import platform
from types import SimpleNamespace
from data_integrity import sha256
from build_kitchen_source_extension_v20 import read,write,verify
from train_kitchen_source_head_v20 import START_SHA,DATA_SHA,DEPENDENCIES as V20_DEPENDENCIES,frozen_schedule

PRIMARY=['best.pt','last.pt']
PREFIX=10
FEATURES=23
DEPENDENCIES=['train_kitchen_source_head_v20.py']+V20_DEPENDENCIES
ROOT=Path(__file__).resolve().parents[2]


def feature_bn_state(core):
    from backbone_freeze_audit import is_backbone_key
    return {k:v.detach().cpu().clone() for k,v in core.state_dict().items()
            if is_backbone_key(k,FEATURES) and k.endswith(('running_mean','running_var','num_batches_tracked'))}


class PartialFeatureAudit:
    def __init__(self,core,out):
        from backbone_freeze_audit import FrozenBackboneAudit
        from optimizer_cadence_audit import OptimizerCadenceAudit
        if len(core.model)!=FEATURES+1:raise ValueError('Unexpected architecture')
        self.blocks=PREFIX;self.out=out;self.epochs=[]
        self.initial={k:v.detach().cpu().clone() for k,v in core.state_dict().items()}
        self.feature_bn=feature_bn_state(core)
        self.freeze=FrozenBackboneAudit(core,PREFIX,out/'frozen_prefix_audit.json')
        self.cadence=OptimizerCadenceAudit(out/'optimizer_cadence_audit.json')

    def on_start(self,trainer):
        import torch
        from backbone_freeze_audit import is_backbone_key,state_digest
        actual=trainer.model.state_dict()
        if actual.keys()!=self.initial.keys() or any(not torch.equal(v.detach().cpu(),self.initial[k]) for k,v in actual.items()):
            raise RuntimeError('Actual initialization differs from v16 best')
        for n,p in trainer.model.named_parameters():
            if p.requires_grad==(is_backbone_key(n,PREFIX) or '.dfl.' in n):raise RuntimeError('Wrong partial freeze mask')
        if trainer.accumulate!=1:raise RuntimeError('Wrong accumulation')
        self.freeze.on_start(trainer);self.cadence.attach(trainer.optimizer)
        write(self.out/'actual_initialization_audit.json',dict(exact_v16_best_state_equal=True,
            source_sha256=START_SHA,state_sha256=state_digest(self.initial),state_tensors=len(actual),
            frozen_prefix_blocks=PREFIX,feature_bn_buffers=len(self.feature_bn),
            trainable_parameters=sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)))
        print('V21_INIT: exact v16; frozen prefix10; adaptive features10-22 plus head23',flush=True)

    def batch_start(self,trainer):
        import torch
        # Trainer sets training mode before this callback. Keep the same feature
        # running statistics as head-only v20; late BN affine weights may learn.
        for module in trainer.model.model[:FEATURES]:
            for child in module.modules():
                if isinstance(child,torch.nn.BatchNorm2d):child.eval()
        self.freeze.on_batch_start(trainer)
        if any(m.training for layer in trainer.model.model[:FEATURES] for m in layer.modules() if isinstance(m,torch.nn.BatchNorm2d)):
            raise RuntimeError('Feature BN statistics would update')

    def epoch_end(self,trainer):
        import torch
        from backbone_freeze_audit import is_backbone_key,state_digest
        self.freeze.on_epoch_end(trainer)
        live={k:v.detach().cpu() for k,v in trainer.model.state_dict().items()}
        if any(v.is_floating_point() and not torch.isfinite(v).all() for v in live.values()):raise RuntimeError('Nonfinite state')
        head=[k for k,v in live.items() if k.startswith('model.23.') and k.endswith('.weight') and not torch.equal(v,self.initial[k])]
        feature=[k for k,v in live.items() if is_backbone_key(k,FEATURES) and not is_backbone_key(k,PREFIX) and k.endswith('.weight') and not torch.equal(v,self.initial[k])]
        if not head or not feature:raise RuntimeError('Head and late features must actually learn')
        if any(not torch.equal(live[k],v) for k,v in self.feature_bn.items()):raise RuntimeError('Feature BN buffers changed')
        dfl='model.23.dfl.conv.weight'
        if not torch.equal(live[dfl],self.initial[dfl]):raise RuntimeError('Fixed DFL changed')
        if trainer.ema is not None:
            ema=trainer.ema.ema.state_dict()
            fixed={**self.freeze.reference,**self.feature_bn,dfl:self.initial[dfl]}
            with torch.no_grad():
                for k,v in fixed.items():ema[k].copy_(v.to(ema[k]))
            if any(not torch.equal(ema[k].detach().cpu(),v) for k,v in fixed.items()):raise RuntimeError('EMA fixed state changed')
        self.epochs.append(dict(epoch=trainer.epoch+1,frozen_prefix_unchanged=True,feature_bn_unchanged=True,
            changed_feature_weight_tensors=len(feature),changed_head_weight_tensors=len(head),
            model_state_sha256=state_digest(live),actual_optimizer_steps=len(self.cadence.steps),
            ema_fixed_state_restored=trainer.ema is not None))
        write(self.out/'partial_feature_update_audit.json',self.epochs);self.cadence.on_epoch_end(trainer)


def validate_completion(meta,epochs,cadence):
    if meta['dataset']['manifest_sha256']!=DATA_SHA or meta['source_sha256']!=START_SHA:
        raise ValueError('Wrong completed source/data')
    if not meta['candidate_training'] or meta['config']['epochs']!=12 or meta['config']['freeze']!=PREFIX or meta['frozen_prefix_blocks']!=PREFIX or meta['feature_blocks']!=FEATURES or not meta['feature_bn_statistics_fixed']:
        raise ValueError('Wrong partial-feature route')
    if meta['checkpoint_screen']!=PRIMARY or set(meta['checkpoints'])!=set(PRIMARY) or not meta['serialized_frozen_state_equal']:
        raise ValueError('Both audited saved checkpoints required')
    if set(meta['serialized_adaptation'])!=set(PRIMARY) or any(min(r.values())<=0 for r in meta['serialized_adaptation'].values()):
        raise ValueError('Saved adaptation missing')
    if len(epochs)!=12 or [r['epoch'] for r in epochs]!=list(range(1,13)) or not all(
        r['frozen_prefix_unchanged'] and r['feature_bn_unchanged'] and r['changed_head_weight_tensors']>0
        and r['changed_feature_weight_tensors']>0 and r['ema_fixed_state_restored'] for r in epochs):
        raise ValueError('Incomplete partial-feature audit')
    if cadence['actual_optimizer_steps']!=meta['actual_optimizer_steps'] or cadence['batches']!=meta['actual_batches'] or meta['actual_optimizer_steps']<=0:
        raise ValueError('Wrong actual optimizer cadence')


def rehearse(model,data,audit,config,out):
    import torch
    from ultralytics.cfg import get_cfg
    from ultralytics.data.dataset import YOLODataset
    from backbone_freeze_audit import is_backbone_key
    rows=read(data.parent/'build_manifest.json')['lineage']
    groups=[([r for r in rows if r.get('v20_new_source') and r['visible_target']=='flame'],4),
        ([r for r in rows if r.get('v20_new_source') and r['visible_target']=='visible_no_flame' and r['image'].startswith('context3')],1),
        ([r for r in rows if r.get('v20_new_source') and r['visible_target']=='visible_no_flame' and r['image'].startswith('context4')],1),
        ([r for r in rows if r['origin']=='legacy_train' and r['visible_flame_train_revision']],2)]
    chosen=[]
    for candidates,count in groups:
        if len(candidates)<count:raise ValueError('Missing rehearsal group')
        chosen.extend(sorted(candidates,key=lambda r:r['image'])[:count])
    out.mkdir(parents=True);paths=out/'rehearsal_images.txt'
    paths.write_text('\n'.join(str(data.parent/'images/train'/r['image']) for r in chosen)+'\n',encoding='utf-8')
    ds=YOLODataset(img_path=str(paths),data={'nc':1,'names':{0:'fire'}},imgsz=640,batch_size=8,augment=False,rect=False,cache=False)
    batch=ds.collate_fn([ds[i] for i in range(len(ds))]);batch['img']=batch['img'].float()/255
    if len(batch['im_file'])!=8:raise RuntimeError('Wrong rehearsal size')
    torch.set_num_threads(2);core=model.model.cpu().train();core.args=get_cfg(overrides=config)
    for n,p in core.named_parameters():p.requires_grad_(not is_backbone_key(n,PREFIX) and '.dfl.' not in n)
    optimizer=torch.optim.AdamW([p for p in core.parameters() if p.requires_grad],lr=config['lr0'],weight_decay=config['weight_decay'])
    trainer=SimpleNamespace(model=core,optimizer=optimizer,accumulate=1,epoch=0,ema=None)
    try:
        audit.on_start(trainer);audit.batch_start(trainer);loss,parts=core.loss(batch)
        if not torch.isfinite(loss).all():raise RuntimeError('Nonfinite rehearsal loss')
        loss.sum().backward();grads=[p.grad for p in core.parameters() if p.requires_grad]
        if not grads or not all(g is not None and torch.isfinite(g).all() for g in grads):raise RuntimeError('Missing/nonfinite learned gradients')
        if any(p.grad is not None for p in core.parameters() if not p.requires_grad):raise RuntimeError('Frozen parameter has gradient')
        optimizer.step();audit.cadence.on_batch_end(trainer);audit.epoch_end(trainer)
        if len(audit.cadence.steps)!=1:raise RuntimeError('Wrong actual rehearsal updates')
        write(out/'cpu_rehearsal_complete.json',dict(candidate_training=False,images=8,actual_optimizer_steps=1,
            loss=[float(v) for v in parts.detach()],all_learned_gradients_finite=True,frozen_gradients_absent=True,
            audit=audit.epochs[0],inputs=[dict(image=r['image'],sha256=r['sha256'],label_sha256=r['label_sha256']) for r in chosen]))
    finally:audit.cadence.close()


def main():
    import torch,ultralytics
    from ultralytics import YOLO
    from train_scratch_semantic_v16 import verify_trainer_reconstruction
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['start','data','out']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--rehearse-cpu',action='store_true');p.add_argument('--rehearsal',type=Path);a=p.parse_args()
    out=a.out.resolve();marker=out.parent/(out.name+'_preflight.json')
    if out.exists() or marker.exists():raise ValueError('Existing run/preflight')
    if sha256(a.start)!=START_SHA:raise ValueError('Source checkpoint changed')
    dataset=verify(a.data.resolve().parent,DATA_SHA)
    previous_path=ROOT/'docs/fire_kitchen_source_v20_results_20261003.json'
    diagnostic_path=ROOT/'docs/kitchen_source_learnability_v1_results_20261003.json'
    previous,diagnostic=read(previous_path),read(diagnostic_path)
    if previous['training']['actual_optimizer_steps']!=3618 or previous['training']['dataset']['manifest_sha256']!=DATA_SHA or not previous['training']['serialized_checkpoint_features_equal']:
        raise ValueError('Completed v20 comparison missing')
    if diagnostic['actual_optimizer_steps_total']!=320 or diagnostic['trained_weights_saved'] or diagnostic['reserved_predicted_or_trained']:
        raise ValueError('Completed diagnostic prerequisites missing')
    model=YOLO(str(a.start));rebuilt=verify_trainer_reconstruction(model,a.start)
    if model.names!={0:'fire'}:raise ValueError('Wrong class')
    audit=PartialFeatureAudit(model.model,out);config=frozen_schedule(a.data,out,a.start,PREFIX)
    before={k:v for k,v in previous['training']['config'].items() if k not in ['freeze','name','project']}
    after={k:v for k,v in config.items() if k not in ['freeze','name','project']}
    if before!=after:raise ValueError('Unintended v20 schedule/data difference')
    identity=dict(role='V21 partial feature adaptation on full v20 mixed data; feature BN fixed; no teacher',
        source_sha256=START_SHA,dataset=dataset,config=config,frozen_prefix_blocks=PREFIX,feature_blocks=FEATURES,
        feature_bn_statistics_fixed=True,candidate_training=not a.rehearse_cpu,checkpoint_screen=PRIMARY,
        teacher_loss=False,actual_trainer_reconstruction=rebuilt,script_sha256=sha256(Path(__file__)),
        dependencies={n:sha256(Path(__file__).with_name(n)) for n in DEPENDENCIES},
        prerequisites={str(p.relative_to(ROOT)):sha256(p) for p in [previous_path,diagnostic_path]},
        v20_schedule_equal_except_freeze_and_output=True,
        environment=dict(python=platform.python_version(),torch=torch.__version__,ultralytics=ultralytics.__version__))
    if a.rehearse_cpu:
        rehearse(model,a.data.resolve(),audit,config,out);write(out/'cpu_rehearsal_identity.json',identity);return
    if a.rehearsal is None:raise ValueError('Final-source CPU rehearsal required')
    old=read(a.rehearsal/'cpu_rehearsal_identity.json');check=read(a.rehearsal/'cpu_rehearsal_complete.json')
    for key in ['script_sha256','dependencies','prerequisites','dataset','source_sha256']:
        if old[key]!=identity[key]:raise ValueError('CPU rehearsal differs from actual source/schedule: '+key)
    old_config={k:v for k,v in old['config'].items() if k not in ['name','project']}
    new_config={k:v for k,v in identity['config'].items() if k not in ['name','project']}
    if old_config!=new_config:raise ValueError('CPU rehearsal schedule differs')
    if old['candidate_training'] or check['actual_optimizer_steps']!=1 or not check['all_learned_gradients_finite'] or not check['frozen_gradients_absent'] or not check['audit']['frozen_prefix_unchanged'] or not check['audit']['feature_bn_unchanged'] or check['audit']['changed_feature_weight_tensors']<=0:
        raise ValueError('Incomplete CPU rehearsal')
    identity['cpu_rehearsal']=dict(path=str(a.rehearsal.resolve()),identity_sha256=sha256(a.rehearsal/'cpu_rehearsal_identity.json'),complete_sha256=sha256(a.rehearsal/'cpu_rehearsal_complete.json'))
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    write(marker,identity)
    source_bytes={n:Path(__file__).with_name(n).read_bytes() for n in [Path(__file__).name]+DEPENDENCIES}
    def capture(trainer):
        if Path(trainer.save_dir).resolve()!=out:raise RuntimeError('Trainer changed output')
        snapshot=out/'source_snapshot';snapshot.mkdir()
        for n,payload in source_bytes.items():
            if Path(__file__).with_name(n).read_bytes()!=payload:raise RuntimeError('Source changed at launch')
            (snapshot/n).write_bytes(payload)
    model.add_callback('on_train_start',capture);model.add_callback('on_train_start',audit.on_start)
    model.add_callback('on_train_batch_start',audit.batch_start);model.add_callback('on_train_batch_end',audit.cadence.on_batch_end)
    model.add_callback('on_train_epoch_end',audit.epoch_end)
    try:model.train(**config)
    except Exception as error:write(out/'failed.json',dict(error_type=type(error).__name__,message=str(error)));raise
    finally:audit.cadence.close()
    fixed={**audit.freeze.reference,**audit.feature_bn,'model.23.dfl.conv.weight':audit.initial['model.23.dfl.conv.weight']}
    saved_adaptation={}
    for checkpoint in (out/'weights').glob('*.pt'):
        live=YOLO(str(checkpoint)).model.state_dict()
        if any(not torch.equal(live[k].detach().cpu(),v) for k,v in fixed.items()):raise RuntimeError('Saved fixed prefix/BN/DFL changed')
        from backbone_freeze_audit import is_backbone_key
        changed=[k for k,v in live.items() if k.endswith('.weight') and not torch.equal(v.detach().cpu(),audit.initial[k])]
        counts=dict(feature_weight_tensors=sum(is_backbone_key(k,FEATURES) and not is_backbone_key(k,PREFIX) for k in changed),head_weight_tensors=sum(k.startswith('model.23.') for k in changed))
        if min(counts.values())<=0:raise RuntimeError('Saved checkpoint has no feature/head adaptation')
        saved_adaptation[checkpoint.name]=counts
    identity.update(checkpoints={p.name:sha256(p) for p in (out/'weights').glob('*.pt')},
        actual_optimizer_steps=len(audit.cadence.steps),actual_batches=audit.cadence.batches,serialized_frozen_state_equal=True,serialized_adaptation=saved_adaptation,
        completed_audits={n:sha256(out/n) for n in ['actual_initialization_audit.json','frozen_prefix_audit.json','partial_feature_update_audit.json','optimizer_cadence_audit.json']})
    validate_completion(identity,audit.epochs,read(out/'optimizer_cadence_audit.json'));write(out/'experiment_meta.json',identity)
    print('V21 completed; best/last still require fixed image/video comparisons',flush=True)


if __name__=='__main__':main()
