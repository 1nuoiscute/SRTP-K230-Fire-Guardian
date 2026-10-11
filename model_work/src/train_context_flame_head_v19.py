"""Fixed-feature v19 route from v16 on reviewed train-context data; no teacher."""
import argparse
from pathlib import Path
import platform
from types import SimpleNamespace
from data_integrity import sha256
from curate_train_context_v19 import read, write, verify

START_SHA='e770cf68d12cd43b4494a1439dc9c65f14d1d5fc570b2e2e379cc084384418fd'
DATA_SHA='9971c9530a04f1dd962a7c2d9e229d73d5b7e1a47aab5d3d966975d6e78d831a'
PRIMARY=['best.pt','last.pt']
DEPENDENCIES=['curate_train_context_v19.py','train_frozen_flame_head_v18.py','train_flame_extension_v17.py','train_scratch_semantic_v16.py','extend_flame_review_v17.py','backbone_freeze_audit.py','optimizer_cadence_audit.py','data_integrity.py']


def frozen_schedule(data,out,start,blocks):
    from train_flame_extension_v17 import schedule
    config=schedule(data,out,start);config.update(freeze=blocks)
    return config


class HeadAudit:
    def __init__(self,core,out):
        from backbone_freeze_audit import FrozenBackboneAudit
        from optimizer_cadence_audit import OptimizerCadenceAudit
        self.blocks=len(core.model)-1;self.out=out;self.epochs=[]
        self.initial={k:v.detach().cpu().clone() for k,v in core.state_dict().items()}
        self.freeze=FrozenBackboneAudit(core,self.blocks,out/'feature_audit.json')
        self.cadence=OptimizerCadenceAudit(out/'optimizer_cadence_audit.json')

    def on_start(self,trainer):
        import torch
        from backbone_freeze_audit import is_backbone_key,state_digest
        actual=trainer.model.state_dict()
        if actual.keys()!=self.initial.keys() or any(not torch.equal(v.detach().cpu(),self.initial[k]) for k,v in actual.items()):raise RuntimeError('Actual v19 start differs')
        for n,p in trainer.model.named_parameters():
            if p.requires_grad==(is_backbone_key(n,self.blocks) or '.dfl.' in n):raise RuntimeError('Wrong v19 freeze scope')
        if trainer.accumulate!=1:raise RuntimeError('Wrong accumulation')
        self.freeze.on_start(trainer);self.cadence.attach(trainer.optimizer)
        write(self.out/'actual_initialization_audit.json',dict(exact_v16_best_state_equal=True,source_sha256=START_SHA,
            state_sha256=state_digest(self.initial),state_tensors=len(actual),fixed_feature_blocks=self.blocks,
            trainable_parameters=sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)))
        print('V19_INIT_AUDIT: exact v16 best; feature parameters/buffers fixed, head trainable',flush=True)

    def batch_start(self,trainer):
        self.freeze.on_batch_start(trainer)

    def epoch_end(self,trainer):
        import torch
        from backbone_freeze_audit import state_digest
        self.freeze.on_epoch_end(trainer)
        live={k:v.detach().cpu() for k,v in trainer.model.state_dict().items()}
        if any(v.is_floating_point() and not torch.isfinite(v).all() for v in live.values()):raise RuntimeError('Nonfinite state')
        changes=[k for k,v in live.items() if k.startswith(f'model.{self.blocks}.') and k.endswith('.weight') and not torch.equal(v,self.initial[k])]
        if not changes:raise RuntimeError('Head did not learn')
        dfl=f'model.{self.blocks}.dfl.conv.weight'
        if dfl in live and not torch.equal(live[dfl],self.initial[dfl]):raise RuntimeError('Fixed DFL changed')
        if trainer.ema is not None:
            # EMA arithmetic may alter fixed values; restore actual source bytes.
            ema=trainer.ema.ema.state_dict()
            with torch.no_grad():
                for k,v in self.freeze.reference.items():ema[k].copy_(v.to(ema[k]))
            if any(not torch.equal(ema[k].detach().cpu(),v) for k,v in self.freeze.reference.items()):raise RuntimeError('EMA fixed features changed')
        self.epochs.append(dict(epoch=trainer.epoch+1,feature_unchanged=True,changed_head_weight_tensors=len(changes),
            model_state_sha256=state_digest(live),actual_optimizer_steps=len(self.cadence.steps),ema_fixed_features_restored=trainer.ema is not None))
        write(self.out/'head_update_audit.json',self.epochs);self.cadence.on_epoch_end(trainer)


def validate_completion(meta,epochs,cadence):
    if meta['dataset']['manifest_sha256']!=DATA_SHA:raise ValueError('Wrong completed v19 data')
    if not meta['candidate_training'] or meta['source_sha256']!=START_SHA or meta['config']['epochs']!=12 or meta['config']['freeze']!=meta['fixed_feature_blocks'] or meta['checkpoint_screen']!=PRIMARY or set(meta['checkpoints'])!=set(PRIMARY):raise ValueError('Wrong completed route')
    if len(epochs)!=12 or [r['epoch'] for r in epochs]!=list(range(1,13)) or not all(r['feature_unchanged'] and r['changed_head_weight_tensors']>0 and r['ema_fixed_features_restored'] for r in epochs):raise ValueError('Incomplete head/feature audit')
    if cadence['actual_optimizer_steps']!=meta['actual_optimizer_steps'] or cadence['batches']!=meta['actual_batches'] or meta['actual_optimizer_steps']<=0:raise ValueError('Optimizer audit differs')


def rehearse(model,data,audit,config,out):
    import torch
    from ultralytics.cfg import get_cfg
    from ultralytics.data.dataset import YOLODataset
    from backbone_freeze_audit import is_backbone_key
    rows=read(data.parent/'build_manifest.json')['lineage']
    groups=[([r for r in rows if r.get('v19_context_review_status')=='flame'],2),([r for r in rows if r.get('v19_context_review_status')=='negative'],2),
        ([r for r in rows if r['visible_flame_train_revision'] and not r.get('v19_context_review_status')],2),
        ([r for r in rows if r['origin']=='legacy_train' and r['teacher_preserve'] and not r.get('v19_context_review_status')],1),
        ([r for r in rows if r['origin']!='legacy_train' and r.get('boxes_xyxy')],1)]
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
    for n,p in core.named_parameters():p.requires_grad_(not is_backbone_key(n,audit.blocks) and '.dfl.' not in n)
    for module in core.model[:audit.blocks]:module.eval()
    optimizer=torch.optim.AdamW([p for p in core.parameters() if p.requires_grad],lr=config['lr0'],weight_decay=config['weight_decay'])
    trainer=SimpleNamespace(model=core,optimizer=optimizer,accumulate=1,epoch=0,ema=None)
    try:
        audit.on_start(trainer);audit.batch_start(trainer);loss,items=core.loss(batch)
        if not torch.isfinite(loss).all():raise RuntimeError('Nonfinite loss')
        loss.sum().backward();grads=[p.grad for p in core.parameters() if p.requires_grad]
        if not grads or not all(g is not None and torch.isfinite(g).all() for g in grads):raise RuntimeError('Missing/nonfinite head gradients')
        if any(p.grad is not None for n,p in core.named_parameters() if is_backbone_key(n,audit.blocks)):raise RuntimeError('Feature gradient present')
        optimizer.step();audit.cadence.on_batch_end(trainer);audit.epoch_end(trainer)
        if len(audit.cadence.steps)!=1:raise RuntimeError('Wrong update count')
        write(out/'cpu_rehearsal_complete.json',dict(candidate_training=False,images=8,actual_optimizer_steps=1,
            loss=[float(v) for v in items.detach()],all_learned_head_gradients_finite=True,feature_gradients_absent=True,
            audit=audit.epochs[0],source_inputs=[dict(image_sha256=r['sha256'],label_sha256=r['label_sha256']) for r in chosen]))
    finally:audit.cadence.close()


def main():
    import torch,ultralytics
    from ultralytics import YOLO
    from train_scratch_semantic_v16 import verify_trainer_reconstruction
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['start','data','out']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--rehearse-cpu',action='store_true');a=p.parse_args()
    out=a.out.resolve();marker=out.parent/(out.name+'_preflight.json')
    if out.exists() or marker.exists():raise ValueError('Existing run/preflight')
    if sha256(a.start)!=START_SHA:raise ValueError('Source checkpoint changed')
    dataset=verify(a.data.resolve().parent,DATA_SHA)
    previous_path=Path(__file__).resolve().parents[2]/'docs/fire_frozen_flame_head_v18_results_20261003.json'
    previous=read(previous_path)
    if previous['training']['actual_optimizer_steps']!=3198 or previous['comparison_plan']['data_manifest_sha256']!='8e70dea69f39beafa0f17b3a6d4b483c537622fdf52271755fd8f3e2447bd54d' or not previous['training']['serialized_checkpoint_features_equal']:raise ValueError('v18 completion missing')
    model=YOLO(str(a.start));rebuilt=verify_trainer_reconstruction(model,a.start)
    if model.names!={0:'fire'}:raise ValueError('Wrong classes')
    audit=HeadAudit(model.model,out);config=frozen_schedule(a.data,out,a.start,audit.blocks)
    identity=dict(role='Fixed v16 features on v19 reviewed train contexts; v18 schedule, head-only route, no teacher',reference_v18_results_sha256=sha256(previous_path),source_sha256=START_SHA,
        dataset=dataset,config=config,fixed_feature_blocks=audit.blocks,candidate_training=not a.rehearse_cpu,
        checkpoint_screen=PRIMARY,teacher_loss=False,actual_trainer_reconstruction=rebuilt,
        script_sha256=sha256(Path(__file__)),dependencies={n:sha256(Path(__file__).with_name(n)) for n in DEPENDENCIES},
        environment=dict(python=platform.python_version(),torch=torch.__version__,ultralytics=ultralytics.__version__))
    if a.rehearse_cpu:
        rehearse(model,a.data.resolve(),audit,config,out);write(out/'cpu_rehearsal_identity.json',identity);return
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    write(marker,identity)
    model.add_callback('on_train_start',audit.on_start);model.add_callback('on_train_batch_start',audit.batch_start)
    model.add_callback('on_train_batch_end',audit.cadence.on_batch_end);model.add_callback('on_train_epoch_end',audit.epoch_end)
    try:model.train(**config)
    except Exception as error:write(out/'failed.json',dict(error_type=type(error).__name__,message=str(error)));raise
    finally:audit.cadence.close()
    for checkpoint in (out/'weights').glob('*.pt'):
        live=YOLO(str(checkpoint)).model.state_dict()
        if any(not torch.equal(live[k].detach().cpu(),v) for k,v in audit.freeze.reference.items()):raise RuntimeError('Serialized checkpoint features differ')
    identity.update(checkpoints={p.name:sha256(p) for p in (out/'weights').glob('*.pt')},actual_optimizer_steps=len(audit.cadence.steps),actual_batches=audit.cadence.batches,
        completed_audits={n:sha256(out/n) for n in ['actual_initialization_audit.json','feature_audit.json','head_update_audit.json','optimizer_cadence_audit.json']},serialized_checkpoint_features_equal=True)
    validate_completion(identity,audit.epochs,read(out/'optimizer_cadence_audit.json'));write(out/'experiment_meta.json',identity)
    print('V19 completed; both best/last require fixed image/video comparison',flush=True)


if __name__=='__main__':main()
