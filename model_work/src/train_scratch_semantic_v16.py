"""Continue v14 features on reviewed flame geometry; bounded full-network pilot.

This is fine-tuning from an audited random-origin model, not a new scratch run
or an initialization-only ablation. Preserve all previous artifacts.
"""
import argparse
import json
from pathlib import Path
import platform

from data_integrity import sha256
from semantic_fire_dataset import verify, read

START_SHA = 'a46690756e65826060392688e98ceb268c9ecfa3f007e5796fa3c38ef4b5833e'
DATA_SHA = 'f3237c9ae34804918968a34945d51886266313aed83b7938e012586e11471069'
VIDEO_SHA = '2309f2b6d03961612a64e65822cb7feb57e1733541f438280d4e8b9e2c5d133a'
V15_SHA = 'ad486fc5668ee4e980bc4f66e95c114e02dbc11924972afbda8400ac66d0ef00'
PRIMARY = ['best.pt', 'last.pt']


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def prerequisites(start, data, video, semantic):
    if sha256(start) != START_SHA or sha256(video) != VIDEO_SHA or sha256(semantic) != V15_SHA:
        raise ValueError('Start or completed comparison identity differs')
    paired, previous = read(video), read(semantic)
    if (paired['models_sha256']['scratch_last'] != START_SHA or paired['total_samples'] != 307
            or paired['model_inferences'] != 1535 or not paired['source_frame_shared']):
        raise ValueError('Missing paired video comparison')
    if set(previous['models']) != {'v3', 'v13_best', 'semantic_v15_best', 'semantic_v15_last'}:
        raise ValueError('Incomplete v15 comparison')
    meta_path = start.parents[1] / 'experiment_meta.json'
    meta = read(meta_path)
    if meta['config']['epochs'] != 100 or meta['checkpoints']['last.pt'] != START_SHA:
        raise ValueError('Expected completed random-origin v14')
    if sha256(meta_path) != '9401b3a10ec496dbc784ee036a7c1266201964adcaa648c02c0a8f5bc3f95538':
        raise ValueError('v14 completion record differs')
    for name, expected in meta['completed_audits'].items():
        if sha256(start.parents[1] / name) != expected:
            raise ValueError('v14 training audit changed')
    return verify(data.parent, DATA_SHA)


def schedule(data, out, start):
    # In the pinned library pretrained=False discards even a loaded local model.
    # Explicit path makes the intended fine-tuning source unambiguous.
    return dict(data=str(data.resolve()), epochs=12, patience=0, pretrained=str(start.resolve()), resume=False,
                imgsz=640, batch=8, nbs=8, device=0, workers=0, seed=20261005,
                optimizer='AdamW', lr0=.0001, lrf=.1, weight_decay=.0005, warmup_epochs=0,
                freeze=None, hsv_h=.015, hsv_s=.3, hsv_v=.25, degrees=3, translate=.05,
                scale=.2, fliplr=.5, mosaic=0, mixup=0, close_mosaic=0,
                project=str(out.parent), name=out.name, exist_ok=False,
                save=True, save_period=-1, plots=True, val=True)


def verify_trainer_reconstruction(model, start):
    import torch
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.nn.tasks import load_checkpoint
    from backbone_freeze_audit import state_digest
    trainer = DetectionTrainer.__new__(DetectionTrainer)
    trainer.data = {'nc': 1, 'channels': 3}
    weights, _ = load_checkpoint(str(start.resolve()))
    rebuilt = trainer.get_model(cfg=model.model.yaml, weights=weights, verbose=False)
    source, actual = model.model.state_dict(), rebuilt.state_dict()
    differing = [k for k in source if k not in actual or not torch.equal(source[k].cpu(), actual[k].cpu())]
    if source.keys() != actual.keys() or differing:
        raise RuntimeError('Framework trainer reconstruction changed supplied source state')
    return dict(exact_source_state_equal=True, state_tensors=len(source),
                rebuilt_state_sha256=state_digest({k:v.detach().cpu() for k,v in actual.items()}))


class FullNetworkAudit:
    def __init__(self, core, out):
        from optimizer_cadence_audit import OptimizerCadenceAudit
        self.initial = {k: v.detach().cpu().clone() for k, v in core.state_dict().items()}
        self.out, self.blocks, self.epochs = out, len(core.model)-1, []
        self.cadence = OptimizerCadenceAudit(out / 'optimizer_cadence_audit.json')

    def on_start(self, trainer):
        import torch
        from backbone_freeze_audit import state_digest
        actual = trainer.model.state_dict()
        if actual.keys() != self.initial.keys() or any(not torch.equal(v.detach().cpu(), self.initial[k]) for k, v in actual.items()):
            raise RuntimeError('Actual start is not exact v14 last state')
        frozen = [n for n, p in trainer.model.named_parameters() if not p.requires_grad]
        if frozen != [f'model.{self.blocks}.dfl.conv.weight'] or trainer.accumulate != 1:
            raise RuntimeError('Unexpected frozen parameters or gradient accumulation')
        self.cadence.attach(trainer.optimizer)
        write(self.out / 'actual_initialization_audit.json', dict(
            exact_v14_last_state_equal=True, source_sha256=START_SHA,
            initial_state_sha256=state_digest(self.initial), frozen_parameters=frozen,
            trainable_parameters=sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)))
        print('V16_INIT_AUDIT: exact v14 last state; all learned feature/head weights trainable', flush=True)

    def epoch_end(self, trainer):
        import torch
        from backbone_freeze_audit import is_backbone_key, state_digest
        state = {k: v.detach().cpu() for k, v in trainer.model.state_dict().items()}
        if any(v.is_floating_point() and not torch.isfinite(v).all() for v in state.values()):
            raise RuntimeError('Nonfinite learned state')
        changed = [k for k, v in state.items() if not torch.equal(v, self.initial[k])]
        feature = sum(is_backbone_key(k, self.blocks) and k.endswith('.weight') for k in changed)
        head = sum(k.startswith(f'model.{self.blocks}.') and k.endswith('.weight') for k in changed)
        if not feature or not head:
            raise RuntimeError('Features and head must both learn')
        dfl = f'model.{self.blocks}.dfl.conv.weight'
        if not torch.equal(state[dfl], self.initial[dfl]):
            raise RuntimeError('Fixed DFL bins changed')
        self.epochs.append(dict(epoch=trainer.epoch+1, changed_feature_weight_tensors=feature,
                                changed_head_weight_tensors=head, model_state_sha256=state_digest(state),
                                cumulative_actual_optimizer_steps=len(self.cadence.steps)))
        write(self.out / 'full_network_update_audit.json', self.epochs)
        self.cadence.on_epoch_end(trainer)


def rehearse(model, data, audit, config, out):
    import torch
    from types import SimpleNamespace
    from ultralytics.cfg import get_cfg
    from ultralytics.data.dataset import YOLODataset
    rows = read(data.parent / 'build_manifest.json')['lineage']
    groups = [([r for r in rows if r['visible_flame_train_revision']], 2),
              ([r for r in rows if r['retained_parent_revision']], 2),
              ([r for r in rows if r['teacher_preserve']], 2),
              ([r for r in rows if r['origin'] != 'legacy_train' and r.get('boxes_xyxy')], 1),
              ([r for r in rows if r['origin'] != 'legacy_train' and not r.get('boxes_xyxy')], 1)]
    selected = []
    for candidates, count in groups:
        if len(candidates) < count:
            raise ValueError('Missing rehearsal source group')
        selected.extend(sorted(candidates, key=lambda r: r['image'])[:count])
    paths = out / 'rehearsal_images.txt'
    paths.parent.mkdir(parents=True)
    paths.write_text('\n'.join(str(data.parent/'images/train'/r['image']) for r in selected)+'\n', encoding='utf-8')
    ds = YOLODataset(img_path=str(paths), data={'nc':1, 'names':{0:'fire'}}, imgsz=640,
                     batch_size=8, augment=False, rect=False, cache=False)
    batch = ds.collate_fn([ds[i] for i in range(len(ds))])
    if len(batch['im_file']) != 8:
        raise RuntimeError('Unexpected rehearsal batch')
    batch['img'] = batch['img'].float()/255
    torch.set_num_threads(2)
    core = model.model.cpu().train()
    core.args = get_cfg(overrides=config)
    for name, parameter in core.named_parameters():
        parameter.requires_grad_('.dfl.' not in name)
    optimizer = torch.optim.AdamW([p for p in core.parameters() if p.requires_grad], lr=config['lr0'], weight_decay=config['weight_decay'])
    trainer = SimpleNamespace(model=core, optimizer=optimizer, accumulate=1, epoch=0)
    try:
        audit.on_start(trainer)
        loss, items = core.loss(batch)
        if not torch.isfinite(loss).all():
            raise RuntimeError('Nonfinite rehearsal loss')
        loss.sum().backward()
        gradients = [p.grad for p in core.parameters() if p.requires_grad]
        if not gradients or not all(g is not None and torch.isfinite(g).all() for g in gradients):
            raise RuntimeError('Missing or nonfinite learned gradients')
        optimizer.step()
        audit.cadence.on_batch_end(trainer)
        audit.epoch_end(trainer)
        if len(audit.cadence.steps) != 1:
            raise RuntimeError('Expected one discarded rehearsal update')
        write(out/'cpu_rehearsal_complete.json', dict(candidate_training=False,
            role='One discarded CPU update; no candidate checkpoint or performance claim',
            images=8, augmented=False, actual_optimizer_steps=1, loss=[float(v) for v in items.detach()],
            all_learned_gradients_finite=True, full_network_audit=audit.epochs[0],
            source_inputs=[dict(image_sha256=r['sha256'], label_sha256=r['label_sha256']) for r in selected]))
    finally:
        audit.cadence.close()


def main():
    import torch
    import ultralytics
    from ultralytics import YOLO
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--start', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--video-comparison', type=Path, required=True)
    p.add_argument('--semantic-comparison', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--rehearse-cpu', action='store_true')
    a=p.parse_args()
    out=a.out.resolve()
    marker=out.parent/(out.name+'_preflight.json')
    if out.exists() or marker.exists():
        raise SystemExit('Existing output/preflight; do not restart or overwrite')
    checked=prerequisites(a.start.resolve(), a.data.resolve(), a.video_comparison, a.semantic_comparison)
    if not a.rehearse_cpu and not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable')
    model=YOLO(str(a.start))
    if model.names != {0:'fire'}:
        raise ValueError('Expected single fire class')
    reconstructed = verify_trainer_reconstruction(model, a.start)
    audit=FullNetworkAudit(model.model, out)
    config=schedule(a.data, out, a.start)
    identity=dict(role='Full-network semantic adaptation from v14 last; not a fresh random run',
        source_sha256=START_SHA, dataset=checked, config=config, teacher_loss=False,
        completed_video_comparison_sha256=VIDEO_SHA, completed_v15_comparison_sha256=V15_SHA,
        checkpoint_screen=PRIMARY, candidate_training=not a.rehearse_cpu,
        actual_trainer_reconstruction=reconstructed,
        script_sha256=sha256(Path(__file__)),
        dependencies={n:sha256(Path(__file__).with_name(n)) for n in
            ['semantic_fire_dataset.py','optimizer_cadence_audit.py','backbone_freeze_audit.py','data_integrity.py']},
        environment=dict(python=platform.python_version(), torch=torch.__version__, ultralytics=ultralytics.__version__),
        assessment='Project-scoped video utility, geometry and no-fire costs; no automatic historical-AP veto or adoption')
    if a.rehearse_cpu:
        rehearse(model, a.data.resolve(), audit, config, out)
        write(out/'cpu_rehearsal_identity.json', identity)
        return
    write(marker, identity)
    model.add_callback('on_train_start', audit.on_start)
    model.add_callback('on_train_batch_end', audit.cadence.on_batch_end)
    model.add_callback('on_train_epoch_end', audit.epoch_end)
    try:
        model.train(**config)
    except Exception as error:
        write(out/'failed.json', dict(error_type=type(error).__name__, message=str(error)))
        raise
    finally:
        audit.cadence.close()
    if len(audit.epochs)!=12 or sha256(a.start)!=START_SHA:
        raise RuntimeError('Incomplete fixed schedule or changed starting checkpoint')
    identity['checkpoints']={w.name:sha256(w) for w in (out/'weights').glob('*.pt')}
    if set(identity['checkpoints']) != set(PRIMARY):
        raise RuntimeError('Unexpected candidate inventory')
    identity['actual_optimizer_steps']=len(audit.cadence.steps)
    identity['actual_batches']=audit.cadence.batches
    identity['completed_audits']={n:sha256(out/n) for n in
        ['actual_initialization_audit.json','full_network_update_audit.json','optimizer_cadence_audit.json']}
    write(out/'experiment_meta.json', identity)
    print('V16 completed; best/last require separate project comparison', flush=True)


if __name__=='__main__':
    main()
