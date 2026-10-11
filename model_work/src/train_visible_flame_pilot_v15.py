"""Prepare/rehearse a semantic fire-head pilot; GPU run requires completed scratch comparison."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import platform
from types import SimpleNamespace
import torch
import ultralytics
from ultralytics import YOLO
from backbone_freeze_audit import FrozenBackboneAudit, is_backbone_key, state_digest
from data_integrity import sha256
from fire_teacher_loss import FireTeacherLoss
from optimizer_cadence_audit import OptimizerCadenceAudit
from semantic_fire_dataset import verify, read

START_SHA = 'c2b2a8bdc66fe452c80a1786f918f62f8686ad3b0f11061a761969ee02b09966'
PARENT_SHA = '56496195e463bd11f1071bb7299dab8198175911403016d11ef79bc03c61ccc7'
PRIMARY = ['best.pt', 'last.pt']
DEPENDENCIES = ('semantic_fire_dataset.py', 'train_flame_semantic_review.py',
                'prepare_train_flame_semantic_review.py', 'fire_teacher_loss.py',
                'fire_label_revision_dataset.py', 'backbone_freeze_audit.py',
                'optimizer_cadence_audit.py', 'dataset_preflight.py', 'data_integrity.py')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def schedule(data, out, blocks):
    return dict(data=str(data.resolve()), epochs=12, patience=0, imgsz=640, batch=8, nbs=8,
                device=0, workers=0, seed=20261004, optimizer='AdamW', lr0=.0001, lrf=.1,
                weight_decay=.0005, warmup_epochs=0, freeze=blocks, hsv_h=.015, hsv_s=.3,
                hsv_v=.25, degrees=3, translate=.05, scale=.2, fliplr=.5, mosaic=0, mixup=0,
                project=str(out.parent), name=out.name, exist_ok=False, save=True, save_period=-1,
                plots=True, val=True)


def completed_comparison(path, expected):
    if path is None or expected is None or sha256(path) != expected:
        raise ValueError('A fixed completed scratch comparison is required before GPU training')
    comparison = read(path)
    names = {'v3', 'v13_best', 'scratch_v14_best', 'scratch_v14_last'}
    if set(comparison['models']) != names or comparison['plan']['data_manifest_sha256'] != PARENT_SHA:
        raise ValueError('Wrong scratch comparison route')
    if comparison['models']['v13_best']['weights_sha256'] != START_SHA:
        raise ValueError('Selected v13 identity differs')
    models = comparison['plan']['models']
    for name in names:
        if sha256(Path(models[name]['weights'])) != models[name]['expected_sha256']:
            raise ValueError('Compared model identity changed')
        if comparison['models'][name]['weights_sha256'] != models[name]['expected_sha256']:
            raise ValueError('Incomplete comparison result')
    run = Path(models['scratch_v14_last']['weights']).parents[1]
    meta_path = run / 'experiment_meta.json'
    if sha256(meta_path) != comparison['plan']['source_experiment_meta_sha256']:
        raise ValueError('Scratch completion identity changed')
    meta = read(meta_path)
    if meta['config']['epochs'] != 100 or meta['primary_checkpoint_screen'] != PRIMARY:
        raise ValueError('Scratch schedule not completed as declared')
    for name, digest in meta['completed_audits'].items():
        if sha256(run / name) != digest:
            raise ValueError('Scratch completed audit changed')
    if len(read(run / 'full_network_update_audit.json')) != 100:
        raise ValueError('Incomplete full-network training')
    return dict(comparison_sha256=expected, scratch_experiment_meta_sha256=sha256(meta_path))


class TeacherAudit:
    def __init__(self, core, preserve_paths, out):
        self.out = out
        self.initial = {k: v.detach().cpu().clone() for k, v in core.state_dict().items()}
        self.blocks = len(core.model) - 1
        self.freeze = FrozenBackboneAudit(core, self.blocks, out / 'feature_audit.json')
        self.teacher = deepcopy(core.model[-1]).eval().requires_grad_(False)
        self.teacher_reference = {k: v.detach().cpu().clone() for k, v in self.teacher.state_dict().items()}
        self.bn = {k: v for k, v in self.initial.items() if k.startswith(f'model.{self.blocks}.')
                   and any(k.endswith(s) for s in ('running_mean', 'running_var', 'num_batches_tracked'))}
        self.preserve = preserve_paths
        self.cadence = OptimizerCadenceAudit(out / 'optimizer_cadence_audit.json')
        self.epochs = []
        self.criterion = None

    def on_start(self, trainer):
        if any(not torch.equal(v.detach().cpu(), self.initial[k]) for k, v in trainer.model.state_dict().items()):
            raise RuntimeError('Actual v13 initial state differs')
        self.freeze.on_start(trainer)
        if trainer.accumulate != 1:
            raise RuntimeError('Unexpected accumulation')
        self.cadence.attach(trainer.optimizer)
        self.criterion = FireTeacherLoss(trainer.model, self.teacher.to(trainer.device), self.preserve)
        trainer.model.criterion = self.criterion
        write(self.out / 'actual_initialization_audit.json',
              dict(exact_selected_v13_state_equal=True, actual_state_sha256=state_digest(self.initial),
                   trainable_parameters=sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)))

    def on_batch(self, trainer):
        self.freeze.on_batch_start(trainer)
        for module in trainer.model.model[-1].modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()

    def epoch_end(self, trainer):
        self.freeze.on_epoch_end(trainer)
        if any(not torch.equal(v.detach().cpu(), self.teacher_reference[k]) for k, v in self.teacher.state_dict().items()):
            raise RuntimeError('Teacher state changed')
        if any(p.requires_grad or p.grad is not None for p in self.teacher.parameters()):
            raise RuntimeError('Teacher received gradient')
        live = trainer.model.state_dict()
        if any(not torch.equal(live[k].detach().cpu(), v) for k, v in self.bn.items()):
            raise RuntimeError('Head BN changed')
        ema = trainer.ema.ema.state_dict()
        with torch.no_grad():
            for k, v in {**self.freeze.reference, **self.bn}.items():
                ema[k].copy_(v.to(ema[k]))
                if not torch.equal(ema[k].detach().cpu(), v):
                    raise RuntimeError('EMA fixed state changed')
        values = self.criterion.samples
        if not values:
            raise RuntimeError('Teacher loss not evaluated')
        self.epochs.append(dict(epoch=trainer.epoch + 1, batches=len(values),
                                mean_cls_kl=sum(v[0] for v in values) / len(values),
                                mean_box_kl=sum(v[1] for v in values) / len(values),
                                preserved_images=sum(v[2] for v in values), total_images=sum(v[3] for v in values),
                                teacher_unchanged=True, feature_and_head_bn_ema_unchanged=True))
        self.criterion.samples.clear()
        write(self.out / 'teacher_audit.json', self.epochs)
        self.cadence.on_epoch_end(trainer)


def rehearse(model, rows, data, audit, config, out):
    """One discarded CPU batch through real YOLO transforms/loss, not a candidate training run."""
    from ultralytics.cfg import get_cfg
    from ultralytics.data.dataset import YOLODataset
    torch.set_num_threads(2)
    buckets = [([r for r in rows if r['visible_flame_train_revision']], 2),
               ([r for r in rows if r['retained_parent_revision']], 2),
               ([r for r in rows if r['teacher_preserve']], 2),
               ([r for r in rows if r['origin'] != 'legacy_train' and r.get('boxes_xyxy')], 1),
               ([r for r in rows if r['origin'] != 'legacy_train' and not r.get('boxes_xyxy')], 1)]
    selected = []
    for bucket, count in buckets:
        if len(bucket) < count:
            raise ValueError('Missing required rehearsal group')
        selected.extend(sorted(bucket, key=lambda r: r['image'])[:count])
    subset = out / 'cpu_rehearsal_paths.txt'
    subset.parent.mkdir(parents=True)
    subset.write_text('\n'.join(str(data.parent / 'images/train' / r['image']) for r in selected) + '\n', encoding='utf-8')
    dataset = YOLODataset(img_path=str(subset), data={'nc': 1, 'names': {0: 'fire'}},
                          imgsz=640, batch_size=8, augment=False, rect=False, cache=False)
    batch = dataset.collate_fn([dataset[i] for i in range(len(dataset))])
    if len(batch['im_file']) != 8:
        raise RuntimeError('Wrong real rehearsal batch size')
    batch['img'] = batch['img'].float() / 255
    core = model.model.cpu().train()
    core.args = get_cfg(overrides=config)
    for name, parameter in core.named_parameters():
        parameter.requires_grad_(not is_backbone_key(name, audit.blocks) and '.dfl.' not in name)
    for name, module in core.named_modules():
        if is_backbone_key(name, audit.blocks) and isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
    optimizer = torch.optim.AdamW([p for p in core.parameters() if p.requires_grad], lr=.0001, weight_decay=.0005)
    trainer = SimpleNamespace(model=core, optimizer=optimizer, device=torch.device('cpu'),
                              accumulate=1, epoch=0, ema=SimpleNamespace(ema=deepcopy(core)))
    try:
        audit.on_start(trainer)
        audit.on_batch(trainer)
        loss, items = core.loss(batch)
        if not torch.isfinite(loss).all():
            raise RuntimeError('Nonfinite real-data loss')
        loss.sum().backward()
        head_gradients = [p.grad for p in core.model[-1].parameters() if p.requires_grad]
        if not head_gradients or not all(g is not None and torch.isfinite(g).all() for g in head_gradients):
            raise RuntimeError('Missing/nonfinite head gradients')
        if not any(float(g.abs().sum()) > 0 for g in head_gradients):
            raise RuntimeError('Head receives no learning gradient')
        if any(p.grad is not None for n, p in core.named_parameters() if is_backbone_key(n, audit.blocks)):
            raise RuntimeError('Frozen features received gradient')
        optimizer.step()
        audit.cadence.on_batch_end(trainer)
        audit.epoch_end(trainer)
        changed = [k for k, v in core.state_dict().items() if not torch.equal(v.detach().cpu(), audit.initial[k])]
        if not changed or any(is_backbone_key(k, audit.blocks) or k in audit.bn for k in changed):
            raise RuntimeError('Wrong actual CPU update scope')
        record = dict(role='Discarded CPU rehearsal; no checkpoint or accuracy claim', candidate_training=False,
                      batch_images=8, augmented=False, loss=[float(v) for v in items.detach()],
                      actual_optimizer_steps=len(audit.cadence.steps), changed_head_state_tensors=len(changed),
                      fixed_feature_and_bn_equal=True, teacher_unchanged=True,
                      teacher_preserved_images=audit.epochs[0]['preserved_images'],
                      images=[dict(image=r['image'], image_sha256=r['sha256'], label_sha256=r['label_sha256'],
                                   teacher_preserve=r['teacher_preserve']) for r in selected])
        if record['teacher_preserved_images'] != 2 or record['actual_optimizer_steps'] != 1:
            raise RuntimeError('Real teacher mask/update differs')
        write(out / 'cpu_rehearsal_complete.json', record)
        print(json.dumps({k: v for k, v in record.items() if k != 'images'}, indent=2), flush=True)
    finally:
        audit.cadence.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--expected-manifest-sha256', required=True)
    p.add_argument('--start', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--rehearse-cpu', action='store_true')
    p.add_argument('--comparison', type=Path)
    p.add_argument('--expected-comparison-sha256')
    a = p.parse_args()
    out = a.out.resolve()
    marker = out.parent / (out.name + '_preflight.json')
    if out.exists() or marker.exists():
        raise SystemExit('Existing output/preflight; do not restart')
    prerequisite = None if a.rehearse_cpu else completed_comparison(a.comparison, a.expected_comparison_sha256)
    if not a.rehearse_cpu and not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable')
    if sha256(a.start) != START_SHA:
        raise SystemExit('Selected v13 checkpoint differs')
    checked = verify(a.data.parent, a.expected_manifest_sha256)
    rows = read(a.data.parent / 'build_manifest.json')['lineage']
    preserve = [a.data.parent / 'images/train' / r['image'] for r in rows if r['teacher_preserve']]
    model = YOLO(str(a.start))
    if model.names != {0: 'fire'}:
        raise SystemExit('Wrong source classes')
    audit = TeacherAudit(model.model, preserve, out)
    config = schedule(a.data, out, audit.blocks)
    identity = dict(role='Semantic source-quarantine pilot from selected v13; fixed features and head BN; exposed development',
                    source_sha256=START_SHA, dataset=checked, config=config,
                    checkpoint_screen=PRIMARY, teacher_state_sha256=state_digest(audit.teacher_reference),
                    teacher_loss=dict(class_gain=1., box_bin_gain=1., box_teacher_confidence=.05,
                                      applies_to='1332 unchanged retained legacy images; all revised and derived images exempt'),
                    scratch_comparison_prerequisite=prerequisite,
                    candidate_training=not a.rehearse_cpu, script_sha256=sha256(Path(__file__)),
                    dependencies={n: sha256(Path(__file__).with_name(n)) for n in DEPENDENCIES},
                    environment=dict(python=platform.python_version(), torch=torch.__version__,
                                     ultralytics=ultralytics.__version__, device='cpu' if a.rehearse_cpu else torch.cuda.get_device_name(0)))
    if a.rehearse_cpu:
        rehearse(model, rows, a.data.resolve(), audit, config, out)
        write(out / 'cpu_rehearsal_identity.json', identity)
        return
    marker.write_text(json.dumps(identity, indent=2) + '\n', encoding='utf-8')
    model.add_callback('on_train_start', audit.on_start)
    model.add_callback('on_train_batch_start', audit.on_batch)
    model.add_callback('on_train_batch_end', audit.cadence.on_batch_end)
    model.add_callback('on_train_epoch_end', audit.epoch_end)
    try:
        model.train(**config)
        audit.freeze.verify_state(model.trainer, 'train_complete')
        if len(audit.epochs) != 12:
            raise RuntimeError('Incomplete fixed pilot schedule')
        if not any(not torch.equal(v.detach().cpu(), audit.initial[k]) for k, v in model.trainer.model.state_dict().items()):
            raise RuntimeError('No learned state changed')
    finally:
        audit.cadence.close()
    identity['checkpoints'] = {w.name: sha256(w) for w in (out / 'weights').glob('*.pt')}
    if set(identity['checkpoints']) != set(PRIMARY):
        raise RuntimeError('Unexpected primary checkpoint inventory')
    identity['actual_optimizer_steps'] = len(audit.cadence.steps)
    identity['actual_batches'] = audit.cadence.batches
    identity['completed_audits'] = {n: sha256(out / n) for n in
                                     ('actual_initialization_audit.json', 'teacher_audit.json', 'feature_audit.json', 'optimizer_cadence_audit.json')}
    write(out / 'experiment_meta.json', identity)
    print(json.dumps(identity, indent=2), flush=True)


if __name__ == '__main__':
    main()
