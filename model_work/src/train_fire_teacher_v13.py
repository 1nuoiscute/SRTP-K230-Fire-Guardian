"""Train only the original fire head; preserve feature and BN state and audit distillation."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import platform
import torch
import ultralytics
from ultralytics import YOLO
from backbone_freeze_audit import FrozenBackboneAudit, state_digest
from data_integrity import sha256
from fire_label_revision_dataset import verify, read
from fire_teacher_loss import FireTeacherLoss
from optimizer_cadence_audit import OptimizerCadenceAudit


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--expected-manifest-sha256', required=True)
    p.add_argument('--start', type=Path, required=True)
    p.add_argument('--expected-start-sha256', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    out = a.out.resolve()
    marker = out.parent / (out.name + '_preflight.json')
    if out.exists() or marker.exists():
        raise SystemExit('Existing run/preflight; do not restart')
    if sha256(a.start) != a.expected_start_sha256 or not torch.cuda.is_available():
        raise SystemExit('CUDA or source identity unavailable')
    dataset = verify(a.data.parent, a.expected_manifest_sha256)
    rows = read(a.data.parent / 'build_manifest.json')['lineage']
    preserve = [a.data.parent / 'images/train' / r['image'] for r in rows if r['teacher_preserve']]
    model = YOLO(str(a.start))
    if model.names != {0: 'fire'}:
        raise SystemExit('Wrong source classes')
    initial = {k: v.detach().cpu().clone() for k, v in model.model.state_dict().items()}
    blocks = len(model.model.model) - 1
    freeze = FrozenBackboneAudit(model.model, blocks, out / 'feature_audit.json')
    teacher = deepcopy(model.model.model[-1]).eval().requires_grad_(False)
    teacher_reference = {k: v.detach().cpu().clone() for k, v in teacher.state_dict().items()}
    bn_reference = {k: v for k, v in initial.items() if k.startswith(f'model.{blocks}.') and any(k.endswith(s) for s in ('running_mean', 'running_var', 'num_batches_tracked'))}
    cadence = OptimizerCadenceAudit(out / 'optimizer_cadence_audit.json')
    epochs = []
    criterion = None

    def on_start(trainer):
        nonlocal criterion
        if any(not torch.equal(v.detach().cpu(), initial[k]) for k, v in trainer.model.state_dict().items()):
            raise RuntimeError('Actual initial weights differ')
        freeze.on_start(trainer)
        if trainer.accumulate != 1:
            raise RuntimeError('Unexpected accumulation')
        cadence.attach(trainer.optimizer)
        criterion = FireTeacherLoss(trainer.model, teacher.to(trainer.device), preserve)
        trainer.model.criterion = criterion

    def on_batch(trainer):
        freeze.on_batch_start(trainer)
        for module in trainer.model.model[-1].modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()

    def epoch_end(trainer):
        freeze.on_epoch_end(trainer)
        for k, v in teacher.state_dict().items():
            if not torch.equal(v.detach().cpu(), teacher_reference[k]) or v.requires_grad:
                raise RuntimeError('Teacher state changed')
        live = trainer.model.state_dict()
        if any(not torch.equal(live[k].detach().cpu(), v) for k, v in bn_reference.items()):
            raise RuntimeError('Head BatchNorm buffers changed')
        ema = trainer.ema.ema.state_dict()
        with torch.no_grad():
            for k, v in {**freeze.reference, **bn_reference}.items():
                ema[k].copy_(v.to(ema[k]))
                if not torch.equal(ema[k].detach().cpu(), v):
                    raise RuntimeError('EMA fixed state changed')
        values = criterion.samples
        if not values:
            raise RuntimeError('Teacher loss was never evaluated')
        epochs.append(dict(epoch=trainer.epoch + 1, batches=len(values),
                           mean_cls_kl=sum(v[0] for v in values) / len(values),
                           mean_box_kl=sum(v[1] for v in values) / len(values),
                           preserved_images=sum(v[2] for v in values), total_images=sum(v[3] for v in values),
                           teacher_unchanged=True, feature_and_head_bn_ema_unchanged=True))
        criterion.samples.clear()
        (out / 'teacher_audit.json').write_text(json.dumps(epochs, indent=2) + '\n', encoding='utf-8')
        cadence.on_epoch_end(trainer)

    model.add_callback('on_train_start', on_start)
    model.add_callback('on_train_batch_start', on_batch)
    model.add_callback('on_train_batch_end', cadence.on_batch_end)
    model.add_callback('on_train_epoch_end', epoch_end)
    schedule = dict(data=str(a.data.resolve()), epochs=12, patience=0, imgsz=640, batch=8, nbs=8,
                    device=0, workers=0, seed=20261002, optimizer='AdamW', lr0=.0001, lrf=.1,
                    weight_decay=.0005, warmup_epochs=0, freeze=blocks, hsv_h=.015, hsv_s=.3,
                    hsv_v=.25, degrees=3, translate=.05, scale=.2, fliplr=.5, mosaic=0, mixup=0,
                    project=str(out.parent), name=out.name, exist_ok=False, save=True, save_period=3,
                    plots=True, val=True)
    identity = dict(role='Fixed v3 features; fire head only; legacy teacher retention; exposed development',
                    source_sha256=sha256(a.start), script_sha256=sha256(Path(__file__)), dataset=dataset,
                    config=schedule, teacher_state_sha256=state_digest(teacher_reference),
                    teacher_loss=dict(class_gain=1., box_bin_gain=1., box_teacher_confidence=.05,
                                      applies_to='Only unchanged legacy train images; revised/derived labels exempt'),
                    checkpoint_screen=['best.pt', 'last.pt', 'epoch3.pt', 'epoch6.pt', 'epoch9.pt'],
                    environment=dict(python=platform.python_version(), torch=torch.__version__,
                                     ultralytics=ultralytics.__version__, gpu=torch.cuda.get_device_name(0)),
                    dependencies={n: sha256(Path(__file__).with_name(n)) for n in ('fire_teacher_loss.py', 'fire_label_revision_dataset.py', 'backbone_freeze_audit.py', 'optimizer_cadence_audit.py', 'dataset_preflight.py', 'data_integrity.py')})
    marker.write_text(json.dumps(identity, indent=2) + '\n', encoding='utf-8')
    try:
        model.train(**schedule)
        freeze.verify_state(model.trainer, 'train_complete')
    finally:
        cadence.close()
    identity['checkpoints'] = {w.name: sha256(w) for w in (out / 'weights').glob('*.pt')}
    identity['actual_optimizer_steps'] = len(cadence.steps)
    identity['actual_batches'] = cadence.batches
    (out / 'experiment_meta.json').write_text(json.dumps(identity, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(identity, indent=2), flush=True)


if __name__ == '__main__':
    main()
