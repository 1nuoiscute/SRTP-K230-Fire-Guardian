"""Partial-feature smoke candidate: train modules 10-22 plus head on the extension manifest.

Same data, same start point, and same 40-epoch schedule as train_smoke_extension_v5,
but instead of a fully frozen shared feature stack, feature modules 0-9 stay frozen
while modules 10-22 train. All BatchNorm2d running statistics and counters across
feature modules 0-22 remain fixed (feature BN affine in 10-22 is learnable), and the
head DFL is fixed. This is an independent smoke candidate; it is not the fire model's
shared backbone and must not replace the old dual-head packaging.
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
from train_smoke_extension_v5 import verify_extension

FEATURE_BLOCKS = 23  # modules 0-22 are feature; module 23 is the head
FROZEN_PREFIX = 10   # modules 0-9 stay fully frozen


def module_index(name):
    parts = name.split('.')
    if len(parts) > 1 and parts[0] == 'model' and parts[1].isdigit():
        return int(parts[1])
    return None


def is_feature_key(name):
    idx = module_index(name)
    return idx is not None and idx < FEATURE_BLOCKS


def is_head_key(name):
    idx = module_index(name)
    return idx is not None and idx >= FEATURE_BLOCKS


def is_dfl_key(name):
    return '.dfl.' in name


def main():
    p = argparse.ArgumentParser(); p.add_argument('--data', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True); p.add_argument('--start', type=Path, required=True)
    p.add_argument('--expected-start-sha256', required=True); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); out = a.out.resolve(); marker = out.parent / (out.name + '_preflight.json')
    if out.exists() or marker.exists(): raise SystemExit('Existing run/preflight; no restart')
    if not torch.cuda.is_available() or sha256(a.start) != a.expected_start_sha256:
        raise SystemExit('CUDA/start identity unavailable')
    manifest = verify_extension(a.data, a.manifest)
    group_split = {}; image_count = 0
    for row in manifest['images']:
        if row['split'] not in ('train', 'val'): raise SystemExit('Non train/val split: ' + row['file'])
        if sha256(Path(row['file'])) != row['image_sha256']: raise SystemExit('Image identity changed: ' + row['file'])
        if sha256(Path(row['label'])) != row['label_sha256']: raise SystemExit('Label identity changed: ' + row['file'])
        if group_split.setdefault(row['group'], row['split']) != row['split']:
            raise SystemExit('Group crosses splits: ' + row['group'])
        image_count += 1
    dataset = {'manifest_sha256': sha256(a.manifest), 'image_count': image_count,
               'manifest_role': 'custom extension manifest; v4 val members unchanged, new samples train-only'}

    model = YOLO(str(a.start))
    if model.names != {0: 'smoke'}: raise SystemExit('Single-class smoke names changed')
    initial = {k: v.detach().cpu().clone() for k, v in model.model.state_dict().items()}

    # Learnable-vs-fixed partition of feature keys: BN affine inside modules 10-22 trains.
    bn_affine_trainable = set()
    for name, module in model.model.named_modules():
        idx = module_index(name)
        if idx is not None and FROZEN_PREFIX <= idx < FEATURE_BLOCKS and isinstance(module, torch.nn.BatchNorm2d):
            bn_affine_trainable.add(name + '.weight'); bn_affine_trainable.add(name + '.bias')
    for name, param in model.model.named_parameters():
        if is_dfl_key(name): param.requires_grad_(False)
    fixed_keys = {k for k in initial if (is_feature_key(k) and module_index(k) < FROZEN_PREFIX)
                  or (is_feature_key(k) and k.endswith(('running_mean', 'running_var', 'num_batches_tracked')))
                  or is_dfl_key(k)}
    late_feature_keys = {k for k in initial
                         if is_feature_key(k) and module_index(k) >= FROZEN_PREFIX and k.endswith('.weight') and k not in fixed_keys}
    head_keys = {k for k in initial if is_head_key(k) and k.endswith('.weight') and not is_dfl_key(k)}
    fixed_reference = {k: initial[k] for k in fixed_keys}

    freeze = FrozenBackboneAudit(model.model, FROZEN_PREFIX, out / 'frozen_prefix_audit.json')
    cadence = OptimizerCadenceAudit(out / 'optimizer_cadence_audit.json'); ema_checks = []

    def feature_bn_modules(trainer):
        return [(n, m) for n, m in trainer.model.named_modules()
                if is_feature_key(n) and isinstance(m, torch.nn.BatchNorm2d)]

    def initial_check(trainer):
        state = trainer.model.state_dict()
        if state.keys() != initial.keys() or any(
                not torch.equal(value.detach().cpu(), initial[key]) for key, value in state.items()):
            raise RuntimeError('Actual initial state differs')
        freeze.on_start(trainer)
        late_grad = [n for n, prm in trainer.model.named_parameters()
                     if module_index(n) is not None and FROZEN_PREFIX <= module_index(n) < FEATURE_BLOCKS
                     and prm.requires_grad]
        head_grad = [n for n, prm in trainer.model.named_parameters() if is_head_key(n) and prm.requires_grad]
        dfl_grad = [n for n, prm in trainer.model.named_parameters() if is_dfl_key(n) and prm.requires_grad]
        if not late_grad or not head_grad or dfl_grad:
            raise RuntimeError(f'Partial-feature mask invalid: late={len(late_grad)}, head={len(head_grad)}, dfl_grad={dfl_grad[:5]}')
        for n, m in feature_bn_modules(trainer):
            if FROZEN_PREFIX <= module_index(n) < FEATURE_BLOCKS:
                for affine in ('weight', 'bias'):
                    if not getattr(m, affine).requires_grad:
                        raise RuntimeError('Late feature BN affine is not learnable: ' + n)
        if trainer.accumulate != 1: raise RuntimeError('Unexpected accumulation')
        cadence.attach(trainer.optimizer)

    def batch_start(trainer):
        freeze.on_batch_start(trainer)
        # Hold all feature BN running statistics fixed for this batch.
        for _, m in feature_bn_modules(trainer):
            m.eval()

    def epoch_end(trainer):
        freeze.on_epoch_end(trainer)
        state = trainer.model.state_dict()
        # All fixed feature BN buffers and the DFL must be unchanged.
        for key in fixed_keys:
            if not torch.equal(state[key].detach().cpu(), fixed_reference[key]):
                raise RuntimeError('Fixed feature/DFL tensor changed: ' + key)
        late_changed = [k for k in late_feature_keys
                        if not torch.equal(state[k].detach().cpu(), fixed_reference.get(k, initial[k]))]
        head_changed = [k for k in head_keys if not torch.equal(state[k].detach().cpu(), initial[k])]
        if not late_changed: raise RuntimeError('No late feature parameter actually changed')
        if not head_changed: raise RuntimeError('No head parameter actually changed')
        ema_state = trainer.ema.ema.state_dict()
        with torch.no_grad():
            for key in fixed_keys:
                ema_state[key].copy_(fixed_reference[key].to(device=ema_state[key].device,
                                                             dtype=ema_state[key].dtype))
                if not torch.equal(ema_state[key].detach().cpu(), fixed_reference[key]):
                    raise RuntimeError('EMA fixed feature/DFL differs')
        ema_checks.append({'epoch': trainer.epoch + 1, 'fixed_features_equal': True,
                           'late_feature_changed_tensors': len(late_changed),
                           'head_changed_tensors': len(head_changed)})
        (out / 'fixed_ema_audit.json').write_text(json.dumps(ema_checks, indent=2) + '\n', encoding='utf-8')
        cadence.on_epoch_end(trainer)

    model.add_callback('on_train_start', initial_check)
    model.add_callback('on_train_batch_start', batch_start)
    model.add_callback('on_train_batch_end', cadence.on_batch_end)
    model.add_callback('on_train_epoch_end', epoch_end)

    def capture_sources(trainer):
        if Path(trainer.save_dir).resolve() != out: raise ValueError('Trainer changed output directory')
        snapshot = out / 'source_snapshot'; snapshot.mkdir()
        for name in [Path(__file__).name, 'backbone_freeze_audit.py', 'optimizer_cadence_audit.py',
                     'data_integrity.py', 'train_smoke_extension_v5.py']:
            shutil.copyfile(Path(__file__).with_name(name), snapshot / name)
    model.add_callback('on_train_start', capture_sources)

    schedule = dict(data=str(a.data.resolve()), epochs=40, patience=0, imgsz=640, batch=4, nbs=4, device=0, workers=0,
                    seed=20261002, optimizer='AdamW', lr0=.0005, lrf=.1, weight_decay=.0005,
                    warmup_epochs=0, warmup_bias_lr=.001, freeze=FROZEN_PREFIX, hsv_h=.015, hsv_s=.3, hsv_v=.25,
                    degrees=3, translate=.05, scale=.2, fliplr=.5, mosaic=.2, mixup=0, close_mosaic=5,
                    project=str(out.parent), name=out.name, exist_ok=False, save=True, plots=True, val=True)

    identity = {'role': 'Partial-feature smoke candidate (modules 10-22 + head trainable, feature BN stats and DFL fixed); '
                        'independent of the fire dual-head model; exposed development only, no test/on-device acceptance',
                'script_sha256': sha256(Path(__file__)), 'data_config_sha256': sha256(a.data),
                'manifest_sha256': sha256(a.manifest), 'dataset': dataset,
                'start_sha256': sha256(a.start), 'config': schedule,
                'frozen_prefix_plan': freeze.plan,
                'partial_feature_plan': {
                    'frozen_prefix_modules': FROZEN_PREFIX, 'feature_modules': FEATURE_BLOCKS,
                    'fixed_tensor_count': len(fixed_keys),
                    'fixed_scope': 'modules 0-9 all tensors; all feature BN running buffers; DFL',
                    'learnable_scope': 'modules 10-22 parameters and BN affine; head module 23 except DFL',
                    'late_feature_trainable_tensors': len(late_feature_keys),
                    'head_trainable_tensors': len(head_keys)},
                'labels_note': 'Only visible smoke appearance; unknown cooking plumes not assigned negatives; no fire-event truth claimed.',
                'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                                'ultralytics': ultralytics.__version__,
                                'gpu': torch.cuda.get_device_name(0)},
                'local_dependency_sha256': {name: sha256(Path(__file__).with_name(name)) for name in
                                            ('backbone_freeze_audit.py', 'optimizer_cadence_audit.py',
                                             'data_integrity.py', 'train_smoke_extension_v5.py')}}
    marker.write_text(json.dumps(identity, indent=2), encoding='utf-8')
    try:
        model.train(**schedule); freeze.verify_state(model.trainer, 'train_complete')
    finally: cadence.close()

    identity['completed_frozen_prefix_audit_sha256'] = sha256(out / 'frozen_prefix_audit.json')
    identity['completed_fixed_ema_sha256'] = sha256(out / 'fixed_ema_audit.json')
    identity['completed_optimizer_cadence_sha256'] = sha256(out / 'optimizer_cadence_audit.json')
    identity['actual_optimizer_steps'] = len(cadence.steps); identity['actual_batches'] = cadence.batches
    for name in ('best.pt', 'last.pt'):
        identity[name + '_sha256'] = sha256(out / 'weights' / name)
        state = YOLO(str(out / 'weights' / name)).model.state_dict()
        if any(not torch.equal(state[k].detach().cpu(), fixed_reference[k]) for k in fixed_keys):
            raise RuntimeError('Saved fixed feature/DFL changed: ' + name)
        late_changed = sum(1 for k in late_feature_keys if not torch.equal(state[k].detach().cpu(), initial[k]))
        head_changed = sum(1 for k in head_keys if not torch.equal(state[k].detach().cpu(), initial[k]))
        if not late_changed or not head_changed:
            raise RuntimeError('Saved checkpoint lacks late-feature/head changes: ' + name)
        identity[name + '_fixed_equal'] = True
        identity[name + '_late_feature_changed_tensors'] = late_changed
        identity[name + '_head_changed_tensors'] = head_changed
    (out / 'experiment_meta.json').write_text(json.dumps(identity, indent=2), encoding='utf-8')
    print(json.dumps(identity, indent=2))


if __name__ == '__main__':
    main()
