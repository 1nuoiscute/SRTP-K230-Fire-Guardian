"""Verify saved checkpoints retain frozen feature and head-BN state, not just live training state."""
import argparse
import json
from pathlib import Path
import torch
from ultralytics import YOLO
from backbone_freeze_audit import is_backbone_key, state_digest
from data_integrity import sha256


def verify(source, run, out):
    if out.exists():
        raise ValueError('Existing verification result')
    meta = json.loads((run / 'experiment_meta.json').read_text())
    if sha256(source) != meta['source_sha256']:
        raise ValueError('Source checkpoint differs')
    for name, digest in meta['dependencies'].items():
        if sha256(Path(__file__).with_name(name)) != digest:
            raise ValueError('Training dependency differs: ' + name)
    if sha256(Path(__file__).with_name('train_fire_teacher_v13.py')) != meta['script_sha256']:
        raise ValueError('Training script differs')
    model = YOLO(str(source)).model.float().cpu()
    blocks = len(model.model) - 1
    reference = model.state_dict()
    fixed = {k: v for k, v in reference.items() if is_backbone_key(k, blocks) or
             (k.startswith(f'model.{blocks}.') and any(k.endswith(s) for s in ('running_mean', 'running_var', 'num_batches_tracked')))}
    result = dict(source_sha256=sha256(source), script_sha256=sha256(Path(__file__)), fixed_tensors=len(fixed), models={})
    for name in meta['checkpoint_screen']:
        path = run / 'weights' / name
        if sha256(path) != meta['checkpoints'][name]:
            raise ValueError('Saved checkpoint differs')
        candidate = YOLO(str(path)).model.float().cpu()
        state = candidate.state_dict()
        if candidate.names != {0: 'fire'} or state.keys() != reference.keys():
            raise ValueError('Saved model architecture changed')
        if any(not torch.equal(v, state[k]) for k, v in fixed.items()):
            raise ValueError('Saved frozen feature/BN state changed')
        changed = [k for k in state if not torch.equal(state[k], reference[k])]
        if not changed or any(not k.startswith(f'model.{blocks}.') for k in changed):
            raise ValueError('Changes are not confined to the fire head')
        result['models'][name] = dict(checkpoint_sha256=sha256(path), model_state_sha256=state_digest(state),
                                    fixed_state_equal=True, changed_head_tensors=len(changed),
                                    parameters=sum(p.numel() for p in candidate.parameters()))
    features = json.loads((run / 'feature_audit.json').read_text())
    teacher = json.loads((run / 'teacher_audit.json').read_text())
    cadence = json.loads((run / 'optimizer_cadence_audit.json').read_text())
    if len(teacher) != meta['config']['epochs'] or not all(r['teacher_unchanged'] and r['feature_and_head_bn_ema_unchanged'] for r in teacher):
        raise ValueError('Incomplete teacher/EMA audit')
    if not all(r['unchanged'] for r in features['checks']) or features['checks'][-1]['stage'] != 'train_complete':
        raise ValueError('Incomplete live feature audit')
    if cadence['batches'] != meta['actual_batches'] or cadence['actual_optimizer_steps'] != meta['actual_optimizer_steps']:
        raise ValueError('Optimizer cadence differs')
    result.update(live_checks=len(features['checks']), teacher_and_ema_epochs=len(teacher),
                  actual_batches=cadence['batches'], actual_optimizer_steps=cadence['actual_optimizer_steps'],
                  best_state_equal_epoch3=result['models']['best.pt']['model_state_sha256'] == result['models']['epoch3.pt']['model_state_sha256'])
    out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    verify(a.source.resolve(), a.run.resolve(), a.out.resolve())
