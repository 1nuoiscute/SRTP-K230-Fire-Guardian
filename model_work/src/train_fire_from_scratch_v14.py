"""Full-network random-initialization fire experiment with actual initialization/update audits."""
import argparse
import json
from pathlib import Path
import platform
from copy import deepcopy
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from backbone_freeze_audit import is_backbone_key, state_digest
from data_integrity import sha256
from fire_label_revision_dataset import verify, read
from optimizer_cadence_audit import OptimizerCadenceAudit
from prepare_random_fire_start import random_model
import yaml


class RandomOnlyTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        if weights is not None or self.args.pretrained is not False or self.args.resume:
            raise RuntimeError('Scratch route rejects all supplied weights and resume')
        config = yaml.safe_load(Path(self.args.model).read_text(encoding='utf-8'))
        if self.data['nc'] != 1 or self.data['names'] != {0: 'fire'}:
            raise RuntimeError('Scratch task classes changed')
        self.random_construction_calls = getattr(self, 'random_construction_calls', 0) + 1
        if self.random_construction_calls != 1:
            raise RuntimeError('Unexpected repeated random construction')
        return random_model(config, self.args.seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--expected-manifest-sha256', required=True)
    p.add_argument('--initial', type=Path, required=True)
    p.add_argument('--expected-initialization-sha256', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    out, initial = a.out.resolve(), a.initial.resolve()
    marker = out.parent / (out.name + '_preflight.json')
    if out.exists() or marker.exists():
        raise SystemExit('Existing run/preflight; inspect existing process instead of restarting')
    if sha256(initial / 'initialization.json') != a.expected_initialization_sha256 or not torch.cuda.is_available():
        raise SystemExit('Initialization identity or CUDA unavailable')
    init = read(initial / 'initialization.json')
    architecture = initial / 'yolo11s_fire_scratch.yaml'
    checkpoint = initial / 'random_initial.pt'
    if sha256(architecture) != init['architecture_sha256'] or sha256(checkpoint) != init['random_initial_sha256']:
        raise SystemExit('Initialization artifact identity differs')
    checked = verify(a.data.parent, a.expected_manifest_sha256)
    expected = {k: v.detach().cpu().clone() for k, v in YOLO(str(checkpoint)).model.float().state_dict().items()}
    if state_digest(expected) != init['initial_state_sha256']:
        raise SystemExit('Initial state identity differs')
    model = YOLO(str(architecture))
    cadence = OptimizerCadenceAudit(out / 'optimizer_cadence_audit.json')
    epochs = []

    def on_start(trainer):
        actual = trainer.model.state_dict()
        if actual.keys() != expected.keys() or any(not torch.equal(v.detach().cpu(), expected[k]) for k, v in actual.items()):
            raise RuntimeError('Actual scratch initialization differs from exact recorded state')
        frozen = [k for k, v in trainer.model.named_parameters() if not v.requires_grad]
        if frozen != ['model.23.dfl.conv.weight'] or trainer.accumulate != 1 or trainer.random_construction_calls != 1:
            raise RuntimeError('Unexpected frozen parameters, accumulation or initialization route')
        cadence.attach(trainer.optimizer)
        record = dict(actual_initial_state_sha256=state_digest({k: v.detach().cpu() for k, v in actual.items()}),
                      exact_random_initial_state_equal=True, supplied_weights=None, pretrained=False, resume=False,
                      initialization_constructions=trainer.random_construction_calls, frozen_parameters=frozen,
                      trainable_parameters=sum(v.numel() for v in trainer.model.parameters() if v.requires_grad))
        (out / 'actual_initialization_audit.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
        print('RANDOM_INIT_AUDIT: exact recorded random state, no learned weights supplied', flush=True)

    def on_epoch(trainer):
        state = {k: v.detach().cpu() for k, v in trainer.model.state_dict().items()}
        changed = [k for k, v in state.items() if not torch.equal(v, expected[k])]
        blocks = len(trainer.model.model) - 1
        backbone = sum(is_backbone_key(k, blocks) and k.endswith('.weight') for k in changed)
        head = sum(k.startswith(f'model.{blocks}.') and k.endswith('.weight') for k in changed)
        if not backbone or not head:
            raise RuntimeError('Full-network optimizer did not update both features and detection head')
        epochs.append(dict(epoch=trainer.epoch + 1, changed_feature_weight_tensors=backbone,
                           changed_head_weight_tensors=head, model_state_sha256=state_digest(state),
                           cumulative_actual_optimizer_steps=len(cadence.steps)))
        (out / 'full_network_update_audit.json').write_text(json.dumps(epochs, indent=2) + '\n', encoding='utf-8')
        cadence.on_epoch_end(trainer)

    model.add_callback('on_train_start', on_start)
    model.add_callback('on_train_batch_end', cadence.on_batch_end)
    model.add_callback('on_train_epoch_end', on_epoch)
    config = dict(data=str(a.data.resolve()), pretrained=False, resume=False, epochs=100, patience=0,
                  imgsz=640, batch=8, nbs=8, device=0, workers=0, seed=init['seed'],
                  optimizer='AdamW', lr0=.001, lrf=.01, weight_decay=.0005, warmup_epochs=3,
                  warmup_bias_lr=.001, freeze=None, hsv_h=.015, hsv_s=.3, hsv_v=.25,
                  degrees=3, translate=.05, scale=.2, fliplr=.5, mosaic=.5, mixup=0,
                  close_mosaic=10, project=str(out.parent), name=out.name, exist_ok=False,
                  save=True, save_period=10, plots=True, val=True)
    identity = dict(role='Full network from random initialization; exposed development route comparison',
                    initialization=init, initialization_manifest_sha256=a.expected_initialization_sha256,
                    dataset=checked, config=config, script_sha256=sha256(Path(__file__)),
                    primary_checkpoint_screen=['best.pt', 'last.pt'],
                    periodic_snapshots='Zero-based epoch0,10,20,...90 are convergence diagnostics; not independent tests',
                    comparison_limit='Different optimizer/length/augmentation from teacher-head v13; not an initialization-only causal ablation',
                    environment=dict(python=platform.python_version(), torch=torch.__version__,
                                     ultralytics=ultralytics.__version__, gpu=torch.cuda.get_device_name(0)),
                    dependencies={name: sha256(Path(__file__).with_name(name)) for name in
                        ('prepare_random_fire_start.py', 'fire_label_revision_dataset.py', 'optimizer_cadence_audit.py', 'backbone_freeze_audit.py', 'data_integrity.py', 'dataset_preflight.py')})
    marker.write_text(json.dumps(identity, indent=2) + '\n', encoding='utf-8')
    try:
        model.train(trainer=RandomOnlyTrainer, **config)
    finally:
        cadence.close()
    if len(epochs) != config['epochs']:
        raise RuntimeError('Incomplete fixed scratch schedule')
    identity['checkpoints'] = {w.name: sha256(w) for w in (out / 'weights').glob('*.pt')}
    identity['actual_optimizer_steps'] = len(cadence.steps)
    identity['actual_batches'] = cadence.batches
    identity['completed_audits'] = {n: sha256(out / n) for n in
                                    ('actual_initialization_audit.json', 'full_network_update_audit.json', 'optimizer_cadence_audit.json')}
    (out / 'experiment_meta.json').write_text(json.dumps(identity, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(identity, indent=2), flush=True)


if __name__ == '__main__':
    main()
