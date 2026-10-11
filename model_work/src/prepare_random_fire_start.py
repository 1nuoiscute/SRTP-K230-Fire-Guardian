"""Create an architecture-matched, fully random initial state; copy no learned weights."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
import yaml
from backbone_freeze_audit import state_digest
from data_integrity import sha256


def random_model(config, seed):
    # Isolate the initializer RNG from training/augmentation RNG consumption.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = DetectionModel(deepcopy(config), ch=3, nc=1, verbose=False)
    model.names = {0: 'fire'}
    return model


def prepare(reference, expected_reference_sha, out, seed):
    if out.exists() or sha256(reference) != expected_reference_sha:
        raise ValueError('Output exists or architecture reference changed')
    source = YOLO(str(reference)).model.float().cpu()
    config = deepcopy(source.yaml)
    config.pop('yaml_file', None)
    config['nc'] = 1
    if config.get('scale') != 's':
        raise ValueError('Expected original YOLO11s architecture')
    initial = random_model(config, seed)
    state, original = initial.state_dict(), source.state_dict()
    if state.keys() != original.keys() or any(state[k].shape != original[k].shape for k in state):
        raise ValueError('Random model architecture differs from reference')
    convolution_keys = [k for k, v in state.items() if v.ndim == 4 and k.endswith('.weight') and '.dfl.' not in k]
    if not convolution_keys or any(torch.equal(state[k], original[k]) for k in convolution_keys):
        raise ValueError('A learned convolution was inherited instead of initialized')
    repeat = random_model(config, seed).state_dict()
    if any(not torch.equal(v, repeat[k]) for k, v in state.items()):
        raise ValueError('Initializer is not reproducible')
    out.mkdir(parents=True)
    arch = out / 'yolo11s_fire_scratch.yaml'
    arch.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    checkpoint = out / 'random_initial.pt'
    torch.save(dict(model=deepcopy(initial).float(), epoch=-1, optimizer=None,
                    train_args=dict(task='detect', imgsz=640), version=ultralytics.__version__,
                    date=datetime.now(timezone.utc).isoformat()), checkpoint)
    loaded = YOLO(str(checkpoint)).model.float().cpu().state_dict()
    if any(not torch.equal(v, loaded[k]) for k, v in state.items()):
        raise ValueError('Random initial artifact changed during serialization')
    meta = dict(role='Random initial state; reference checkpoint contributes architecture only',
                seed=seed, reference_sha256=expected_reference_sha,
                architecture_sha256=sha256(arch), random_initial_sha256=sha256(checkpoint),
                initial_state_sha256=state_digest(state), random_convolutions=len(convolution_keys),
                all_random_convolutions_differ_from_reference=True, reproducible_initial_state=True,
                parameters=sum(p.numel() for p in initial.parameters()),
                script_sha256=sha256(Path(__file__)), torch=torch.__version__, ultralytics=ultralytics.__version__)
    (out / 'initialization.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--expected-reference-sha256', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=20261003)
    a = p.parse_args()
    prepare(a.reference.resolve(), a.expected_reference_sha256, a.out.resolve(), a.seed)
