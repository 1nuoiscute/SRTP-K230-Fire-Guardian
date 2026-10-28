"""Summarize trainable-state displacement between v21 and v22 checkpoints."""
import argparse
import json
from pathlib import Path

from data_integrity import sha256


PAIRS = {
    'best': (
        Path('model_work/runs/fire-kitchen-partial-features-v21-20261028/weights/best.pt'),
        Path('model_work/runs/fire-source-sampling-v22-20261028/weights/best.pt'),
    ),
    'last': (
        Path('model_work/runs/fire-kitchen-partial-features-v21-20261028/weights/last.pt'),
        Path('model_work/runs/fire-source-sampling-v22-20261028/weights/last.pt'),
    ),
}


def layer_index(name):
    parts = name.split('.')
    if len(parts) > 2 and parts[0] == 'model' and parts[1].isdigit():
        return int(parts[1])
    return None


def groups(name):
    layer = layer_index(name)
    parameter = name.endswith(('.weight', '.bias'))
    head = name.startswith('model.23.')
    dfl = '.dfl.' in name
    result = []
    if parameter and not dfl and (layer is None or layer >= 10):
        result.append('trainable')
    if parameter and layer is not None and 10 <= layer < 23:
        result.append('features')
    if parameter and head and not dfl:
        result.append('head')
    if parameter and layer is not None and 10 <= layer < 18:
        result.append('features_10_17')
    if parameter and layer is not None and 18 <= layer < 23:
        result.append('features_18_22')
    return result


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Refusing overwrite')
    result = {'role': 'state-displacement diagnostic; not a performance result', 'pairs': {}}
    for pair, (left_path, right_path) in PAIRS.items():
        left = torch.load(left_path, map_location='cpu', weights_only=False)
        right = torch.load(right_path, map_location='cpu', weights_only=False)
        left_state = (left.get('ema') or left['model']).state_dict()
        right_state = (right.get('ema') or right['model']).state_dict()
        records = {}
        for group in (
            'trainable',
            'features',
            'head',
            'features_10_17',
            'features_18_22',
        ):
            keys = [name for name in left_state if group in groups(name)]
            deltas = [
                (right_state[name].float() - left_state[name].float()).double()
                for name in keys
            ]
            sum_sq = sum(float(value.square().sum()) for value in deltas)
            numel = sum(value.numel() for value in deltas)
            records[group] = {
                'tensors': len(deltas),
                'numel': numel,
                'sum_sq': sum_sq,
                'mean_sq': sum_sq / numel,
                'rms': (sum_sq / numel) ** 0.5,
                'max_abs': max((float(value.abs().max()) for value in deltas), default=0.0),
            }
        result['pairs'][pair] = {
            'left': str(left_path.resolve()),
            'left_sha256': sha256(left_path),
            'right': str(right_path.resolve()),
            'right_sha256': sha256(right_path),
            'groups': records,
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + '\n', encoding='utf-8'
    )
    print(json.dumps(result['pairs'], indent=2))


if __name__ == '__main__':
    main()
