"""Development-only interpolation of v21/v22 layers 10-17 plus the head."""
import argparse
import copy
from pathlib import Path

from build_kitchen_source_extension_v20 import read, write
from checkpoint_blend import architecture
from compare_fire_training_routes import micro_score
from data_integrity import sha256
from screen_fire_soup_v23 import (
    DATA_SHA,
    FEATURES,
    PREFIX,
    BASE,
    NEW_DATA,
    OLD_DATA,
    PAIRS,
    PRIOR,
    REFERENCE_ROLE,
    REVIEW,
    SRC,
    V21_RUN,
    V22_RUN,
    candidate_tests,
    source_counts,
    verify_fixed_state,
)
from screen_fire_teacher_v13 import run_job


SPECS = (
    ('early8_head_a50', 0.50),
    ('early8_head_a75', 0.75),
)


def layer_index(name):
    parts = name.split('.')
    if len(parts) > 2 and parts[0] == 'model' and parts[1].isdigit():
        return int(parts[1])
    return None


def selected(name):
    layer = layer_index(name)
    is_head = name.startswith('model.23.') and name != 'model.23.dfl.conv.weight'
    is_early_feature = (
        layer is not None
        and PREFIX <= layer < 18
        and name.endswith(('.weight', '.bias'))
    )
    return is_head or is_early_feature


def blend_selected(base, adapted, alpha):
    import torch

    if not 0 < alpha < 1:
        raise ValueError('Interior alpha required')
    if set(base) != set(adapted):
        raise ValueError('Endpoint state keys differ')
    result = {}
    selected_count = 0
    copied_float_count = 0
    integer_count = 0
    for name, left in base.items():
        right = adapted[name]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f'State shape/dtype differs: {name}')
        if left.is_complex() or left.is_quantized:
            raise ValueError(f'Unsupported state tensor: {name}')
        if left.is_floating_point():
            if selected(name):
                value = torch.lerp(
                    left.detach().cpu().float(), right.detach().cpu().float(), alpha
                )
                selected_count += 1
            else:
                value = left.detach().cpu().float().clone()
                copied_float_count += 1
            if not torch.isfinite(value).all():
                raise ValueError(f'Nonfinite blended state: {name}')
            result[name] = value
        else:
            result[name] = left.detach().cpu().clone()
            integer_count += 1
    if not selected_count:
        raise ValueError('Selection contains no tensors')
    return result, {
        'selected_floating_tensors': selected_count,
        'copied_floating_tensors': copied_float_count,
        'integer_buffers': integer_count,
    }


def build_blends(pair, left_path, right_path, left_sha, right_sha, out):
    import torch
    import ultralytics

    if out.exists():
        raise ValueError('Existing blend output')
    if sha256(left_path) != left_sha or sha256(right_path) != right_sha:
        raise ValueError(f'{pair}: endpoint checkpoint changed')
    checkpoints = [
        torch.load(path, map_location='cpu', weights_only=False)
        for path in (left_path, right_path)
    ]
    models = [
        checkpoint.get('ema') if checkpoint.get('ema') is not None else checkpoint['model']
        for checkpoint in checkpoints
    ]
    if architecture(models[0]) != architecture(models[1]):
        raise ValueError(f'{pair}: architecture differs')
    left = models[0].state_dict()
    right = models[1].state_dict()
    out.mkdir(parents=True)
    records = []
    for suffix, alpha in SPECS:
        state, counts = blend_selected(left, right, alpha)
        model = copy.deepcopy(models[0]).cpu().float().eval()
        model.load_state_dict(state, strict=True)
        label = f'{pair}_{suffix}'
        record = {
            'label': label,
            'pair': pair,
            'mask': 'early8_head',
            'alpha': alpha,
            'counts': counts,
            'base_sha256': left_sha,
            'adapted_sha256': right_sha,
            'role': 'exposed development partial interpolation; no training or adoption',
            'torch': torch.__version__,
            'ultralytics': ultralytics.__version__,
        }
        path = out / (suffix + '.pt')
        torch.save(
            {
                'model': model,
                'ema': None,
                'epoch': -1,
                'optimizer': None,
                'train_args': copy.deepcopy(checkpoints[0].get('train_args', {})),
                'blend_metadata': record,
                'version': ultralytics.__version__,
            },
            path,
        )
        records.append(
            {
                **record,
                'weights': str(path.resolve()),
                'expected_sha256': sha256(path),
            }
        )
    return records


def screen(out):
    if out.exists():
        raise ValueError('Existing output; inspect it rather than restarting')
    v21 = read(V21_RUN / 'experiment_meta.json')
    v22 = read(V22_RUN / 'experiment_meta.json')
    prior = read(PRIOR / 'comparison_complete.json')
    if (
        v21['source_sha256'] != v22['source_sha256']
        or v21['dataset']['manifest_sha256'] != DATA_SHA
        or v22['dataset']['manifest_sha256'] != DATA_SHA
        or prior['plan']['data_manifest_sha256'] != DATA_SHA
    ):
        raise ValueError('Source lineage or comparison scope differs')
    source_hashes = {
        'best': (v21['checkpoints']['best.pt'], v22['checkpoints']['best.pt']),
        'last': (v21['checkpoints']['last.pt'], v22['checkpoints']['last.pt']),
    }
    references = {
        name: prior['models'][name]
        for name in ('v21_best', 'v21_last', 'v22_best', 'v22_last')
    }
    records = []
    models = []
    fixed_checks = {}
    for pair, paths in PAIRS.items():
        print('Building early-layer blends for ' + pair, flush=True)
        built = build_blends(
            pair,
            paths[0],
            paths[1],
            source_hashes[pair][0],
            source_hashes[pair][1],
            out / 'weights' / pair,
        )
        fixed_checks[pair] = verify_fixed_state(
            pair, [Path(record['weights']) for record in built]
        )
        records.extend(built)
        models.extend(
            {
                'label': record['label'],
                'weights': record['weights'],
                'expected_sha256': record['expected_sha256'],
            }
            for record in built
        )

    evaluators = [
        'eval_blue_localization.py',
        'eval_fire_fulltest.py',
        'eval_hardcase_frames.py',
        'eval_legacy_source_localization.py',
        'score_legacy_fire_semantics.py',
        'eval_kitchen_source_v20.py',
        'checkpoint_blend.py',
    ]
    plan = {
        'role': 'V23c development-only early-feature plus head interpolation',
        'mask': 'feature layers 10-17 weight/bias plus model.23 except DFL',
        'alpha_definition': '0=v21 endpoint; 1=v22 endpoint',
        'specs': [{'suffix': suffix, 'alpha': alpha} for suffix, alpha in SPECS],
        'parents': {
            pair: {
                'v21': {'path': str(PAIRS[pair][0]), 'sha256': source_hashes[pair][0]},
                'v22': {'path': str(PAIRS[pair][1]), 'sha256': source_hashes[pair][1]},
            }
            for pair in ('best', 'last')
        },
        'fixed_state_checks': fixed_checks,
        'acceptance_rule': prior['plan']['acceptance_rule'],
        'reference_comparison_sha256': sha256(PRIOR / 'comparison_complete.json'),
        'dataset_manifest_sha256': DATA_SHA,
        'script_sha256': sha256(Path(__file__)),
        'evaluators': {name: sha256(SRC / name) for name in evaluators},
        'reserved_sources_predicted': False,
        'independent_test': False,
        'on_board': False,
        'automatic_adoption': False,
    }
    write(out / 'plan.json', plan)
    write(out / 'models.json', models)
    result = {'plan': plan, 'blends': records, 'models': {}}

    for model in models:
        label = model['label']
        weight = Path(model['weights'])
        if sha256(weight) != model['expected_sha256']:
            raise ValueError(f'{label}: blend changed before evaluation')
        record = next(item for item in records if item['label'] == label)
        print('Evaluating ' + label, flush=True)
        args = ['--weights', str(weight)]
        result['models'][label] = {
            'weights_sha256': model['expected_sha256'],
            'alpha': record['alpha'],
            'pair': record['pair'],
            'mask': record['mask'],
            'blue': run_job(
                out, label + '_blue', 'eval_blue_localization.py', args, 'summary.json'
            ),
            'legacy': run_job(
                out, label + '_legacy', 'eval_fire_fulltest.py', args, 'metrics_summary.json'
            ),
            'frames': run_job(
                out,
                label + '_frames',
                'eval_hardcase_frames.py',
                args + ['--data', str(OLD_DATA)],
                'summary.json',
            ),
            'new_training_sources': run_job(
                out,
                label + '_new_sources',
                'eval_kitchen_source_v20.py',
                args + ['--data', str(NEW_DATA)],
                'summary.json',
            ),
        }
        write(out / 'comparison_partial.json', result)

    sources = run_job(
        out,
        'legacy_sources',
        'eval_legacy_source_localization.py',
        ['--data', str(BASE), '--models', str(out / 'models.json')],
        'summary.json',
    )
    semantics = run_job(
        out,
        'visible_flame',
        'score_legacy_fire_semantics.py',
        [
            '--review',
            str(REVIEW),
            '--source-cache',
            str(out / 'legacy_sources'),
            '--models',
            *[model['label'] for model in models],
        ],
        'summary.json',
    )
    acceptance = {}
    for label, row in result['models'].items():
        row['sources'] = sources['models'][label]['by_provenance']
        row['visible_flame'] = semantics['models'][label]
        row['micro'] = micro_score(row['sources'], row['frames']['by_origin'])
        reference_name = REFERENCE_ROLE[row['pair']]
        tests = candidate_tests(row, references[reference_name])
        acceptance[label] = {
            'reference': reference_name,
            'tests': tests,
            'passed': all(tests.values()),
            'automatic_adoption': False,
        }
        if row['new_training_sources']['weights_sha256'] != row['weights_sha256']:
            raise ValueError(f'{label}: new-source evaluation identity differs')
    result['references'] = references
    result['acceptance'] = acceptance
    passing = [label for label, row in acceptance.items() if row['passed']]
    result['passing_candidates'] = passing
    result['decision'] = (
        'No automatic adoption; passing candidates require paired video and visual review.'
        if passing
        else 'No early-feature interpolated checkpoint passes all fixed development gates.'
    )
    for model in models:
        if sha256(Path(model['weights'])) != model['expected_sha256']:
            raise ValueError(f'{model["label"]}: blend changed during evaluation')
    write(out / 'comparison_complete.json', result)
    compact = {}
    for label, row in result['models'].items():
        new = source_counts(row['new_training_sources'])
        indoor = row['sources']['nofire_real_indoor']
        compact[label] = {
            'alpha': row['alpha'],
            'blue': row['blue']['primary_localized_iou50'],
            'tight_f1': row['visible_flame']['supplemental']['0.5']['micro_f1'],
            'indoor_images': indoor['images_with_fp'],
            'indoor_boxes': indoor['fp'],
            'new_source': new,
            'legacy_map50': row['legacy']['map50'],
            'passed': acceptance[label]['passed'],
        }
    print(compact, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    screen(args.out.resolve())
