"""Development-only v21/v22 checkpoint interpolation screen.

This runner never trains, adopts a model, exports ONNX, or touches reserved
sources. It creates three fixed interpolation coefficients per checkpoint role
and evaluates every result on the same exposed development suites.
"""
import argparse
from pathlib import Path

from build_kitchen_source_extension_v20 import read, verify, write
from checkpoint_blend import build_blends
from compare_fire_training_routes import micro_score
from data_integrity import sha256
from screen_fire_teacher_v13 import BASE, ROOT, SRC, run_job
from train_fire_source_sampling_v22 import DATA_SHA, FEATURES, PREFIX


V21_RUN = ROOT / 'model_work/runs/fire-kitchen-partial-features-v21-20261028'
V22_RUN = ROOT / 'model_work/runs/fire-source-sampling-v22-20261028'
PRIOR = ROOT / 'model_work/out/kitchen_source_sampling_v22_comparison_20261028'
REVIEW = ROOT / 'model_work/out/legacy_visible_flame_audit_20261002'
OLD_DATA = Path('D:/SRTP_Datasets/fire_train_context_v19_20261003')
NEW_DATA = Path('D:/SRTP_Datasets/fire_kitchen_source_v20_20261003')
ALPHAS = (0.25, 0.50, 0.75)
PAIRS = {
    'best': (V21_RUN / 'weights/best.pt', V22_RUN / 'weights/best.pt'),
    'last': (V21_RUN / 'weights/last.pt', V22_RUN / 'weights/last.pt'),
}
REFERENCE_ROLE = {'best': 'v21_best', 'last': 'v21_last'}


def source_counts(summary):
    return {
        key: sum(row[key] for row in summary['frames'])
        for key in ('tp', 'fp', 'fn')
    }


def fixed_key(name, torch):
    from backbone_freeze_audit import is_backbone_key

    if is_backbone_key(name, PREFIX):
        return True
    if (
        is_backbone_key(name, FEATURES)
        and name.endswith(('running_mean', 'running_var', 'num_batches_tracked'))
    ):
        return True
    return name == 'model.23.dfl.conv.weight'


def verify_fixed_state(pair, blend_paths):
    import torch
    from ultralytics import YOLO

    left_path, right_path = PAIRS[pair]
    left = YOLO(str(left_path)).model.state_dict()
    right = YOLO(str(right_path)).model.state_dict()
    if set(left) != set(right):
        raise ValueError(f'{pair}: endpoint state keys differ')
    fixed = [name for name in left if fixed_key(name, torch)]
    if not fixed:
        raise ValueError(f'{pair}: no fixed tensors identified')
    changed = [name for name in fixed if not torch.equal(left[name], right[name])]
    if changed:
        raise ValueError(f'{pair}: fixed endpoint state differs: {changed[:5]}')
    for path in blend_paths:
        state = YOLO(str(path)).model.state_dict()
        changed = [name for name in fixed if not torch.equal(left[name], state[name])]
        if changed:
            raise ValueError(f'{pair}: blended fixed state differs: {changed[:5]}')
    return {
        'fixed_tensors': len(fixed),
        'endpoint_fixed_state_equal': True,
        'blended_fixed_state_equal': True,
    }


def candidate_tests(candidate, reference):
    new_source = source_counts(candidate['new_training_sources'])
    indoor = candidate['sources']['nofire_real_indoor']
    reference_indoor = reference['sources']['nofire_real_indoor']
    return {
        'new_source_tp_at_least_3': new_source['tp'] >= 3,
        'new_source_fn_at_most_1': new_source['fn'] <= 1,
        'new_source_fp_at_most_5': new_source['fp'] <= 5,
        'tight_f1_not_below_v21': (
            candidate['visible_flame']['supplemental']['0.5']['micro_f1']
            >= reference['visible_flame']['supplemental']['0.5']['micro_f1']
        ),
        'blue_localized_at_least_4': candidate['blue']['primary_localized_iou50'] >= 4,
        'indoor_images_not_above_v21': (
            indoor['images_with_fp'] <= reference_indoor['images_with_fp']
        ),
        'indoor_boxes_not_above_v21': indoor['fp'] <= reference_indoor['fp'],
        'legacy_map50_not_below_v21': (
            candidate['legacy']['map50'] >= reference['legacy']['map50']
        ),
    }


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
    ):
        raise ValueError('v21/v22 lineage or dataset identity differs')
    if verify(NEW_DATA, DATA_SHA) != v22['dataset']:
        raise ValueError('Current v20 dataset differs from training metadata')
    if (
        prior['plan']['data_manifest_sha256'] != DATA_SHA
        or prior['plan']['reserved_sources_predicted']
        or not prior['plan']['new_sources_are_training_exposed']
    ):
        raise ValueError('Prior comparison scope differs')

    endpoint_meta = {'best': v21['checkpoints']['best.pt'], 'last': v21['checkpoints']['last.pt']}
    adapted_meta = {'best': v22['checkpoints']['best.pt'], 'last': v22['checkpoints']['last.pt']}
    for pair, (left, right) in PAIRS.items():
        if sha256(left) != endpoint_meta[pair] or sha256(right) != adapted_meta[pair]:
            raise ValueError(f'{pair}: endpoint checkpoint changed')

    references = {
        name: prior['models'][name]
        for name in ('v21_best', 'v21_last', 'v22_best', 'v22_last')
    }
    if (
        references['v21_best']['weights_sha256'] != endpoint_meta['best']
        or references['v21_last']['weights_sha256'] != endpoint_meta['last']
        or references['v22_best']['weights_sha256'] != adapted_meta['best']
        or references['v22_last']['weights_sha256'] != adapted_meta['last']
    ):
        raise ValueError('Prior comparison uses different endpoints')

    blend_records = {}
    fixed_checks = {}
    models = []
    for pair in ('best', 'last'):
        left, right = PAIRS[pair]
        weights_out = out / 'weights' / pair
        records = build_blends(
            left,
            right,
            endpoint_meta[pair],
            adapted_meta[pair],
            list(ALPHAS),
            weights_out,
        )
        fixed_checks[pair] = verify_fixed_state(
            pair, [Path(record['weights']) for record in records]
        )
        for record in records:
            label = f'{pair}_a{int(round(record["alpha"] * 100)):02d}'
            entry = dict(record, label=label, pair=pair)
            blend_records[label] = entry
            models.append(
                {
                    'label': label,
                    'weights': record['weights'],
                    'expected_sha256': record['expected_sha256'],
                }
            )

    evaluator_names = [
        'eval_blue_localization.py',
        'eval_fire_fulltest.py',
        'eval_hardcase_frames.py',
        'eval_legacy_source_localization.py',
        'score_legacy_fire_semantics.py',
        'eval_kitchen_source_v20.py',
    ]
    plan = {
        'role': 'V23 development-only interpolation screen of the v21/v22 branch',
        'parents': {
            'best': {
                'v21': {'path': str(PAIRS['best'][0]), 'sha256': endpoint_meta['best']},
                'v22': {'path': str(PAIRS['best'][1]), 'sha256': adapted_meta['best']},
            },
            'last': {
                'v21': {'path': str(PAIRS['last'][0]), 'sha256': endpoint_meta['last']},
                'v22': {'path': str(PAIRS['last'][1]), 'sha256': adapted_meta['last']},
            },
        },
        'alphas': list(ALPHAS),
        'alpha_definition': '0=v21 endpoint; 1=v22 endpoint',
        'blend_policy': (
            'FP32 linear interpolation of floating parameters and buffers; '
            'integer buffers from v21; no BN recalibration'
        ),
        'fixed_state_checks': fixed_checks,
        'comparison': {
            'conf': 0.25,
            'nms_iou': 0.6,
            'match_iou': 0.5,
            'new_development_images': 14,
            'old_diagnostics_data': str(OLD_DATA),
            'new_data': str(NEW_DATA),
        },
        'acceptance_rule': {
            'new_source_counts': 'TP >= 3, FN <= 1, FP <= 5',
            'tight_flame': 'microF1 not below the corresponding v21 role',
            'blue': 'at least 4 of 5 primary flames localized',
            'indoor_nofire': 'images and boxes not above the corresponding v21 role',
            'legacy': 'mAP50 not below the corresponding v21 role',
        },
        'reference_comparison_sha256': sha256(PRIOR / 'comparison_complete.json'),
        'dataset_manifest_sha256': DATA_SHA,
        'script_sha256': sha256(Path(__file__)),
        'evaluators': {name: sha256(SRC / name) for name in evaluator_names},
        'reserved_sources_predicted': False,
        'independent_test': False,
        'on_board': False,
        'automatic_adoption': False,
    }
    write(out / 'plan.json', plan)
    write(out / 'models.json', models)

    result = {'plan': plan, 'blends': blend_records, 'models': {}}
    for model in models:
        label = model['label']
        weight = Path(model['weights'])
        if sha256(weight) != model['expected_sha256']:
            raise ValueError(f'{label}: blend changed before evaluation')
        print('Evaluating ' + label, flush=True)
        args = ['--weights', str(weight)]
        result['models'][label] = {
            'weights_sha256': model['expected_sha256'],
            'alpha': blend_records[label]['alpha'],
            'pair': blend_records[label]['pair'],
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

    listing = out / 'candidate_models.json'
    write(listing, models)
    sources = run_job(
        out,
        'legacy_sources',
        'eval_legacy_source_localization.py',
        ['--data', str(BASE), '--models', str(listing)],
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
        else 'No interpolated checkpoint passes all fixed development gates.'
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
