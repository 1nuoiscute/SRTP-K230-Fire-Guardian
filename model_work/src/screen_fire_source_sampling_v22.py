"""Compare v22 best/last with the frozen v21 comparison controls."""
import argparse
from copy import deepcopy
from pathlib import Path

from data_integrity import sha256
from build_kitchen_source_extension_v20 import read, write, verify
from train_fire_source_sampling_v22 import DATA_SHA, validate_completion
from screen_fire_teacher_v13 import ROOT, SRC, BASE, run_job


def source_counts(summary):
    return {
        key: sum(row[key] for row in summary['frames'])
        for key in ('tp', 'fp', 'fn')
    }


def acceptance_tests(candidate, reference):
    new_source = source_counts(candidate['new_training_sources'])
    reference_new_source = source_counts(reference['new_training_sources'])
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
        'legacy_map50_not_below_v21': candidate['legacy']['map50'] >= reference['legacy']['map50'],
    }


def screen(run, data, review, previous, out):
    if out.exists():
        raise ValueError('Existing comparison')
    meta = read(run / 'experiment_meta.json')
    validate_completion(
        meta,
        read(run / 'partial_feature_update_audit.json'),
        read(run / 'optimizer_cadence_audit.json'),
        read(run / 'source_sampling_actual.json'),
    )
    if not meta['serialized_frozen_state_equal'] or verify(data, DATA_SHA) != meta['dataset']:
        raise ValueError('Unverified trained data or saved features')

    entry = 'train_fire_source_sampling_v22.py'
    snapshot = run / 'source_snapshot'
    if sha256(snapshot / entry) != meta['script_sha256']:
        raise ValueError('Training entry snapshot changed')
    for name, digest in meta['dependencies'].items():
        if sha256(SRC / name) != digest or sha256(snapshot / name) != digest:
            raise ValueError('Training dependency or snapshot changed')
    for name, digest in meta['completed_audits'].items():
        if sha256(run / name) != digest:
            raise ValueError('Completed training audit changed')
    if meta['actual_batches'] != 3624 or meta['actual_optimizer_steps'] <= 0:
        raise ValueError('Unexpected actual training cadence')

    protocol = read(ROOT / 'docs/kitchen_partial_features_v21_comparison_protocol_20261003.json')
    for name, digest in protocol['script_sha256'].items():
        if sha256(SRC / name) != digest:
            raise ValueError('Frozen comparison implementation changed')
    if (
        protocol['data_manifest_sha256'] != DATA_SHA
        or protocol['reserved_predictions_allowed']
        or protocol['candidate_files'] != ['best.pt', 'last.pt']
    ):
        raise ValueError('Frozen comparison scope differs')
    previous_sha256 = sha256(previous / 'comparison_complete.json')
    prior = read(previous / 'comparison_complete.json')
    if (
        previous_sha256 != protocol['reference_image_comparison_sha256']
        and prior['plan'].get('reference_comparison_sha256')
        != protocol['reference_image_comparison_sha256']
    ) or (
        sha256(ROOT / 'docs/MODEL_ADOPTION_CRITERIA_20261002.md')
        != protocol['policy_sha256']
    ):
        raise ValueError('Frozen reference comparison or project policy changed')

    preflight = read(ROOT / 'docs/kitchen_source_extension_v20_preflight_20261003.json')
    plan_data = read(ROOT / 'docs/kitchen_source_extension_v20_plan_20261003.json')
    if (
        sha256(ROOT / 'docs/kitchen_source_extension_v20_plan_20261003.json')
        != preflight['plan_sha256']
        or DATA_SHA != preflight['data']['manifest_sha256']
        or prior['plan']['data_manifest_sha256'] != DATA_SHA
    ):
        raise ValueError('Frozen comparison plan changed')
    for name, digest in prior['plan']['evaluator_sha256'].items():
        if sha256(SRC / name) != digest:
            raise ValueError('Reference evaluator changed')
    if (
        sha256(review / 'review_complete.json')
        != prior['plan']['supplemental_review_sha256']
        or sha256(ROOT / 'docs/MODEL_ADOPTION_CRITERIA_20261002.md')
        != prior['plan']['policy_sha256']
    ):
        raise ValueError('Supplemental labels or adoption policy changed')

    old_data = Path(plan_data['comparison']['old_diagnostics_data'])
    parent_meta = read(old_data / 'build_manifest.json')
    new_meta = read(data / 'build_manifest.json')
    if (
        parent_meta['lineage'] != new_meta['lineage'][: len(parent_meta['lineage'])]
        or sha256(old_data / 'build_manifest.json') != plan_data['parent_manifest_sha256']
    ):
        raise ValueError('Cached old diagnostic membership changed')

    listing = deepcopy(prior['plan']['models'])
    for model in listing:
        model['weights'] = str((ROOT / model['weights']).resolve())
        if prior['models'][model['label']]['weights_sha256'] != model['expected_sha256']:
            raise ValueError('Cached reference weight identity differs')
    for label, filename in [('v22_best', 'best.pt'), ('v22_last', 'last.pt')]:
        listing.append(
            {
                'label': label,
                'weights': str(run / 'weights' / filename),
                'expected_sha256': meta['checkpoints'][filename],
            }
        )
    if any(sha256(Path(model['weights'])) != model['expected_sha256'] for model in listing):
        raise ValueError('Checkpoint changed')

    out.mkdir(parents=True)
    write(out / 'models.json', listing)
    plan = dict(
        role='V22 controlled source-sampling experiment on the frozen v21 comparison',
        models=listing,
        data_manifest_sha256=DATA_SHA,
        source_experiment_meta_sha256=sha256(run / 'experiment_meta.json'),
        source_sampling_actual_sha256=sha256(run / 'source_sampling_actual.json'),
        reference_comparison_sha256=sha256(previous / 'comparison_complete.json'),
        script_sha256=sha256(Path(__file__)),
        evaluator_sha256={
            name: sha256(SRC / name)
            for name in [
                'eval_blue_localization.py',
                'eval_fire_fulltest.py',
                'eval_hardcase_frames.py',
                'eval_legacy_source_localization.py',
                'score_legacy_fire_semantics.py',
                'eval_kitchen_source_v20.py',
            ]
        },
        supplemental_review_sha256=sha256(review / 'review_complete.json'),
        policy_sha256=prior['plan']['policy_sha256'],
        reserved_sources_predicted=False,
        new_sources_are_training_exposed=True,
        acceptance_rule={
            'comparison': 'v22_best vs v21_best and v22_last vs v21_last',
            'new_source_counts': 'TP >= 3, FN <= 1, FP <= 5',
            'tight_flame': 'microF1 not below the corresponding v21 role',
            'blue': 'at least 4 of 5 primary flames localized',
            'indoor_nofire': 'images and boxes not above the corresponding v21 role',
            'legacy': 'mAP50 not below the corresponding v21 role',
            'audits': 'freeze, feature-BN, sampling, cadence and checkpoint audits pass',
        },
    )
    write(out / 'comparison_plan.json', plan)

    result = {
        'plan': plan,
        'models': {
            model['label']: deepcopy(prior['models'][model['label']])
            for model in listing[:-2]
        },
    }
    for label in ['v22_best', 'v22_last']:
        model = next(model for model in listing if model['label'] == label)
        args = ['--weights', model['weights']]
        print('Evaluating ' + label, flush=True)
        result['models'][label] = dict(
            weights_sha256=model['expected_sha256'],
            blue=run_job(out, label + '_blue', 'eval_blue_localization.py', args, 'summary.json'),
            legacy=run_job(
                out, label + '_legacy', 'eval_fire_fulltest.py', args, 'metrics_summary.json'
            ),
            frames=run_job(
                out,
                label + '_frames',
                'eval_hardcase_frames.py',
                args + ['--data', str(old_data)],
                'summary.json',
            ),
        )
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
            str(review),
            '--source-cache',
            str(out / 'legacy_sources'),
            '--models',
            *[model['label'] for model in listing],
        ],
        'summary.json',
    )
    from compare_fire_training_routes import micro_score
    from model_project_review import video_review_plan

    result['visible_flame'] = semantics
    for model in listing:
        label = model['label']
        row = result['models'][label]
        row['sources'] = sources['models'][label]['by_provenance']
        row['visible_flame'] = semantics['models'][label]
        row['micro'] = micro_score(row['sources'], row['frames']['by_origin'])
        if label.startswith('v22_'):
            row['new_training_sources'] = run_job(
                out,
                label + '_new_sources',
                'eval_kitchen_source_v20.py',
                ['--weights', model['weights'], '--data', str(data)],
                'summary.json',
            )
        elif row['new_training_sources']['weights_sha256'] != model['expected_sha256']:
            raise ValueError('Cached new-source summary identity differs')
        write(out / 'comparison_partial.json', result)

    acceptance = {}
    for candidate, reference in [
        ('v22_best', 'v21_best'),
        ('v22_last', 'v21_last'),
    ]:
        tests = acceptance_tests(result['models'][candidate], result['models'][reference])
        acceptance[candidate] = dict(
            reference=reference,
            tests=tests,
            passed=all(tests.values()),
            training_audits_passed=True,
            independent_test=False,
            on_board=False,
            automatic_adoption=False,
        )
    result['acceptance'] = acceptance
    result['project_review'] = video_review_plan(
        result['models'],
        'v16_best',
        ['v21_best', 'v21_last', 'v22_best', 'v22_last'],
    )
    result['automatic_default_replacement'] = False
    if any(
        sha256(Path(model['weights'])) != model['expected_sha256']
        for model in listing
    ):
        raise ValueError('Weights changed during evaluation')
    write(out / 'comparison_complete.json', result)
    print(
        'V22 image comparison complete; paired videos and actual visual review remain mandatory',
        flush=True,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['run', 'data', 'review', 'previous', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    screen(
        args.run.resolve(),
        args.data.resolve(),
        args.review.resolve(),
        args.previous.resolve(),
        args.out.resolve(),
    )
