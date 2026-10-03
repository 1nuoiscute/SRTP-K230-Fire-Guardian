"""Compare completed random full-network training against frozen v3 and selected v13."""
import argparse
import csv
import json
from pathlib import Path
from data_integrity import sha256
from fire_label_revision_dataset import verify
from screen_fire_teacher_v13 import GATE, ROOT, SRC, run_job, write

BASE = Path('D:/SRTP_Datasets/kitchen_fire_binary_codex_20260927')


def micro_score(sources, frames):
    groups = list(sources.values()) + list(frames.values())
    counts = {k: sum(g[k] for g in groups) for k in ('tp', 'fp', 'fn')}
    denominator = 2 * counts['tp'] + counts['fp'] + counts['fn']
    return dict(**counts, micro_f1=2 * counts['tp'] / denominator if denominator else None,
                scope='286 old images plus 55 fully labelled development-exposed frames; blue primary-only photos excluded')


def candidate_gate(candidate, sources, v3, v13):
    tests = dict(legacy_map50=candidate['legacy']['map50'] >= v3['legacy']['map50'] - GATE['max_legacy_map50_drop'],
                 blue_gain=candidate['blue']['primary_localized_iou50'] >= v3['blue']['primary_localized_iou50'] + GATE['min_blue_gain'],
                 indoor_false_positive=sources['nofire_real_indoor']['fp'] <= v3['sources']['nofire_real_indoor']['fp'] + GATE['max_nofire_indoor_fp_gain'],
                 ks_flame_recall=sources['ks_flame']['tp'] >= GATE['min_ks_flame_tp'],
                 kitchen_stove_recall=sources['kitchen_stove_fire']['tp'] >= GATE['min_kitchen_stove_tp'],
                 development_f1_gain_over_v13=candidate['micro']['micro_f1'] > v13['micro']['micro_f1'],
                 blue_not_less_than_v13=candidate['blue']['primary_localized_iou50'] >= v13['blue']['primary_localized_iou50'])
    return tests


def compare(run, data, out):
    if out.exists():
        raise ValueError('Existing comparison; no overwrite')
    meta = json.loads((run / 'experiment_meta.json').read_text())
    verified = verify(data, meta['dataset']['manifest_sha256'])
    if verified != meta['dataset'] or meta['config']['pretrained'] is not False or meta['config']['resume'] is not False:
        raise ValueError('Scratch dataset/route differs from recorded experiment')
    for name, digest in meta['dependencies'].items():
        if sha256(SRC / name) != digest:
            raise ValueError('Scratch dependency changed')
    if sha256(SRC / 'train_fire_from_scratch_v14.py') != meta['script_sha256']:
        raise ValueError('Scratch training source changed')
    for name, digest in meta['completed_audits'].items():
        if sha256(run / name) != digest:
            raise ValueError('Scratch completed audit changed')
    initial = json.loads((run / 'actual_initialization_audit.json').read_text())
    updates = json.loads((run / 'full_network_update_audit.json').read_text())
    if initial['actual_initial_state_sha256'] != meta['initialization']['initial_state_sha256'] or not initial['exact_random_initial_state_equal'] or initial['supplied_weights'] is not None:
        raise ValueError('Actual random initialization is not verified')
    if len(updates) != 100 or meta['config']['epochs'] != 100 or meta['primary_checkpoint_screen'] != ['best.pt', 'last.pt']:
        raise ValueError('Incomplete fixed scratch schedule')
    if not all(r['changed_feature_weight_tensors'] and r['changed_head_weight_tensors'] for r in updates):
        raise ValueError('Full-network learning audit incomplete')
    models = dict(v3=ROOT / 'model_work/runs/fire-binary-user-video-v3/weights/best.pt',
                  v13_best=ROOT / 'model_work/out/fire_teacher_v13_selected_20261002/fire_v13.pt',
                  scratch_v14_best=run / 'weights/best.pt', scratch_v14_last=run / 'weights/last.pt')
    expected = dict(v3='48f284f9094fe334e02c66647f95fed8bfb3396b09f4488753508f514ed38980',
                    v13_best='c2b2a8bdc66fe452c80a1786f918f62f8686ad3b0f11061a761969ee02b09966',
                    scratch_v14_best=meta['checkpoints']['best.pt'], scratch_v14_last=meta['checkpoints']['last.pt'])
    if any(sha256(v) != expected[k] for k, v in models.items()):
        raise ValueError('Comparison checkpoint changed')
    identities = []
    for r in csv.DictReader((BASE / 'manifest.csv').open(encoding='utf-8-sig')):
        if r['split'] == 'test':
            image = BASE / 'images/test' / r['file']
            if sha256(image) != r['sha256']:
                raise ValueError('Legacy image changed')
            identities.append(dict(image_sha256=sha256(image), label_sha256=sha256(BASE / 'labels/test' / (image.stem + '.txt'))))
    out.mkdir(parents=True)
    plan = dict(role='Random full-network versus retained-feature development routes, not initialization-only causal ablation',
                models={k: dict(weights=str(v), expected_sha256=expected[k]) for k, v in models.items()},
                old_gate=GATE, extra_selection='Greater paired development micro-F1 than v13 and no fewer blue primary localizations',
                legacy_input_identities=identities, data_manifest_sha256=verified['manifest_sha256'],
                blue_annotations_sha256=sha256(ROOT / 'docs/blue_diagnostic_primary_boxes_20260930.json'),
                source_experiment_meta_sha256=sha256(run / 'experiment_meta.json'), script_sha256=sha256(Path(__file__)),
                evaluator_sha256={n: sha256(SRC / n) for n in ('eval_blue_localization.py', 'eval_fire_fulltest.py', 'eval_hardcase_frames.py', 'eval_legacy_source_localization.py')})
    write(out / 'comparison_plan.json', plan)
    results = dict(plan=plan, models={})
    for label, weights in models.items():
        print('Evaluating route ' + label, flush=True)
        args = ['--weights', str(weights)]
        blue = run_job(out, label + '_blue', 'eval_blue_localization.py', args, 'summary.json')
        legacy = run_job(out, label + '_legacy', 'eval_fire_fulltest.py', args, 'metrics_summary.json')
        frames = run_job(out, label + '_frames', 'eval_hardcase_frames.py', args + ['--data', str(data)], 'summary.json')
        results['models'][label] = dict(weights_sha256=expected[label], blue=blue, legacy=legacy, frames=frames)
        write(out / 'comparison_partial.json', results)
    listing = out / 'models.json'
    write(listing, [dict(label=k, weights=str(v), expected_sha256=expected[k]) for k, v in models.items()])
    sources = run_job(out, 'legacy_sources', 'eval_legacy_source_localization.py', ['--data', str(BASE), '--models', str(listing)], 'summary.json')
    for label, row in results['models'].items():
        row['sources'] = sources['models'][label]['by_provenance']
        row['micro'] = micro_score(row['sources'], row['frames']['by_origin'])
    passing = []
    for label in ('scratch_v14_best', 'scratch_v14_last'):
        row = results['models'][label]
        row['gate'] = candidate_gate(row, row['sources'], results['models']['v3'], results['models']['v13_best'])
        row['gate_passed'] = all(row['gate'].values())
        if row['gate_passed']:
            passing.append(label)
    passing.sort(key=lambda k: (-results['models'][k]['micro']['micro_f1'], -results['models'][k]['blue']['primary_localized_iou50'], -results['models'][k]['legacy']['map50']))
    results['passing_scratch_candidates'] = passing
    results['candidate_for_video_review'] = passing[0] if passing else None
    results['incumbent_remains_v13_until_review'] = True
    write(out / 'comparison_complete.json', results)
    print(json.dumps({k: dict(map50=r['legacy']['map50'], blue=r['blue']['primary_localized_iou50'], micro=r['micro'], gate=r.get('gate')) for k, r in results['models'].items()}, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    compare(a.run.resolve(), a.data.resolve(), a.out.resolve())
