"""Compare completed semantic head candidates without replacing the old fire regression targets."""
import argparse
import csv
from pathlib import Path
import torch
from ultralytics import YOLO
from backbone_freeze_audit import is_backbone_key, state_digest
from compare_fire_training_routes import micro_score
from data_integrity import sha256
from semantic_fire_dataset import verify, read
from screen_fire_teacher_v13 import BASE, GATE, ROOT, SRC, run_job, write
from train_visible_flame_pilot_v15 import PRIMARY, START_SHA, completed_comparison
from model_project_review import video_review_plan


def candidate_gate(row, v3, v13):
    a, b = row['visible_flame'], v13['visible_flame']
    gate = dict(legacy_map50=row['legacy']['map50'] >= v3['legacy']['map50'] - GATE['max_legacy_map50_drop'],
                blue_gain=row['blue']['primary_localized_iou50'] >= v3['blue']['primary_localized_iou50'] + GATE['min_blue_gain'],
                indoor_false_positive=row['sources']['nofire_real_indoor']['fp'] <= v3['sources']['nofire_real_indoor']['fp'] + GATE['max_nofire_indoor_fp_gain'],
                ks_flame_recall=row['sources']['ks_flame']['tp'] >= GATE['min_ks_flame_tp'],
                kitchen_stove_recall=row['sources']['kitchen_stove_fire']['tp'] >= GATE['min_kitchen_stove_tp'],
                blue_not_less_than_v13=row['blue']['primary_localized_iou50'] >= v13['blue']['primary_localized_iou50'],
                development_micro_f1_not_less_than_v13=row['micro']['micro_f1'] >= v13['micro']['micro_f1'],
                visible_flame_mean_best_iou_gain=a['mean_best_prediction_iou'] > b['mean_best_prediction_iou'])
    for threshold in ('0.3', '0.5'):
        suffix = threshold.replace('.', '')
        gate['visible_flame_tp_gain_iou' + suffix] = a['supplemental'][threshold]['tp'] > b['supplemental'][threshold]['tp']
        gate['visible_flame_f1_gain_iou' + suffix] = a['supplemental'][threshold]['micro_f1'] > b['supplemental'][threshold]['micro_f1']
    return gate


def verify_completed_training(run, data):
    meta = read(run / 'experiment_meta.json')
    checked = verify(data, meta['dataset']['manifest_sha256'])
    if checked != meta['dataset'] or not meta['candidate_training'] or meta['source_sha256'] != START_SHA:
        raise ValueError('Semantic candidate training identity differs')
    if meta['config']['epochs'] != 12 or meta['checkpoint_screen'] != PRIMARY or meta['config']['save_period'] != -1:
        raise ValueError('Fixed semantic schedule/primary candidates differ')
    if sha256(SRC / 'train_visible_flame_pilot_v15.py') != meta['script_sha256']:
        raise ValueError('Semantic training source changed')
    for name, digest in meta['dependencies'].items():
        if sha256(SRC / name) != digest:
            raise ValueError('Semantic training dependency changed')
    for name, digest in meta['completed_audits'].items():
        if sha256(run / name) != digest:
            raise ValueError('Completed semantic audit changed')
    epochs = read(run / 'teacher_audit.json')
    cadence = read(run / 'optimizer_cadence_audit.json')
    if len(epochs) != 12 or not all(e['teacher_unchanged'] and e['feature_and_head_bn_ema_unchanged'] for e in epochs):
        raise ValueError('Incomplete semantic teacher/fixed-state audit')
    if cadence['actual_optimizer_steps'] != meta['actual_optimizer_steps'] or cadence['batches'] != meta['actual_batches'] or not cadence['actual_optimizer_steps']:
        raise ValueError('Actual learning cadence differs')
    prerequisite = meta['scratch_comparison_prerequisite']
    # Locate the frozen comparison by the operator's declared path below;
    # its identity, complete 100-epoch run and model hashes are checked again.
    if prerequisite is None:
        raise ValueError('No completed scratch comparison was used')
    return meta, checked


def screen(run, data, review, comparison, out):
    if out.exists():
        raise ValueError('Existing semantic comparison; no overwrite')
    meta, checked = verify_completed_training(run, data)
    if completed_comparison(comparison, meta['scratch_comparison_prerequisite']['comparison_sha256']) != meta['scratch_comparison_prerequisite']:
        raise ValueError('Scratch comparison prerequisite changed')
    if sha256(review / 'review_complete.json') != '196927d6b67cfe01c89b94a2aedb3e1f8d876ef3bba17e4079033a773c685682':
        raise ValueError('Declared supplementary flame geometry changed')
    models = dict(v3=ROOT / 'model_work/runs/fire-binary-user-video-v3/weights/best.pt',
                  v13_best=ROOT / 'model_work/out/fire_teacher_v13_selected_20261002/fire_v13.pt',
                  semantic_v15_best=run / 'weights/best.pt', semantic_v15_last=run / 'weights/last.pt')
    expected = dict(v3='48f284f9094fe334e02c66647f95fed8bfb3396b09f4488753508f514ed38980', v13_best=START_SHA,
                    semantic_v15_best=meta['checkpoints']['best.pt'], semantic_v15_last=meta['checkpoints']['last.pt'])
    if set(meta['checkpoints']) != set(PRIMARY) or any(sha256(p) != expected[n] for n, p in models.items()):
        raise ValueError('Checkpoint inventory/identity differs')
    reference = YOLO(str(models['v13_best'])).model.cpu().float().eval()
    source = reference.state_dict()
    if read(run / 'actual_initialization_audit.json')['actual_state_sha256'] != state_digest(source):
        raise ValueError('Actual starting state was not the selected v13')
    blocks = len(reference.model) - 1
    fixed = {k: v for k, v in source.items() if is_backbone_key(k, blocks) or '.dfl.' in k
             or any(k.endswith(s) for s in ('running_mean', 'running_var', 'num_batches_tracked'))}
    saved_audit = {}
    for label in ('semantic_v15_best', 'semantic_v15_last'):
        core = YOLO(str(models[label])).model.cpu().float().eval()
        state = core.state_dict()
        if core.names != {0: 'fire'} or state.keys() != source.keys() or any(not torch.equal(v, state[k]) for k, v in fixed.items()):
            raise ValueError('Saved candidate lost fixed features/BN/DFL')
        changed = [k for k, v in source.items() if not torch.equal(v, state[k])]
        if not changed:
            raise ValueError('Saved candidate did not learn')
        saved_audit[label] = dict(fixed_state_equal=True, changed_head_state_tensors=len(changed))
    identities = []
    for row in csv.DictReader((BASE / 'manifest.csv').open(encoding='utf-8-sig')):
        if row['split'] == 'test':
            image = BASE / 'images/test' / row['file']
            if sha256(image) != row['sha256']:
                raise ValueError('Original regression pixels changed')
            identities.append(dict(image_sha256=sha256(image), label_sha256=sha256(BASE / 'labels/test' / (image.stem + '.txt'))))
    out.mkdir(parents=True)
    plan = dict(role='Semantic fire-head source-quarantine development comparison; approximate exposed geometry, no independent acceptance',
                models={n: dict(weights=str(p), expected_sha256=expected[n]) for n, p in models.items()},
                old_gate=GATE, checkpoint_screen=PRIMARY, original_metrics_replaced=False,
                added_gate='Historical diagnostic: blue/micro-F1 not below v13; visible-flame TP/F1 and mean IoU gain',
                selection='User-authorized project review: retain both best/last for video review; historical gate is diagnostic only',
                project_policy_sha256=sha256(ROOT / 'docs/MODEL_ADOPTION_CRITERIA_20261002.md'),
                project_review_script_sha256=sha256(SRC / 'model_project_review.py'),
                legacy_input_identities=identities, data_manifest_sha256=checked['manifest_sha256'],
                visible_flame_review_sha256=sha256(review / 'review_complete.json'),
                source_experiment_meta_sha256=sha256(run / 'experiment_meta.json'), saved_state_audit=saved_audit,
                script_sha256=sha256(Path(__file__)),
                evaluator_sha256={n: sha256(SRC / n) for n in
                                  ('eval_blue_localization.py', 'eval_fire_fulltest.py', 'eval_hardcase_frames.py',
                                   'eval_legacy_source_localization.py', 'score_legacy_fire_semantics.py')})
    write(out / 'comparison_plan.json', plan)
    result = dict(plan=plan, models={})
    for label, weight in models.items():
        print('Semantic route ' + label, flush=True)
        args = ['--weights', str(weight)]
        blue = run_job(out, label + '_blue', 'eval_blue_localization.py', args, 'summary.json')
        legacy = run_job(out, label + '_legacy', 'eval_fire_fulltest.py', args, 'metrics_summary.json')
        frames = run_job(out, label + '_frames', 'eval_hardcase_frames.py', args + ['--data', str(data)], 'summary.json')
        result['models'][label] = dict(weights_sha256=expected[label], blue=blue, legacy=legacy, frames=frames)
        write(out / 'comparison_partial.json', result)
    listing = out / 'models.json'
    write(listing, [dict(label=n, weights=str(p), expected_sha256=expected[n]) for n, p in models.items()])
    sources = run_job(out, 'legacy_sources', 'eval_legacy_source_localization.py',
                      ['--data', str(BASE), '--models', str(listing)], 'summary.json')
    semantics = run_job(out, 'visible_flame', 'score_legacy_fire_semantics.py',
                        ['--review', str(review), '--source-cache', str(out / 'legacy_sources'), '--models', *models], 'summary.json')
    result['visible_flame'] = semantics
    for label, row in result['models'].items():
        row['sources'] = sources['models'][label]['by_provenance']
        row['visible_flame'] = semantics['models'][label]
        row['micro'] = micro_score(row['sources'], row['frames']['by_origin'])
    passing = []
    for label in ('semantic_v15_best', 'semantic_v15_last'):
        row = result['models'][label]
        row['gate'] = candidate_gate(row, result['models']['v3'], result['models']['v13_best'])
        row['gate_passed'] = all(row['gate'].values())
        if row['gate_passed']:
            passing.append(label)
    passing.sort(key=lambda n: (-result['models'][n]['visible_flame']['supplemental']['0.5']['micro_f1'],
                                -result['models'][n]['visible_flame']['mean_best_prediction_iou'],
                                -result['models'][n]['blue']['primary_localized_iou50'], -result['models'][n]['legacy']['map50']))
    if any(sha256(p) != expected[n] for n, p in models.items()):
        raise ValueError('Compared checkpoint changed during evaluation')
    result['passing_semantic_candidates_under_historical_gate'] = passing
    result['historical_gate_now_diagnostic_only'] = True
    result['project_review'] = video_review_plan(result['models'], 'v13_best',
                                               ['semantic_v15_best', 'semantic_v15_last'])
    result['candidate_for_video_review'] = result['project_review']['candidates_in_review_order'][0]
    result['automatic_default_replacement'] = False
    write(out / 'comparison_complete.json', result)
    print('Finished semantic comparison; project video review retains both best/last: ' +
          str(result['project_review']['candidates_in_review_order']), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('run', 'data', 'review', 'comparison', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    screen(a.run.resolve(), a.data.resolve(), a.review.resolve(), a.comparison.resolve(), a.out.resolve())
