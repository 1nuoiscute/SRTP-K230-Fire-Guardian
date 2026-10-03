"""Fixed development screen for declared checkpoints; existing metrics are never overwritten."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from data_integrity import sha256

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'model_work/src'
BASE = Path('D:/SRTP_Datasets/kitchen_fire_binary_codex_20260927')
SOURCE_SHA = '48f284f9094fe334e02c66647f95fed8bfb3396b09f4488753508f514ed38980'
SCREEN = ['best.pt', 'last.pt', 'epoch3.pt', 'epoch6.pt', 'epoch9.pt']
GATE = dict(max_legacy_map50_drop=.01, min_blue_gain=1, max_nofire_indoor_fp_gain=1,
            min_ks_flame_tp=56, min_kitchen_stove_tp=18)


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def run_job(out, label, script, args, result):
    target = out / label
    if target.exists():
        raise RuntimeError('Existing output; do not overwrite ' + str(target))
    with (out / (label + '.log')).open('w', encoding='utf-8') as stream:
        job = subprocess.run([sys.executable, '-B', str(SRC / script), '--out', str(target), *args],
                             cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=1800)
    if job.returncode:
        raise RuntimeError('Evaluation failed: ' + label)
    return json.loads((target / result).read_text(encoding='utf-8'))


def evaluate(run, data, out):
    if out.exists():
        raise ValueError('Existing screen; inspect output rather than restarting')
    meta = json.loads((run / 'experiment_meta.json').read_text())
    if meta['checkpoint_screen'] != SCREEN:
        raise ValueError('Declared checkpoint screen differs')
    models = {'v3': ROOT / 'model_work/runs/fire-binary-user-video-v3/weights/best.pt'}
    if sha256(models['v3']) != SOURCE_SHA:
        raise ValueError('Incumbent changed')
    for name in SCREEN:
        weight = run / 'weights' / name
        if sha256(weight) != meta['checkpoints'][name]:
            raise ValueError('Candidate checkpoint changed')
        models['v13_' + weight.stem] = weight
    rows = [r for r in csv.DictReader((BASE / 'manifest.csv').open(encoding='utf-8-sig')) if r['split'] == 'test']
    identities = []
    for row in rows:
        image = BASE / 'images/test' / row['file']
        label = BASE / 'labels/test' / (image.stem + '.txt')
        if sha256(image) != row['sha256']:
            raise ValueError('Test source differs')
        identities.append(dict(image_sha256=sha256(image), label_sha256=sha256(label)))
    out.mkdir(parents=True)
    plan = dict(role='Exposed development checkpoint selection; not an independent acceptance test',
                gate=GATE, selection='Among passing candidates: most blue localization, highest legacy mAP50, then fewest indoor false positives',
                models={k: dict(weights=str(v), sha256=sha256(v)) for k, v in models.items()},
                legacy_input_identities=identities, old_data_config_sha256=sha256(BASE / 'data.yaml'),
                blue_annotation_sha256=sha256(ROOT / 'docs/blue_diagnostic_primary_boxes_20260930.json'),
                development_data_manifest_sha256=sha256(data / 'build_manifest.json'),
                script_sha256=sha256(Path(__file__)), evaluator_sha256={n: sha256(SRC / n) for n in
                    ('eval_blue_localization.py', 'eval_fire_fulltest.py', 'eval_hardcase_frames.py', 'eval_legacy_source_localization.py', 'eval_development_videos.py')})
    write(out / 'screen_plan.json', plan)
    result = dict(plan=plan, models={})
    for label, weight in models.items():
        print('Screening ' + label, flush=True)
        args = ['--weights', str(weight)]
        blue = run_job(out, label + '_blue', 'eval_blue_localization.py', args, 'summary.json')
        legacy = run_job(out, label + '_legacy', 'eval_fire_fulltest.py', args, 'metrics_summary.json')
        frames = run_job(out, label + '_frames', 'eval_hardcase_frames.py', args + ['--data', str(data)], 'summary.json')
        result['models'][label] = dict(weights_sha256=sha256(weight), blue=blue, legacy=legacy, frames=frames)
        write(out / 'screen_partial.json', result)
    listing = out / 'models.json'
    write(listing, [dict(label=k, weights=str(v), expected_sha256=sha256(v)) for k, v in models.items()])
    sources = run_job(out, 'legacy_sources', 'eval_legacy_source_localization.py',
                      ['--data', str(BASE), '--models', str(listing)], 'summary.json')
    result['legacy_sources'] = sources
    baseline = result['models']['v3']
    base_fp = sources['models']['v3']['by_provenance']['nofire_real_indoor']['fp']
    passing = []
    for label, candidate in result['models'].items():
        groups = sources['models'][label]['by_provenance']
        tests = dict(legacy_map50=candidate['legacy']['map50'] >= baseline['legacy']['map50'] - GATE['max_legacy_map50_drop'],
                     blue_gain=candidate['blue']['primary_localized_iou50'] >= baseline['blue']['primary_localized_iou50'] + GATE['min_blue_gain'],
                     indoor_false_positive=groups['nofire_real_indoor']['fp'] <= base_fp + GATE['max_nofire_indoor_fp_gain'],
                     ks_flame_recall=groups['ks_flame']['tp'] >= GATE['min_ks_flame_tp'],
                     kitchen_stove_recall=groups['kitchen_stove_fire']['tp'] >= GATE['min_kitchen_stove_tp'])
        candidate['gate'] = tests
        candidate['gate_passed'] = all(tests.values())
        if label != 'v3' and candidate['gate_passed']:
            passing.append(label)
    passing.sort(key=lambda k: (-result['models'][k]['blue']['primary_localized_iou50'],
                                -result['models'][k]['legacy']['map50'], sources['models'][k]['by_provenance']['nofire_real_indoor']['fp']))
    result['passing_candidates'] = passing
    result['selected_candidate'] = passing[0] if passing else None
    write(out / 'screen_complete.json', result)
    compact = {k: dict(blue=r['blue']['primary_localized_iou50'], legacy_map50=r['legacy']['map50'],
                       sources=sources['models'][k]['by_provenance'], frames=r['frames']['by_origin'], gate=r['gate'])
               for k, r in result['models'].items()}
    print(json.dumps(dict(selected=result['selected_candidate'], models=compact), indent=2), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    evaluate(a.run.resolve(), a.data.resolve(), a.out.resolve())
