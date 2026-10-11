"""Publish aggregate-only first joint pilot evidence from completed local records."""
import argparse
import csv
import json
from pathlib import Path
from data_integrity import sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); root = a.root.resolve()
    if a.out.exists():
        raise SystemExit('Refusing overwrite of a published result')
    paths = {
        'review': root/'model_work/out/dfire_pilot_review_20261002/review_complete.json',
        'manifest': Path('D:/SRTP_Datasets/smoke_fire_pilot_v1_20261002/build_manifest.json'),
        'initialization': root/'model_work/out/joint_fire_smoke_start_20261002/initialization.json',
        'training': root/'model_work/runs/smoke-fire-joint-pilot-v1-20261002/experiment_meta.json',
        'backbone_audit': root/'model_work/runs/smoke-fire-joint-pilot-v1-20261002/backbone_freeze_audit.json',
        'comparison': root/'model_work/out/joint_pilot_comparison_20261002/comparison_complete.json',
    }
    records = {k: json.loads(v.read_text(encoding='utf-8')) for k, v in paths.items()}
    identities = {k: sha256(v) for k, v in paths.items()}
    review, manifest, init, train, audit, comparison = (records[k] for k in paths)
    if any(x['source_review_sha256'] != identities['review'] for x in
           (manifest, init, train['dataset'], comparison['pilot_dataset'])):
        raise SystemExit('Review identities disagree')
    if any(x['manifest_sha256'] != identities['manifest'] for x in
           (train['dataset'], comparison['pilot_dataset'])):
        raise SystemExit('Dataset identities disagree')
    if train['completed_backbone_audit_sha256'] != identities['backbone_audit']:
        raise SystemExit('Completed audit changed')
    if not audit['checks'] or not all(x['unchanged'] for x in audit['checks']):
        raise SystemExit('Frozen backbone audit incomplete')
    expected = {'v3': init['source_sha256'], 'joint_start': init['start_sha256'],
                'joint_v1_best': train['best.pt_sha256'], 'joint_v1_last': train['last.pt_sha256']}
    if set(comparison['models']) != set(expected):
        raise SystemExit('Comparison is incomplete')
    local_weights = {
        'v3': root/'model_work/runs/fire-binary-user-video-v3/weights/best.pt',
        'joint_start': root/'model_work/out/joint_fire_smoke_start_20261002/start.pt',
        'joint_v1_best': paths['training'].parent/'weights/best.pt',
        'joint_v1_last': paths['training'].parent/'weights/last.pt',
    }
    for name, identity in expected.items():
        if comparison['models'][name]['weights_sha256'] != identity or sha256(local_weights[name]) != identity:
            raise SystemExit('Compared checkpoint changed')
    csv_path = paths['training'].parent/'results.csv'
    with csv_path.open(encoding='utf-8', newline='') as f:
        epochs = list(csv.DictReader(f))
    if not epochs:
        raise SystemExit('No completed epochs')
    baseline = comparison['models']['v3']
    gates = {}
    for name in ('joint_v1_best', 'joint_v1_last'):
        r = comparison['models'][name]
        gate = {
            'legacy_map50_drop_at_most_0_01': baseline['legacy_map']['map50']-r['legacy_map']['map50'] <= .01,
            'blue_primary_gain_at_least_one': r['blue_primary_localized_iou50']-baseline['blue_primary_localized_iou50'] >= 1,
            'no_fire_100_fp_gain_at_most_one': r['legacy_fixed_by_source']['nofire_real_indoor']['fp']-baseline['legacy_fixed_by_source']['nofire_real_indoor']['fp'] <= 1,
            'any_smoke_tp_on_exposed_three_box_val': r['pilot_val_fixed']['1']['tp'] > 0,
        }
        gate['all_pass'] = all(gate.values()); gates[name] = gate
    result = {
        'role': 'first limited two-class development pilot; no independent kitchen/event acceptance',
        'local_record_sha256': identities,
        'summary_script_sha256': sha256(Path(__file__)),
        'review': {'selected': len(review['images']), 'approved': review['approved_images'],
                   'held': review['held_images'], 'viewed_sheets': len(review['sheets']),
                   'selection_profile': review['selection_profile'], 'exclusions': review['exclusions']},
        'dataset': {'train_images': 27, 'val_images': 9, 'reviewed_development_groups': 33,
                    'train_boxes': {'fire': 15, 'smoke': 16}, 'val_boxes': {'fire': 3, 'smoke': 3},
                    'class_mapping': {'raw_0_smoke': 1, 'raw_1_fire': 0}, 'independent_test_images': 0},
        'initialization': {'transfer': init['transfer'],
                           'raw_square_cpu_comparison': init['raw_square_cpu_comparison'],
                           'saved_reload_wrapper_comparison': init['saved_reload_wrapper_comparison']},
        'training': {'environment': train['environment'],
                     'config': {k:v for k,v in train['config'].items() if k not in ('data','project')},
                     'completed_epochs': len(epochs),
                     'epoch_of_max_csv_map50_95': int(max(epochs, key=lambda x:float(x['metrics/mAP50-95(B)']))['epoch']),
                     'results_csv_sha256': sha256(csv_path), 'max_recorded_lr_pg0': max(float(x['lr/pg0']) for x in epochs),
                     'backbone_unchanged_checks': len(audit['checks']), 'backbone_plan': audit['plan']},
        'comparison': comparison, 'development_screen': gates,
        'decision': 'retain v3; neither trained candidate passes the frozen development screen',
        'limitations': ['All pilot validation is visually exposed development data.',
                        'Visible plume appearance does not establish combustion/steam/event truth.',
                        'Legacy smoke labels are unknown; smoke predictions there are unscored.',
                        'No board access, independent two-class F1, kitchen proportion or long alarm-duration acceptance.'],
    }
    # Derive public counts from the manifest rather than trusting the explanatory constants.
    for split in ('train', 'val'):
        rows = [x for x in manifest['images'] if x['split'] == split]
        result['dataset'][split+'_images'] = len(rows)
        result['dataset'][split+'_boxes'] = {cls:sum(x[cls+'_boxes'] for x in rows) for cls in ('fire','smoke')}
    result['dataset']['reviewed_development_groups'] = len({x['group'] for x in manifest['images']})
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'output': str(a.out), 'sha256':sha256(a.out), 'decision':result['decision']}))


if __name__ == '__main__':
    main()
