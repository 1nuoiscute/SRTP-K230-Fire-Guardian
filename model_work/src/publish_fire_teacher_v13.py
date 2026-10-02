"""Prepare the selected local model and a public aggregate, with exact export-source equivalence."""
import argparse
import csv
import json
from pathlib import Path
import shutil
from data_integrity import sha256

ROOT = Path(__file__).resolve().parents[2]


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def publish(run, screen, audit, export_static, export_dynamic, out, public):
    if out.exists() or public.exists():
        raise ValueError('Existing selected version or public summary')
    comparison, state, meta = load(screen), load(audit), load(run / 'experiment_meta.json')
    if comparison['selected_candidate'] != 'v13_best' or not comparison['models']['v13_best']['gate_passed']:
        raise ValueError('This publication recipe requires the selected, passing best checkpoint')
    if not state['best_state_equal_epoch3'] or any(not v['fixed_state_equal'] for v in state['models'].values()):
        raise ValueError('Saved-state protection/export-source equivalence missing')
    for script, digest in comparison['plan']['evaluator_sha256'].items():
        if sha256(ROOT / 'model_work/src' / script) != digest:
            raise ValueError('Evaluator changed after comparison')
    base = Path('D:/SRTP_Datasets/kitchen_fire_binary_codex_20260927')
    current_inputs = []
    for row in csv.DictReader((base / 'manifest.csv').open(encoding='utf-8-sig')):
        if row['split'] == 'test':
            image = base / 'images/test' / row['file']
            current_inputs.append(dict(image_sha256=sha256(image), label_sha256=sha256(base / 'labels/test' / (image.stem + '.txt'))))
    if current_inputs != comparison['plan']['legacy_input_identities'] or sha256(base / 'data.yaml') != comparison['plan']['old_data_config_sha256']:
        raise ValueError('Compared test identities changed')
    if sha256(ROOT / 'docs/blue_diagnostic_primary_boxes_20260930.json') != comparison['plan']['blue_annotation_sha256']:
        raise ValueError('Blue diagnostic annotations changed')
    selected = run / 'weights/best.pt'
    for name in ('best.pt', 'epoch3.pt'):
        if sha256(run / 'weights' / name) != state['models'][name]['checkpoint_sha256'] or meta['checkpoints'][name] != state['models'][name]['checkpoint_sha256']:
            raise ValueError('Selected/export checkpoint differs')
    exports = {}
    for policy, folder in (('static', export_static), ('dynamic', export_dynamic)):
        summary = load(folder / 'summary.json')
        if not summary['all_passed'] or summary['images'] != 80 or summary['weights_sha256'] != meta['checkpoints']['epoch3.pt'] or sha256(folder / 'model.onnx') != summary['onnx_sha256']:
            raise ValueError('Export identity/parity check differs')
        exports[policy] = summary
    out.mkdir(parents=True)
    shutil.copyfile(selected, out / 'fire_v13.pt')
    for policy, folder in (('static', export_static), ('dynamic', export_dynamic)):
        shutil.copyfile(folder / 'model.onnx', out / ('fire_v13_' + policy + '.onnx'))
    assets = {p.name: sha256(p) for p in out.iterdir() if p.is_file()}
    if assets['fire_v13.pt'] != meta['checkpoints']['best.pt'] or any(assets['fire_v13_' + p + '.onnx'] != exports[p]['onnx_sha256'] for p in exports):
        raise ValueError('Selected copy changed')
    models = {}
    for label, row in comparison['models'].items():
        models[label] = dict(weights_sha256=row['weights_sha256'], legacy286={k: row['legacy'][k] for k in ('precision', 'recall', 'map50', 'map50_95')},
                             blue_primary_localized_iou50=row['blue']['primary_localized_iou50'],
                             fixed_threshold_sources=comparison['legacy_sources']['models'][label]['by_provenance'],
                             reviewed_development_frames=row['frames']['by_origin'], gate=row['gate'], gate_passed=row['gate_passed'])
    record = dict(role='Selected PC fire development version; exposed comparisons, no independent event or board acceptance',
                  selected='v13_best', version='fire_v13_20261002', rollback='v3',
                  screen_sha256=sha256(screen), state_audit_sha256=sha256(audit), experiment_meta_sha256=sha256(run / 'experiment_meta.json'),
                  publication_script_sha256=sha256(Path(__file__)), gate=comparison['plan']['gate'], models=models,
                  training=meta['config'], dataset=meta['dataset'], saved_state=state, assets=assets,
                  onnx_export_checkpoint='epoch3.pt', selected_checkpoint='best.pt',
                  export_equivalence='Every model state tensor equals between best.pt and epoch3.pt; container hashes differ',
                  onnx={p: {k: s[k] for k in ('weights_sha256', 'onnx_sha256', 'images', 'passed', 'all_passed', 'raw_coordinate_max_abs', 'raw_probability_max_abs', 'config', 'input_shape', 'output_shape')} for p, s in exports.items()},
                  limitations=['Repeated development samples do not establish independent kitchen generalization.',
                               'Blue primary localization improves 0/5 to 1/5; four blue primary flames still missed.',
                               'Legacy mAP50 decreases by about .002; gains concern the reviewed difficult flames.',
                               'Five of ten reviewed teammate flames still missed.',
                               'Visible fire detection does not classify abnormal fire or validate smoke/event alarm behavior.',
                               'No K230, quantization, live camera or event testing during this disconnected-board run.'])
    save(out / 'version_manifest.json', record)
    save(public, record)
    print(json.dumps(dict(selected=record['selected'], assets=assets, public=str(public)), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('run', 'screen', 'audit', 'export-static', 'export-dynamic', 'out', 'public'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    publish(a.run.resolve(), a.screen.resolve(), a.audit.resolve(), a.export_static.resolve(), a.export_dynamic.resolve(), a.out.resolve(), a.public.resolve())
