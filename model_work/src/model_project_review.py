"""Image evidence orders project video review; historical regression gates do not exclude candidates."""
import argparse
import hashlib
import json
from pathlib import Path

UNDER_POT = ('blue_cardoner', 'blue_ka23')


def under_pot_localized(blue):
    rows = {r['slug']: r for r in blue['per_image']}
    if len(rows) != len(blue['per_image']) or not all(n in rows for n in UNDER_POT):
        raise ValueError('Missing or duplicate fixed under-pot diagnostics')
    return sum(bool(rows[n]['primary_localized_iou50']) for n in UNDER_POT)


def project_signals(row, reference):
    """Describe gains and costs, without treating exposed image scores as acceptance."""
    a, b = row['visible_flame'], reference['visible_flame']
    return dict(
        under_pot_blue_gain=under_pot_localized(row['blue']) - under_pot_localized(reference['blue']),
        total_blue_primary_gain=row['blue']['primary_localized_iou50'] - reference['blue']['primary_localized_iou50'],
        visible_flame_f1_iou05_gain=a['supplemental']['0.5']['micro_f1'] - b['supplemental']['0.5']['micro_f1'],
        visible_flame_mean_iou_gain=a['mean_best_prediction_iou'] - b['mean_best_prediction_iou'],
        legacy_map50_change=row['legacy']['map50'] - reference['legacy']['map50'],
        old_development_micro_f1_change=row['micro']['micro_f1'] - reference['micro']['micro_f1'],
        legacy_indoor_fp_change=row['sources']['nofire_real_indoor']['fp'] - reference['sources']['nofire_real_indoor']['fp'])


def video_review_plan(models, reference_name, candidates):
    if len(candidates) != len(set(candidates)) or reference_name in candidates or not candidates:
        raise ValueError('Declare distinct candidates separately from the retained reference')
    reference = models[reference_name]
    signals = {n: project_signals(models[n], reference) for n in candidates}
    # Scheduling only: do not derive an adoption decision or drop a candidate.
    order = sorted(candidates, key=lambda n: (-signals[n]['under_pot_blue_gain'],
                   -signals[n]['visible_flame_f1_iou05_gain'], -signals[n]['visible_flame_mean_iou_gain'], n))
    return dict(role='Project-scoped development video review scheduling; not adoption or independent acceptance',
                policy='docs/MODEL_ADOPTION_CRITERIA_20261002.md', reference=reference_name,
                all_declared_candidates_retained=True, historical_gate_used_to_exclude=False,
                candidates_in_review_order=order, signals=signals,
                required_review=['visible flame and under-pot recall', 'continuous localization and misses',
                                 'off/food/cooking plume within demo scope', 'runtime/deployment cost and stated tradeoffs'],
                automatic_default_replacement=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--comparison', type=Path, required=True)
    p.add_argument('--expected-comparison-sha256', required=True)
    p.add_argument('--semantics', type=Path, required=True)
    p.add_argument('--expected-semantics-sha256', required=True)
    p.add_argument('--reference', default='v13_best')
    p.add_argument('--candidates', nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    if a.out.exists():
        raise SystemExit('Refusing overwrite of project assessment')
    if digest(a.comparison) != a.expected_comparison_sha256 or digest(a.semantics) != a.expected_semantics_sha256:
        raise ValueError('Completed comparison/semantics identity differs')
    comparison = json.loads(a.comparison.read_text(encoding='utf-8'))
    semantics = json.loads(a.semantics.read_text(encoding='utf-8'))
    source_summary = a.comparison.parent / 'legacy_sources/summary.json'
    if digest(source_summary) != semantics['source_summary_sha256'] or semantics['original_metrics_replaced']:
        raise ValueError('Supplementary semantics is not bound to the compared original predictions')
    for name in [a.reference, *a.candidates]:
        if comparison['models'][name]['weights_sha256'] != semantics['models'][name]['weights_sha256']:
            raise ValueError('Compared and supplemented model identities differ')
        comparison['models'][name]['visible_flame'] = semantics['models'][name]
    plan = video_review_plan(comparison['models'], a.reference, a.candidates)
    root = Path(__file__).resolve().parents[2]
    plan.update(source_comparison_sha256=a.expected_comparison_sha256,
                source_semantics_sha256=a.expected_semantics_sha256,
                policy_sha256=digest(root / 'docs/MODEL_ADOPTION_CRITERIA_20261002.md'),
                script_sha256=digest(Path(__file__)),
                models_sha256={n: comparison['models'][n]['weights_sha256'] for n in [a.reference, *a.candidates]})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(plan, indent=2))


if __name__ == '__main__':
    main()
