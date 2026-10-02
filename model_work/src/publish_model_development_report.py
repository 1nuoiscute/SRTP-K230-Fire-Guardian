"""Bind completed training and unchanged evaluation scope, publish aggregate facts."""
import argparse
import csv
import json
from pathlib import Path
from data_integrity import sha256


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--comparison',type=Path,required=True)
    p.add_argument('--baseline-comparison',type=Path,required=True)
    p.add_argument('--raw-check',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists(): raise SystemExit('Refusing overwrite')
    paths={'training':a.run/'experiment_meta.json','comparison':a.comparison,
           'baseline_comparison':a.baseline_comparison,'raw_check':a.raw_check,
           'cadence':a.run/'optimizer_cadence_audit.json','fire_audit':a.run/'fire_path_audit.json'}
    records={k:json.loads(v.read_text(encoding='utf-8')) for k,v in paths.items()}
    hashes={k:sha256(v) for k,v in paths.items()}
    train,comparison,previous,raw,cadence,audit=(records[k] for k in paths)
    for name,field in (('cadence','completed_optimizer_cadence_sha256'),('fire_audit','completed_fire_path_audit_sha256')):
        if train[field]!=hashes[name]: raise SystemExit('Completed audit changed')
    for key in ('input_records_sha256','fixed_configuration','scope','script_sha256','pilot_dataset'):
        if comparison[key]!=previous[key]: raise SystemExit('Cannot reuse changed baseline scope')
    if train['dataset']!=comparison['pilot_dataset']: raise SystemExit('Training/evaluation dataset differs')
    if comparison['models'].keys()!=raw['models'].keys(): raise SystemExit('Raw/point comparison candidates differ')
    identities={train['best.pt_sha256'],train['last.pt_sha256']}
    if {r['weights_sha256'] for r in comparison['models'].values()}!=identities:
        raise SystemExit('Training/evaluation checkpoints differ')
    for filename in ('best.pt','last.pt'):
        if sha256(a.run/'weights'/filename)!=train[filename+'_sha256']: raise SystemExit('Checkpoint changed')
    for label,row in comparison['models'].items():
        if raw['models'][label]['weights_sha256']!=row['weights_sha256']:
            raise SystemExit('Raw checkpoint differs')
    if not all(c['unchanged'] for c in audit['checks']): raise SystemExit('Preservation audit failed')
    with (a.run/'results.csv').open(encoding='utf-8',newline='') as f: epochs=list(csv.DictReader(f))
    baseline=previous['models']['v3']; gates={}
    for label,row in comparison['models'].items():
        r=raw['models'][label]
        gate={'saved_protected_state_equal':r['protected_saved_state_equal'],
              'raw_fire_geometry_max_abs_at_most_1e_5':max(x['fire_geometry_max_abs_vs_v3'] for x in r['raw_comparison'])<=1e-5,
              'legacy_map50_drop_at_most_0_01':baseline['legacy_map']['map50']-row['legacy_map']['map50']<=.01,
              'original_kitchen_fire_tp_not_reduced':row['legacy_fixed_by_source']['ks_flame']['tp']>=baseline['legacy_fixed_by_source']['ks_flame']['tp'],
              'no_fire_100_fire_fp_not_increased':row['legacy_fixed_by_source']['nofire_real_indoor']['fp']<=baseline['legacy_fixed_by_source']['nofire_real_indoor']['fp'],
              'any_smoke_tp_on_exposed_three_box_val':row['pilot_val_fixed']['1']['tp']>0}
        gate['all_stage_pass']=all(gate.values()); gates[label]=gate
    result={'role':'row-only functional preservation development probe; no global model promotion',
            'summary_script_sha256':sha256(Path(__file__)),'local_record_sha256':hashes,
            'training':{'config':{k:v for k,v in train['config'].items() if k not in ('data','project')},
                        'environment':train['environment'],'dataset':train['dataset'],'start_sha256':train['start_sha256'],
                        'completed_epochs':len(epochs),'epoch_of_max_csv_map50_95':int(max(epochs,key=lambda x:float(x['metrics/mAP50-95(B)']))['epoch']),
                        'fire_path_plan':train['fire_path_plan'],'preservation_checks':len(audit['checks']),
                        'results_csv_sha256':sha256(a.run/'results.csv')},
            'cadence':{'batches':cadence['batches'],'actual_optimizer_steps':cadence['actual_optimizer_steps']},
            'raw_check':raw,'comparison':comparison,'reused_v3_baseline':baseline,'stage_screen':gates,
            'decision':'retain v3; fire path preserved but row-only smoke output has no fixed-threshold localization',
            'limitations':['Visually exposed development validation, three smoke truth boxes.',
                           'Blue flame and independent proposal requirements remain unmet.',
                           'Live/raw state checks do not alone establish final prediction quality.']}
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(a.out),'sha256':sha256(a.out)}))


if __name__=='__main__':main()
