"""Publish bound aggregate evidence for independent smoke-head development."""
import argparse
import csv
import json
from pathlib import Path
from data_integrity import sha256


def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True)
    p.add_argument('--packed',type=Path,required=True); p.add_argument('--comparison',type=Path,required=True)
    p.add_argument('--baseline-comparison',type=Path,required=True); p.add_argument('--onnx-report',type=Path,required=True)
    p.add_argument('--initialization',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists(): raise SystemExit('Refusing overwrite')
    paths={'training':a.run/'experiment_meta.json','shared_audit':a.run/'shared_feature_audit.json',
           'ema_audit':a.run/'shared_ema_audit.json','cadence':a.run/'optimizer_cadence_audit.json',
           'packed_raw_check':a.packed/'raw_saved_check.json','comparison':a.comparison,
           'baseline_comparison':a.baseline_comparison,'onnx_report':a.onnx_report,'initialization':a.initialization}
    r={k:json.loads(v.read_text(encoding='utf-8')) for k,v in paths.items()}; identities={k:sha256(v) for k,v in paths.items()}
    train,shared,ema,cadence,raw,comparison,previous,onnx,init=(r[k] for k in paths)
    for name,field in (('shared_audit','completed_shared_audit_sha256'),('ema_audit','completed_shared_ema_sha256'),('cadence','completed_optimizer_cadence_sha256')):
        if train[field]!=identities[name]: raise SystemExit('Completed audit changed')
    if not all(x['unchanged'] for x in shared['checks']) or not all(x['shared_features_equal'] for x in ema):
        raise SystemExit('Shared-state audit failed')
    for key in ('input_records_sha256','fixed_configuration','scope','script_sha256','pilot_dataset'):
        if comparison[key]!=previous[key]: raise SystemExit('Changed evaluation scope; cannot reuse baseline')
    if train['dataset']['parent_manifest_sha256']!=comparison['pilot_dataset']['manifest_sha256'] or train['start_sha256']!=init['start_sha256']:
        raise SystemExit('Training lineage differs')
    if set(raw['models'])!=set(comparison['models']) or set(raw['models'])!={'dual_v4_start','dual_v4_best','dual_v4_last'}:
        raise SystemExit('Incomplete candidate comparisons')
    source_ids={'dual_v4_start':init['start_sha256'],'dual_v4_best':train['best.pt_sha256'],'dual_v4_last':train['last.pt_sha256']}
    for label,source in source_ids.items():
        row=raw['models'][label]
        if row['source_smoke_sha256']!=source or row['weights_sha256']!=comparison['models'][label]['weights_sha256'] or sha256(a.packed/(label+'.pt'))!=row['weights_sha256']:
            raise SystemExit('Packed candidate identity differs')
    for filename in ('best.pt','last.pt'):
        if sha256(a.run/'weights'/filename)!=train[filename+'_sha256']: raise SystemExit('Student checkpoint changed')
    if onnx['weights_sha256']!=raw['models']['dual_v4_best']['weights_sha256'] or onnx['nms_matched_images']!=onnx['verified_images']:
        raise SystemExit('ONNX candidate/parity differs')
    baseline=previous['models']['v3']; gates={}; numerical={}
    for label,row in comparison['models'].items():
        check=raw['models'][label]; maximum=max(x['fire_geometry_max_abs_vs_v3'] for x in check['raw_comparison'])
        gate={'saved_fire_state_equal':check['protected_saved_state_equal'],'raw_fire_geometry_difference_at_most_1e_5':maximum<=1e-5,
              'legacy_fire_map50_drop_at_most_0_01':baseline['legacy_map']['map50']-row['legacy_map']['map50']<=.01,
              'old_kitchen_fire_tp_not_reduced':row['legacy_fixed_by_source']['ks_flame']['tp']>=baseline['legacy_fixed_by_source']['ks_flame']['tp'],
              'no_fire_100_fire_fp_not_increased':row['legacy_fixed_by_source']['nofire_real_indoor']['fp']<=baseline['legacy_fixed_by_source']['nofire_real_indoor']['fp'],
              'any_smoke_tp_on_exposed_three_box_val':row['pilot_val_fixed']['1']['tp']>0}
        gate['stage_pass']=all(gate.values()); gates[label]=gate
        numerical[label]={k:check[k] for k in ('weights_sha256','source_smoke_sha256','protected_saved_state_equal','parameters','baseline_parameters','added_parameters')}
        numerical[label].update(raw_checked_images=len(check['raw_comparison']),raw_fire_geometry_max_abs=maximum)
    with (a.run/'results.csv').open(encoding='utf-8',newline='') as f: epochs=list(csv.DictReader(f))
    result={'role':'Independent smoke-head developer candidate; default fire v3 retained, no proposal acceptance',
            'summary_script_sha256':sha256(Path(__file__)),'local_record_sha256':identities,'initialization':init,
            'training':{'config':{k:v for k,v in train['config'].items() if k not in ('data','project')},'environment':train['environment'],
                        'dataset':train['dataset'],'completed_epochs':len(epochs),'epoch_of_max_smoke_csv_map50_95':int(max(epochs,key=lambda x:float(x['metrics/mAP50-95(B)']))['epoch']),
                        'shared_live_checks':len(shared['checks']),'shared_ema_checks':len(ema),
                        'results_csv_sha256':sha256(a.run/'results.csv'),'actual_batches':cadence['batches'],'actual_optimizer_steps':cadence['actual_optimizer_steps']},
            'packed_numeric_check':numerical,'comparison':comparison,'reused_v3_baseline':baseline,'stage_screen':gates,'onnx_cpu_consistency':onnx,
            'decision':'retain v3 default; dual_v4_best is a functional smoke development candidate only' if gates['dual_v4_best']['stage_pass'] else 'retain v3; stage screen failed',
            'limitations':['Validation is fully exposed development data with only three smoke truth boxes.',
                           'Visible smoke-like plumes do not identify combustion/steam/events.',
                           'Blue flame remains unresolved; original full-model promotion gate not passed.',
                           'Static CPU ONNX consistency and session-only timings are not KModel, hardware or end-to-end latency evidence.',
                           'Independent kitchen share, two-class F1, long alarm duration and board acceptance remain unmet.']}
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(a.out),'sha256':sha256(a.out),'decision':result['decision']}))


if __name__=='__main__':main()
