"""Aggregate the completed cadence package; bind reuse of unchanged v3 regression."""
import argparse
import csv
import json
from pathlib import Path
from data_integrity import sha256


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); root=a.root.resolve()
    if a.out.exists(): raise SystemExit('Refusing overwrite')
    run=root/'model_work/runs/smoke-fire-joint-cadence-v2-20261002'
    paths={'training':run/'experiment_meta.json','cadence':run/'optimizer_cadence_audit.json',
           'freeze':run/'backbone_freeze_audit.json',
           'comparison':root/'model_work/out/joint_cadence_v2_comparison_20261002/comparison_complete.json',
           'baseline_comparison':root/'model_work/out/joint_pilot_comparison_20261002/comparison_complete.json'}
    records={k:json.loads(v.read_text(encoding='utf-8')) for k,v in paths.items()}
    hashes={k:sha256(v) for k,v in paths.items()}
    train,cadence,freeze,comparison,previous=(records[k] for k in paths)
    if train['completed_optimizer_cadence_sha256']!=hashes['cadence'] or train['completed_backbone_audit_sha256']!=hashes['freeze']:
        raise SystemExit('Completed audits changed')
    if train['dataset']!=comparison['pilot_dataset'] or comparison['pilot_dataset']!=previous['pilot_dataset']:
        raise SystemExit('Dataset identities differ')
    for key in ('input_records_sha256','fixed_configuration','scope','script_sha256'):
        if comparison[key]!=previous[key]: raise SystemExit('Evaluation inputs/configuration changed; cannot reuse baseline')
    if sha256(root/'model_work/runs/fire-binary-user-video-v3/weights/best.pt')!=previous['models']['v3']['weights_sha256']:
        raise SystemExit('Baseline weights changed')
    if set(comparison['models'])!={'joint_v2_best','joint_v2_last'}:
        raise SystemExit('Incomplete candidates')
    for name,filename in (('joint_v2_best','best.pt'),('joint_v2_last','last.pt')):
        expected=train[filename+'_sha256']
        if comparison['models'][name]['weights_sha256']!=expected or sha256(run/'weights'/filename)!=expected:
            raise SystemExit('Candidate changed')
    if train['actual_optimizer_steps']!=cadence['actual_optimizer_steps'] or train['actual_batches']!=cadence['batches']:
        raise SystemExit('Update totals disagree')
    if len(cadence['steps'])!=cadence['actual_optimizer_steps'] or not all(x['unchanged'] for x in freeze['checks']):
        raise SystemExit('Audit is incomplete')
    with (run/'results.csv').open(encoding='utf-8',newline='') as f: epochs=list(csv.DictReader(f))
    baseline=previous['models']['v3']; gates={}
    for name,r in comparison['models'].items():
        gate={'legacy_map50_drop_at_most_0_01':baseline['legacy_map']['map50']-r['legacy_map']['map50']<=.01,
              'blue_primary_gain_at_least_one':r['blue_primary_localized_iou50']-baseline['blue_primary_localized_iou50']>=1,
              'no_fire_100_fp_gain_at_most_one':r['legacy_fixed_by_source']['nofire_real_indoor']['fp']-baseline['legacy_fixed_by_source']['nofire_real_indoor']['fp']<=1,
              'any_smoke_tp_on_exposed_three_box_val':r['pilot_val_fixed']['1']['tp']>0}
        gate['all_pass']=all(gate.values()); gates[name]=gate
    result={'role':'fixed small-data schedule package v2; exposed development only',
            'summary_script_sha256':sha256(Path(__file__)),'local_record_sha256':hashes,
            'training':{'config':{k:v for k,v in train['config'].items() if k not in ('data','project')},
                        'environment':train['environment'],'dataset':train['dataset'],'start_sha256':train['start_sha256'],
                        'completed_epochs':len(epochs),'epoch_of_max_csv_map50_95':int(max(epochs,key=lambda x:float(x['metrics/mAP50-95(B)']))['epoch']),
                        'results_csv_sha256':sha256(run/'results.csv'),'backbone_unchanged_checks':len(freeze['checks'])},
            'optimizer_cadence':{'scope':cadence['scope'],'batches':cadence['batches'],
                                  'actual_optimizer_steps':cadence['actual_optimizer_steps'],
                                  'batches_without_observed_step':cadence['batches']-cadence['actual_optimizer_steps'],
                                  'epochs':cadence['epochs'],'first_update_lr':cadence['steps'][0]['lr'],
                                  'last_update_lr':cadence['steps'][-1]['lr']},
            'comparison':comparison,'reused_v3_baseline':baseline,
            'development_screen':gates,'decision':'retain v3; smoke localization signal gained but fire regression and blue screen fail',
            'limitations':['3 smoke truth boxes on visually exposed validation; not general smoke F1.',
                           'Schedule package changes multiple timing parameters; no one-knob causal attribution.',
                           'Old images have no smoke absence truth; new smoke outputs there remain unscored.',
                           'No independent kitchen/event, long-duration alarms or board acceptance.']}
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(a.out),'sha256':sha256(a.out),'decision':result['decision']}))


if __name__=='__main__':main()
