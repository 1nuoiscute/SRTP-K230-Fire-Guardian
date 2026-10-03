"""Compare v20 best/last with frozen project controls; reserved sources stay untouched."""
import argparse
from copy import deepcopy
from pathlib import Path
from data_integrity import sha256
from build_kitchen_source_extension_v20 import read,write,verify
from train_kitchen_source_head_v20 import validate_completion,DATA_SHA
from screen_fire_teacher_v13 import ROOT,SRC,BASE,run_job


def screen(run,data,review,previous,out):
    if out.exists():raise ValueError('Existing comparison')
    meta=read(run/'experiment_meta.json')
    validate_completion(meta,read(run/'head_update_audit.json'),read(run/'optimizer_cadence_audit.json'))
    if not meta['serialized_checkpoint_features_equal'] or verify(data,DATA_SHA)!=meta['dataset']:
        raise ValueError('Unverified trained data or saved features')
    if sha256(SRC/'train_kitchen_source_head_v20.py')!=meta['script_sha256']:raise ValueError('Training source changed')
    for name,digest in meta['dependencies'].items():
        if sha256(SRC/name)!=digest or sha256(run/'source_snapshot'/name)!=digest:
            raise ValueError('Training dependency or snapshot changed')
    if sha256(run/'source_snapshot/train_kitchen_source_head_v20.py')!=meta['script_sha256']:
        raise ValueError('Training entry snapshot changed')
    for name,digest in meta['completed_audits'].items():
        if sha256(run/name)!=digest:raise ValueError('Completed training audit changed')
    protocol=read(ROOT/'docs/kitchen_source_v20_comparison_protocol_20261003.json')
    for name,digest in protocol['script_sha256'].items():
        if sha256(SRC/name)!=digest:raise ValueError('Frozen comparison implementation changed')
    if protocol['data_manifest_sha256']!=DATA_SHA or protocol['reserved_predictions_allowed'] or protocol['candidate_files']!=['best.pt','last.pt']:
        raise ValueError('Frozen comparison scope differs')
    if sha256(previous/'comparison_complete.json')!=protocol['reference_image_comparison_sha256'] or sha256(ROOT/'docs/MODEL_ADOPTION_CRITERIA_20261002.md')!=protocol['policy_sha256']:
        raise ValueError('Frozen reference comparison or project policy changed')
    prior=read(previous/'comparison_complete.json')
    preflight=read(ROOT/'docs/kitchen_source_extension_v20_preflight_20261003.json')
    plan_data=read(ROOT/'docs/kitchen_source_extension_v20_plan_20261003.json')
    if sha256(ROOT/'docs/kitchen_source_extension_v20_plan_20261003.json')!=preflight['plan_sha256'] or DATA_SHA!=preflight['data']['manifest_sha256']:
        raise ValueError('Frozen comparison plan changed')
    if prior['plan']['data_manifest_sha256']!='9971c9530a04f1dd962a7c2d9e229d73d5b7e1a47aab5d3d966975d6e78d831a':
        raise ValueError('Wrong reference comparison')
    for name,digest in prior['plan']['evaluator_sha256'].items():
        if sha256(SRC/name)!=digest:raise ValueError('Reference evaluator changed')
    if sha256(review/'review_complete.json')!=prior['plan']['supplemental_review_sha256'] or sha256(ROOT/'docs/MODEL_ADOPTION_CRITERIA_20261002.md')!=prior['plan']['policy_sha256']:
        raise ValueError('Supplemental labels or adoption policy changed')
    # Preserve the 55 old exposed diagnostics separately from the added 14 train sources.
    old_data=Path(plan_data['comparison']['old_diagnostics_data'])
    parent_meta=read(old_data/'build_manifest.json');new_meta=read(data/'build_manifest.json')
    if parent_meta['lineage']!=new_meta['lineage'][:len(parent_meta['lineage'])] or sha256(old_data/'build_manifest.json')!=plan_data['parent_manifest_sha256']:
        raise ValueError('Cached old diagnostic membership changed')
    listing=deepcopy(preflight['fixed_reference_models'])
    for model in listing:
        model['weights']=str((ROOT/model['weights']).resolve())
        cached=prior['models'][model['label']]
        if cached['weights_sha256']!=model['expected_sha256']:raise ValueError('Cached reference weight identity differs')
    for label,filename in [('v20_best','best.pt'),('v20_last','last.pt')]:
        listing.append(dict(label=label,weights=str(run/'weights'/filename),expected_sha256=meta['checkpoints'][filename]))
    if any(sha256(Path(m['weights']))!=m['expected_sha256'] for m in listing):raise ValueError('Checkpoint changed')
    out.mkdir(parents=True);write(out/'models.json',listing)
    plan=dict(role='V20 source-balanced head experiment, old regression and new training feedback kept separate',
        models=listing,data_manifest_sha256=DATA_SHA,source_experiment_meta_sha256=sha256(run/'experiment_meta.json'),
        reference_comparison_sha256=sha256(previous/'comparison_complete.json'),script_sha256=sha256(Path(__file__)),
        evaluator_sha256={name:sha256(SRC/name) for name in ['eval_blue_localization.py','eval_fire_fulltest.py','eval_hardcase_frames.py','eval_legacy_source_localization.py','score_legacy_fire_semantics.py','eval_kitchen_source_v20.py']},
        supplemental_review_sha256=sha256(review/'review_complete.json'),policy_sha256=prior['plan']['policy_sha256'],
        reserved_sources_predicted=False,new_sources_are_training_exposed=True)
    write(out/'comparison_plan.json',plan)
    result=dict(plan=plan,models={m['label']:deepcopy(prior['models'][m['label']]) for m in listing[:3]})
    for label in ['v20_best','v20_last']:
        m=next(m for m in listing if m['label']==label);args=['--weights',m['weights']]
        print('Evaluating '+label,flush=True)
        result['models'][label]=dict(weights_sha256=m['expected_sha256'],
            blue=run_job(out,label+'_blue','eval_blue_localization.py',args,'summary.json'),
            legacy=run_job(out,label+'_legacy','eval_fire_fulltest.py',args,'metrics_summary.json'),
            frames=run_job(out,label+'_frames','eval_hardcase_frames.py',args+['--data',str(old_data)],'summary.json'))
        write(out/'comparison_partial.json',result)
    sources=run_job(out,'legacy_sources','eval_legacy_source_localization.py',['--data',str(BASE),'--models',str(out/'models.json')],'summary.json')
    semantics=run_job(out,'visible_flame','score_legacy_fire_semantics.py',['--review',str(review),'--source-cache',str(out/'legacy_sources'),'--models',*[m['label'] for m in listing]],'summary.json')
    from compare_fire_training_routes import micro_score
    from model_project_review import video_review_plan
    result['visible_flame']=semantics
    for model in listing:
        label=model['label'];row=result['models'][label]
        row['sources']=sources['models'][label]['by_provenance'];row['visible_flame']=semantics['models'][label]
        row['micro']=micro_score(row['sources'],row['frames']['by_origin'])
        row['new_training_sources']=run_job(out,label+'_new_sources','eval_kitchen_source_v20.py',['--weights',model['weights'],'--data',str(data)],'summary.json')
        write(out/'comparison_partial.json',result)
    result['project_review']=video_review_plan(result['models'],'v16_best',['v20_best','v20_last'])
    result['automatic_default_replacement']=False
    if any(sha256(Path(m['weights']))!=m['expected_sha256'] for m in listing):raise ValueError('Weights changed during evaluation')
    write(out/'comparison_complete.json',result)
    print('V20 image comparison complete; paired videos and actual visual review remain mandatory',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['run','data','review','previous','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();screen(a.run.resolve(),a.data.resolve(),a.review.resolve(),a.previous.resolve(),a.out.resolve())
