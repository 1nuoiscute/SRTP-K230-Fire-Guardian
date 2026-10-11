"""Compare completed v18 best/last with identity-bound v16/v17 reference metrics."""
import argparse
from copy import deepcopy
from pathlib import Path
from data_integrity import sha256
from extend_flame_review_v17 import read,write,verify
from train_frozen_flame_head_v18 import validate_completion,START_SHA,DATA_SHA
from screen_fire_teacher_v13 import ROOT,SRC,BASE,run_job


def screen(run,data,review,previous,out):
    if out.exists():raise ValueError('Existing comparison')
    meta=read(run/'experiment_meta.json')
    validate_completion(meta,read(run/'head_update_audit.json'),read(run/'optimizer_cadence_audit.json'))
    if not meta['serialized_checkpoint_features_equal']:raise ValueError('Saved features not verified')
    if verify(data,DATA_SHA)!=meta['dataset']:raise ValueError('Data differs')
    if sha256(SRC/'train_frozen_flame_head_v18.py')!=meta['script_sha256']:raise ValueError('Training entry changed')
    for n,s in meta['dependencies'].items():
        if sha256(SRC/n)!=s:raise ValueError('Training dependency changed')
    for n,s in meta['completed_audits'].items():
        if sha256(run/n)!=s:raise ValueError('Training audit changed')
    prior=read(previous/'comparison_complete.json')
    if set(prior['models'])!={'v16_best','v17_best','v17_last'} or prior['plan']['data_manifest_sha256']!=DATA_SHA:raise ValueError('Wrong reference comparison')
    for n,s in prior['plan']['evaluator_sha256'].items():
        if sha256(SRC/n)!=s:raise ValueError('Reference evaluator changed')
    if sha256(review/'review_complete.json')!=prior['plan']['supplemental_review_sha256']:raise ValueError('Supplemental labels changed')
    if sha256(ROOT/'docs/MODEL_ADOPTION_CRITERIA_20261002.md')!=prior['plan']['policy_sha256']:raise ValueError('Project policy changed')
    listing=deepcopy(prior['plan']['models'])
    for label,filename in [('v18_best','best.pt'),('v18_last','last.pt')]:listing.append(dict(label=label,weights=str(run/'weights'/filename),expected_sha256=meta['checkpoints'][filename]))
    if any(sha256(Path(m['weights']))!=m['expected_sha256'] for m in listing):raise ValueError('Checkpoint changed')
    out.mkdir(parents=True);write(out/'models.json',listing)
    plan=dict(role='Fixed-feature versus full-network project development comparison; references cached, new candidates evaluated',
        models=listing,data_manifest_sha256=DATA_SHA,source_experiment_meta_sha256=sha256(run/'experiment_meta.json'),
        reference_comparison_sha256=sha256(previous/'comparison_complete.json'),script_sha256=sha256(Path(__file__)),
        supplemental_review_sha256=sha256(review/'review_complete.json'),policy_sha256=prior['plan']['policy_sha256'],evaluator_sha256=prior['plan']['evaluator_sha256'])
    write(out/'comparison_plan.json',plan);result=dict(plan=plan,models=deepcopy(prior['models']))
    for label in ['v18_best','v18_last']:
        m=next(m for m in listing if m['label']==label);args=['--weights',m['weights']]
        print('Evaluating '+label,flush=True)
        result['models'][label]=dict(weights_sha256=m['expected_sha256'],
            blue=run_job(out,label+'_blue','eval_blue_localization.py',args,'summary.json'),
            legacy=run_job(out,label+'_legacy','eval_fire_fulltest.py',args,'metrics_summary.json'),
            frames=run_job(out,label+'_frames','eval_hardcase_frames.py',args+['--data',str(data)],'summary.json'))
        write(out/'comparison_partial.json',result)
    sources=run_job(out,'legacy_sources','eval_legacy_source_localization.py',['--data',str(BASE),'--models',str(out/'models.json')],'summary.json')
    semantics=run_job(out,'visible_flame','score_legacy_fire_semantics.py',['--review',str(review),'--source-cache',str(out/'legacy_sources'),'--models',*[m['label'] for m in listing]],'summary.json')
    from compare_fire_training_routes import micro_score
    from model_project_review import video_review_plan
    result['visible_flame']=semantics
    for label,row in result['models'].items():
        row['sources']=sources['models'][label]['by_provenance'];row['visible_flame']=semantics['models'][label]
        row['micro']=micro_score(row['sources'],row['frames']['by_origin'])
    result['project_review']=video_review_plan(result['models'],'v16_best',['v18_best','v18_last'])
    result['automatic_default_replacement']=False
    if any(sha256(Path(m['weights']))!=m['expected_sha256'] for m in listing):raise ValueError('Weights changed during evaluation')
    write(out/'comparison_complete.json',result)
    print('V18 image comparison complete; paired videos and actual visual review required',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['run','data','review','previous','out']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();screen(a.run.resolve(),a.data.resolve(),a.review.resolve(),a.previous.resolve(),a.out.resolve())
