"""Compare both completed v17 checkpoints with the unchanged selected v16 incumbent."""
import argparse
from pathlib import Path
from data_integrity import sha256
from extend_flame_review_v17 import read, write, verify
from train_flame_extension_v17 import validate_completion, START_SHA
from screen_fire_teacher_v13 import ROOT, SRC, BASE, run_job


def screen(run,data,review,out):
    if out.exists(): raise ValueError('Existing comparison; do not overwrite')
    meta=read(run/'experiment_meta.json')
    validate_completion(meta,read(run/'full_network_update_audit.json'),read(run/'optimizer_cadence_audit.json'))
    if verify(data,meta['dataset']['manifest_sha256'])!=meta['dataset']:
        raise ValueError('Completed dataset changed')
    for name,digest in meta['dependencies'].items():
        if sha256(SRC/name)!=digest: raise ValueError('Training dependency changed')
    if sha256(SRC/'train_flame_extension_v17.py')!=meta['script_sha256']:
        raise ValueError('Training entry changed')
    for name,digest in meta['completed_audits'].items():
        if sha256(run/name)!=digest: raise ValueError('Completed training audit changed')
    if sha256(review/'review_complete.json')!='196927d6b67cfe01c89b94a2aedb3e1f8d876ef3bba17e4079033a773c685682':
        raise ValueError('Supplemental exposed diagnostic changed')
    init=read(run/'actual_initialization_audit.json')
    if not init['exact_v16_best_state_equal'] or init['source_sha256']!=START_SHA:
        raise ValueError('Actual initialization differs')
    models=dict(v16_best=ROOT/'model_work/runs/fire-scratch-semantic-v16-r2-20261003/weights/best.pt',
        v17_best=run/'weights/best.pt',v17_last=run/'weights/last.pt')
    expected=dict(v16_best=START_SHA,v17_best=meta['checkpoints']['best.pt'],v17_last=meta['checkpoints']['last.pt'])
    if any(sha256(p)!=expected[n] for n,p in models.items()):
        raise ValueError('Compared checkpoint changed')
    out.mkdir(parents=True)
    listing=[dict(label=n,weights=str(p),expected_sha256=expected[n]) for n,p in models.items()]
    plan=dict(role='Project-scoped v17 development comparison; no independent kitchen or automatic adoption',
        models=listing,source_experiment_meta_sha256=sha256(run/'experiment_meta.json'),data_manifest_sha256=meta['dataset']['manifest_sha256'],
        supplemental_review_sha256=sha256(review/'review_complete.json'),script_sha256=sha256(Path(__file__)),
        policy_sha256=sha256(ROOT/'docs/MODEL_ADOPTION_CRITERIA_20261002.md'),
        evaluator_sha256={n:sha256(SRC/n) for n in ['eval_blue_localization.py','eval_fire_fulltest.py','eval_hardcase_frames.py','eval_legacy_source_localization.py','score_legacy_fire_semantics.py','model_project_review.py']})
    write(out/'comparison_plan.json',plan); write(out/'models.json',listing)
    result=dict(plan=plan,models={})
    for name,path in models.items():
        args=['--weights',str(path)]
        print('V17 comparing '+name,flush=True)
        result['models'][name]=dict(weights_sha256=expected[name],
            blue=run_job(out,name+'_blue','eval_blue_localization.py',args,'summary.json'),
            legacy=run_job(out,name+'_legacy','eval_fire_fulltest.py',args,'metrics_summary.json'),
            frames=run_job(out,name+'_frames','eval_hardcase_frames.py',args+['--data',str(data)],'summary.json'))
        write(out/'comparison_partial.json',result)
    sources=run_job(out,'legacy_sources','eval_legacy_source_localization.py',['--data',str(BASE),'--models',str(out/'models.json')],'summary.json')
    semantics=run_job(out,'visible_flame','score_legacy_fire_semantics.py',['--review',str(review),'--source-cache',str(out/'legacy_sources'),'--models',*models],'summary.json')
    from compare_fire_training_routes import micro_score
    from model_project_review import video_review_plan
    result['visible_flame']=semantics
    for name,row in result['models'].items():
        row['sources']=sources['models'][name]['by_provenance']; row['visible_flame']=semantics['models'][name]
        row['micro']=micro_score(row['sources'],row['frames']['by_origin'])
    result['project_review']=video_review_plan(result['models'],'v16_best',['v17_best','v17_last'])
    result['automatic_default_replacement']=False
    if any(sha256(p)!=expected[n] for n,p in models.items()): raise ValueError('Checkpoint changed during comparison')
    write(out/'comparison_complete.json',result)
    print('V17 best/last image comparison complete; paired videos/visual review still required',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run','data','review','out']: p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); screen(a.run.resolve(),a.data.resolve(),a.review.resolve(),a.out.resolve())
