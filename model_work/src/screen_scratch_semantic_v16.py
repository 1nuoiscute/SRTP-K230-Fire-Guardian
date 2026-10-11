"""Evaluate completed v16 best/last against project reference routes."""
import argparse
import csv
from pathlib import Path
from data_integrity import sha256
from semantic_fire_dataset import read, verify
from train_scratch_semantic_v16 import DATA_SHA, PRIMARY, START_SHA


def validate_completion(meta, epochs, cadence):
    if (not meta['candidate_training'] or meta['source_sha256'] != START_SHA
            or meta['config']['epochs'] != 12 or meta['checkpoint_screen'] != PRIMARY
            or set(meta['checkpoints']) != set(PRIMARY) or meta['teacher_loss']):
        raise ValueError('Wrong completed v16 route or candidate inventory')
    if len(epochs) != 12 or [e['epoch'] for e in epochs] != list(range(1, 13)):
        raise ValueError('Incomplete fixed v16 schedule')
    if not all(e['changed_feature_weight_tensors'] > 0 and e['changed_head_weight_tensors'] > 0 for e in epochs):
        raise ValueError('Missing full-network learning evidence')
    if (cadence['actual_optimizer_steps'] != meta['actual_optimizer_steps']
            or cadence['batches'] != meta['actual_batches'] or meta['actual_optimizer_steps'] <= 0):
        raise ValueError('Actual update/batch count differs')


def screen(run, data, review, out):
    import torch
    from ultralytics import YOLO
    from backbone_freeze_audit import is_backbone_key, state_digest
    from compare_fire_training_routes import micro_score
    from model_project_review import video_review_plan
    from screen_fire_teacher_v13 import BASE, ROOT, SRC, run_job, write
    if out.exists():
        raise ValueError('Existing comparison; no overwrite')
    meta = read(run/'experiment_meta.json')
    validate_completion(meta, read(run/'full_network_update_audit.json'), read(run/'optimizer_cadence_audit.json'))
    checked = verify(data, DATA_SHA)
    if checked != meta['dataset'] or sha256(SRC/'train_scratch_semantic_v16.py') != meta['script_sha256']:
        raise ValueError('Training data/source changed')
    for name, expected in meta['dependencies'].items():
        if sha256(SRC/name) != expected:
            raise ValueError('Training dependency changed')
    for name, expected in meta['completed_audits'].items():
        if sha256(run/name) != expected:
            raise ValueError('Completed training audit changed')
    if sha256(review/'review_complete.json') != '196927d6b67cfe01c89b94a2aedb3e1f8d876ef3bba17e4079033a773c685682':
        raise ValueError('Supplemental geometry changed')
    models = dict(v13_best=ROOT/'model_work/out/fire_teacher_v13_selected_20261002/fire_v13.pt',
                  scratch_v14_last=ROOT/'model_work/runs/fire-from-scratch-v14-20261002/weights/last.pt',
                  semantic_v15_last=ROOT/'model_work/runs/fire-visible-flame-v15-20261002/weights/last.pt',
                  scratch_semantic_v16_best=run/'weights/best.pt', scratch_semantic_v16_last=run/'weights/last.pt')
    expected = dict(v13_best='c2b2a8bdc66fe452c80a1786f918f62f8686ad3b0f11061a761969ee02b09966',
                    scratch_v14_last=START_SHA,
                    semantic_v15_last='b323fd44c159ad28519624414a132c9ffa2fc05944d55542be392da67236a6df',
                    scratch_semantic_v16_best=meta['checkpoints']['best.pt'],
                    scratch_semantic_v16_last=meta['checkpoints']['last.pt'])
    if any(sha256(p) != expected[n] for n,p in models.items()):
        raise ValueError('Reference or candidate checkpoint changed')
    reference=YOLO(str(models['scratch_v14_last'])).model.cpu().float().eval()
    source=reference.state_dict()
    if read(run/'actual_initialization_audit.json')['initial_state_sha256'] != state_digest(source):
        raise ValueError('Actual v16 initial state differs from v14 last')
    blocks=len(reference.model)-1
    saved={}
    for label in ('scratch_semantic_v16_best','scratch_semantic_v16_last'):
        core=YOLO(str(models[label])).model.cpu().float().eval()
        state=core.state_dict()
        if core.names!={0:'fire'} or state.keys()!=source.keys() or any(v.is_floating_point() and not torch.isfinite(v).all() for v in state.values()):
            raise ValueError('Saved checkpoint structure or values differ')
        changed=[k for k,v in source.items() if not torch.equal(v,state[k])]
        feature=sum(is_backbone_key(k,blocks) and k.endswith('.weight') for k in changed)
        head=sum(k.startswith(f'model.{blocks}.') and k.endswith('.weight') for k in changed)
        if not feature or not head or not torch.equal(source[f'model.{blocks}.dfl.conv.weight'],state[f'model.{blocks}.dfl.conv.weight']):
            raise ValueError('Saved update scope differs')
        saved[label]=dict(changed_feature_weight_tensors=feature,changed_head_weight_tensors=head)
    inputs=[]
    for row in csv.DictReader((BASE/'manifest.csv').open(encoding='utf-8-sig')):
        if row['split']=='test':
            path=BASE/'images/test'/row['file']
            if sha256(path)!=row['sha256']:
                raise ValueError('Legacy pixels changed')
            inputs.append(dict(image_sha256=sha256(path),label_sha256=sha256(BASE/'labels/test'/(path.stem+'.txt'))))
    out.mkdir(parents=True)
    plan=dict(role='Project-scoped v16 comparison; exposed development data, no automatic adoption',
        models={n:dict(weights=str(p),expected_sha256=expected[n]) for n,p in models.items()},
        source_experiment_meta_sha256=sha256(run/'experiment_meta.json'),data_manifest_sha256=DATA_SHA,
        saved_state_audit=saved,legacy_input_identities=inputs,
        script_sha256=sha256(Path(__file__)),policy_sha256=sha256(ROOT/'docs/MODEL_ADOPTION_CRITERIA_20261002.md'),
        evaluator_sha256={n:sha256(SRC/n) for n in ['eval_blue_localization.py','eval_fire_fulltest.py',
            'eval_hardcase_frames.py','eval_legacy_source_localization.py','score_legacy_fire_semantics.py','model_project_review.py']})
    write(out/'comparison_plan.json',plan)
    result=dict(plan=plan,models={})
    for label,weight in models.items():
        args=['--weights',str(weight)]
        print('V16 comparison: '+label,flush=True)
        result['models'][label]=dict(weights_sha256=expected[label],
            blue=run_job(out,label+'_blue','eval_blue_localization.py',args,'summary.json'),
            legacy=run_job(out,label+'_legacy','eval_fire_fulltest.py',args,'metrics_summary.json'),
            frames=run_job(out,label+'_frames','eval_hardcase_frames.py',args+['--data',str(data)],'summary.json'))
        write(out/'comparison_partial.json',result)
    listing=out/'models.json'
    write(listing,[dict(label=n,weights=str(p),expected_sha256=expected[n]) for n,p in models.items()])
    sources=run_job(out,'legacy_sources','eval_legacy_source_localization.py',['--data',str(BASE),'--models',str(listing)],'summary.json')
    semantics=run_job(out,'visible_flame','score_legacy_fire_semantics.py',
        ['--review',str(review),'--source-cache',str(out/'legacy_sources'),'--models',*models],'summary.json')
    result['visible_flame']=semantics
    for label,row in result['models'].items():
        row['sources']=sources['models'][label]['by_provenance']
        row['visible_flame']=semantics['models'][label]
        row['micro']=micro_score(row['sources'],row['frames']['by_origin'])
    candidates=['scratch_semantic_v16_best','scratch_semantic_v16_last']
    result['project_review_against_v14']=video_review_plan(result['models'],'scratch_v14_last',candidates)
    result['project_review_against_v13']=video_review_plan(result['models'],'v13_best',candidates)
    result['automatic_default_replacement']=False
    if any(sha256(p)!=expected[n] for n,p in models.items()):
        raise ValueError('Compared checkpoint changed')
    write(out/'comparison_complete.json',result)
    print('Finished v16 comparison; both candidates retained for video review',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','data','review','out'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    screen(a.run.resolve(),a.data.resolve(),a.review.resolve(),a.out.resolve())
