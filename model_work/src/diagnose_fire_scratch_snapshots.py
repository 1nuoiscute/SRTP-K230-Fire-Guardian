"""CPU-only fixed periodic diagnostics; never promote or alter an active scratch run."""
import argparse
import csv
import json
import os
from pathlib import Path
from data_integrity import sha256
from screen_fire_teacher_v13 import BASE, ROOT, SRC, run_job, write

PERIODIC = (20,50,90)  # zero-based saved checkpoint names, never final primary candidates


def diagnose(run, epoch, review, out):
    if epoch not in PERIODIC or out.exists():
        raise ValueError('Invalid periodic checkpoint or existing diagnostic')
    preflight_path = run.parent/(run.name+'_preflight.json')
    preflight = json.loads(preflight_path.read_text(encoding='utf-8'))
    if preflight['config']['epochs'] != 100 or preflight['config']['pretrained'] is not False or preflight['config']['resume'] is not False:
        raise ValueError('Fixed random training route changed')
    if sha256(SRC/'train_fire_from_scratch_v14.py') != preflight['script_sha256']:
        raise ValueError('Scratch training source changed')
    if any(sha256(SRC/name)!=digest for name,digest in preflight['dependencies'].items()):
        raise ValueError('Scratch training dependency changed')
    actual_path = run/'actual_initialization_audit.json'
    actual = json.loads(actual_path.read_text(encoding='utf-8'))
    if not actual['exact_random_initial_state_equal'] or actual['supplied_weights'] is not None or actual['actual_initial_state_sha256'] != preflight['initialization']['initial_state_sha256']:
        raise ValueError('Actual random initialization not verified')
    curves = list(csv.DictReader((run/'results.csv').open(encoding='utf-8')))
    curve = next((r for r in curves if int(r['epoch'].strip())==epoch+1),None)
    if curve is None:
        raise ValueError('Periodic snapshot epoch has not completed')
    weight = run/'weights'/f'epoch{epoch}.pt'
    digest = sha256(weight)
    label = f'scratch_v14_epoch{epoch}'
    # These process-local limits apply only to diagnostic subprocesses.
    os.environ['OMP_NUM_THREADS']='2'
    os.environ['MKL_NUM_THREADS']='2'
    os.environ['CUDA_VISIBLE_DEVICES']=''
    out.mkdir(parents=True)
    data = Path(preflight['config']['data']).parent
    plan = dict(role='Periodic convergence diagnostic on development-exposed data; no promotion or training adjustment',
                fixed_periodic_checkpoint_names=[f'epoch{e}.pt' for e in PERIODIC],epoch_zero_based=epoch,
                actual_completed_epochs_for_snapshot=epoch+1,primary_candidates_unchanged=['best.pt','last.pt'],
                checkpoint_sha256=digest,scratch_preflight_sha256=sha256(preflight_path),
                actual_initialization_audit_sha256=sha256(actual_path),dataset_manifest_sha256=preflight['dataset']['manifest_sha256'],
                visible_flame_review_sha256=sha256(review/'review_complete.json'),training_curve_row=curve,
                device='cpu',cpu_threads=2,script_sha256=sha256(Path(__file__)),
                evaluator_sha256={n:sha256(SRC/n) for n in ('eval_blue_localization.py','eval_hardcase_frames.py','eval_legacy_source_localization.py','score_legacy_fire_semantics.py')})
    write(out/'diagnostic_plan.json',plan)
    try:
        args = ['--weights',str(weight),'--device','cpu']
        blue = run_job(out,'blue','eval_blue_localization.py',args,'summary.json')
        frames = run_job(out,'frames','eval_hardcase_frames.py',args+['--data',str(data)],'summary.json')
        models = out/'models.json'
        write(models,[dict(label=label,weights=str(weight),expected_sha256=digest)])
        sources = run_job(out,'legacy_sources','eval_legacy_source_localization.py',
                          ['--data',str(BASE),'--models',str(models),'--device','cpu'],'summary.json')
        semantics = run_job(out,'visible_flame','score_legacy_fire_semantics.py',
                            ['--review',str(review),'--source-cache',str(out/'legacy_sources'),'--models',label],'summary.json')
        if sha256(weight)!=digest:
            raise ValueError('Periodic checkpoint changed during diagnostic')
        result = dict(plan=plan,blue=blue,frames=frames,sources=sources,visible_flame=semantics,
                      primary_selection_performed=False,training_adjusted=False)
        write(out/'diagnostic_complete.json',result)
        print(json.dumps(dict(checkpoint=weight.name,blue_primary=blue['primary_localized_iou50'],
                              frames=frames['by_origin'],sources=sources['models'][label]['by_provenance'],
                              visible_flame=semantics['models'][label]['supplemental']),indent=2),flush=True)
    except Exception as error:
        write(out/'failed.json',dict(error=type(error).__name__,message=str(error),partial_outputs_preserved=True))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--epoch',type=int,choices=PERIODIC,required=True)
    p.add_argument('--review',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a = p.parse_args()
    diagnose(a.run.resolve(),a.epoch,a.review.resolve(),a.out.resolve())
