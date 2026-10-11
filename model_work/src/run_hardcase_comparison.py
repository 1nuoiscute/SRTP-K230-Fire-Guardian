"""Wait for a specifically verified live training process, then compare checkpoints.
Never restarts training. All subprocesses are sequential; logs persist on failure.
No remote publishing or board operation is performed by this runner.
"""
import argparse,json,subprocess,sys,time
from pathlib import Path
import psutil
from data_integrity import sha256
ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'model_work/src'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--training-pid',type=int,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--candidate-label',default='v5')
    p.add_argument('--frame-reference-run',type=Path)
    p.add_argument('--frame-reference-label',default='reference')
    p.add_argument('--legacy-source-counts',action='store_true',help='Also compare fixed-threshold original test sources')
    p.add_argument('--job-timeout-seconds',type=int,default=1800)
    args=p.parse_args();run=args.run.resolve();out=args.out.resolve();data=args.data.resolve()
    if args.job_timeout_seconds <= 0: p.error('Job timeout must be positive')
    marker=out/'comparison_launch.json'
    if marker.exists(): raise SystemExit('Existing comparison launch; inspect handle/logs before restarting')
    process=psutil.Process(args.training_pid)
    command=process.cmdline()
    if not any('train_hardcase_v5.py' in value for value in command) or not any(run.name in value for value in command):
        raise SystemExit('PID is not the named training run')
    created=process.create_time();out.mkdir(parents=True,exist_ok=True)
    marker.write_text(json.dumps({'training_pid':process.pid,'process_created':created,'cmdline':command,'run':str(run),'comparison_script_sha256':sha256(Path(__file__)),'data_manifest_sha256':sha256(data/'build_manifest.json'),'frame_reference_run':str(args.frame_reference_run) if args.frame_reference_run else None,'frame_reference_label':args.frame_reference_label,'job_timeout_seconds':args.job_timeout_seconds,'legacy_source_counts':args.legacy_source_counts},indent=2),encoding='utf-8')
    print(f'Waiting for verified training PID {process.pid}',flush=True)
    while process.is_running() and process.create_time()==created:
        time.sleep(5)
    meta=run/'experiment_meta.json'
    if not meta.exists(): raise SystemExit('Training ended without completion metadata; do not restart automatically')
    identity=json.loads(meta.read_text(encoding='utf-8'))
    for name in ['best.pt','last.pt']:
        if sha256(run/'weights'/name)!=identity[name+'_sha256']: raise SystemExit('Completed checkpoint mismatch')
    models={'v3':ROOT/'model_work/runs/fire-binary-user-video-v3/weights/best.pt',
            args.candidate_label+'_best':run/'weights/best.pt',args.candidate_label+'_last':run/'weights/last.pt'}
    comparison={}
    for label,weights in models.items():
        jobs=[('frames','eval_hardcase_frames.py',['--data',str(data)],'summary.json'),
              ('blue_primary','eval_blue_localization.py',[],'summary.json'),
              ('legacy286','eval_fire_fulltest.py',[],'metrics_summary.json'),
              ('commons13','eval_fire_external_diagnostic.py',['--comparison-data',str(data/'build_manifest.json')],'summary.json'),
              ('old_videos','eval_development_videos.py',['--videos-dir',str(ROOT/'viedos')],'summary.json'),
              ('teammate_videos','eval_development_videos.py',['--videos-dir',str(ROOT/'viedos/视频/视频')],'summary.json')]
        comparison[label]={'weights_sha256':sha256(weights)}
        for kind,script,extra,filename in jobs:
            target=out/(label+'_'+kind)
            if target.exists(): raise SystemExit(f'Existing evaluation output: {target}')
            command=[sys.executable,'-B',str(SRC/script),'--weights',str(weights),'--out',str(target),*extra]
            print(f'Evaluating {label}: {kind}',flush=True)
            with (out/(label+'_'+kind+'.log')).open('w',encoding='utf-8') as log:
                result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=args.job_timeout_seconds)
            if result.returncode: raise SystemExit(f'Evaluation failed: {label}/{kind}; inspect log')
            comparison[label][kind]=json.loads((target/filename).read_text(encoding='utf-8'))
            (out/'comparison_partial.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2),encoding='utf-8')
    frame_references={}
    if args.frame_reference_run:
        reference=args.frame_reference_run.resolve()
        reference_meta=json.loads((reference/'experiment_meta.json').read_text(encoding='utf-8'))
        for name in ['best','last']:
            weights=reference/'weights'/(name+'.pt')
            if sha256(weights)!=reference_meta[name+'.pt_sha256']:
                raise SystemExit('Reference checkpoint identity mismatch')
            label=args.frame_reference_label+'_'+name
            target=out/(label+'_reference_frames')
            if target.exists(): raise SystemExit(f'Existing reference output: {target}')
            command=[sys.executable,'-B',str(SRC/'eval_hardcase_frames.py'),'--weights',str(weights),
                     '--data',str(data),'--out',str(target)]
            print(f'Evaluating frame reference {label}',flush=True)
            with (out/(label+'_reference_frames.log')).open('w',encoding='utf-8') as log:
                completed=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=args.job_timeout_seconds)
            if completed.returncode: raise SystemExit(f'Frame reference failed: {label}; inspect log')
            frame_references[label]=json.loads((target/'summary.json').read_text(encoding='utf-8'))
    legacy_sources=None
    if args.legacy_source_counts:
        model_list=out/'legacy_source_models.json'
        model_list.write_text(json.dumps([{'label':label,'weights':str(weights),'expected_sha256':sha256(weights)} for label,weights in models.items()],indent=2),encoding='utf-8')
        base=Path(json.loads((data/'build_manifest.json').read_text(encoding='utf-8'))['base'])
        target=out/'legacy_source_counts'
        if target.exists(): raise SystemExit(f'Existing evaluation output: {target}')
        command=[sys.executable,'-B',str(SRC/'eval_legacy_source_localization.py'),'--data',str(base),'--models',str(model_list),'--out',str(target)]
        with (out/'legacy_source_counts.log').open('w',encoding='utf-8') as log:
            completed=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=args.job_timeout_seconds)
        if completed.returncode: raise SystemExit('Legacy source comparison failed; inspect log')
        legacy_sources=json.loads((target/'summary.json').read_text(encoding='utf-8'))
    result={'role':'development comparisons; no independent acceptance, no board deployment',
            'data_manifest_sha256':sha256(data/'build_manifest.json'),'models':comparison,
            'frame_references':frame_references,'legacy_source_counts':legacy_sources}

    (out/'comparison_complete.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    compact={}
    for label,r in comparison.items():
        compact[label]={'frames':r['frames']['by_origin'],'blue_primary_localized':r['blue_primary']['primary_localized_iou50'],
                        'legacy_map50':r['legacy286']['map50']}
    print(json.dumps(compact,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__': main()
