"""One frozen pass on a reviewed public-video clip; local details, aggregate summary.
Requires truth for every fixed sample before inference and never overwrites output.
"""
import argparse,hashlib,json,re
from datetime import datetime,timezone
from pathlib import Path
import cv2
from data_integrity import matches,reject_exposed,sha256
ROOT=Path(__file__).resolve().parents[2]

def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--probe',type=Path,required=True)
    ap.add_argument('--clip',required=True)
    ap.add_argument('--truth',type=Path,required=True)
    ap.add_argument('--expected-plan-sha256',required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--device',default='0')
    a=ap.parse_args();a.probe=a.probe.resolve();a.out=a.out.resolve()
    if a.out.exists():raise SystemExit('Refusing overwrite: inspect existing pass')
    plan_path=a.probe/'frozen_plan.json';plan_hash=sha256(plan_path)
    if plan_hash!=a.expected_plan_sha256:raise SystemExit('Frozen plan identity mismatch')
    plan=json.loads(plan_path.read_text(encoding='utf-8'));receipt=json.loads((a.probe/'acquisition.json').read_text(encoding='utf-8'));truth=json.loads(a.truth.read_text(encoding='utf-8'));truth_hash=sha256(a.truth)
    if receipt['plan_sha256']!=plan_hash:raise SystemExit('Acquisition linked to another plan')
    source=next((r for r in receipt['sources'] if r['id']==a.clip),None)
    if source is None:raise SystemExit('Unknown clip')
    spec=next((s for s in plan['sources'] if s['id']==a.clip),None)
    if spec is None:raise SystemExit('Clip not predeclared')
    video=(a.probe/source['file']).resolve()
    if not video.is_relative_to(a.probe):raise SystemExit('Video path escapes probe directory')
    if sha256(video)!=source['sha256'] or hashlib.sha1(video.read_bytes()).hexdigest()!=spec['expected_sha1']:raise SystemExit('Original media changed')
    reject_exposed(video,ROOT/'docs/data_exposure_registry.json')
    if truth['source_sha256']!=source['sha256'] or truth['status']!='approved_no_visible_flame_for_sampled_frames_only' or not truth['reviewed_before_predictions']:raise SystemExit('Truth not approved for this source')
    if any(truth['gt_boxes'].values()):raise SystemExit('This entry point supports reviewed no-visible-flame probes only')
    labels=set()
    for m in plan['models']:
        if not re.fullmatch(r'[a-zA-Z0-9_-]+',m['label']) or m['label'] in labels:raise SystemExit('Unsafe/duplicate model label')
        labels.add(m['label'])
        if sha256(Path(m['weights']))!=m['expected_sha256']:raise SystemExit('Frozen checkpoint changed')
    cfg=plan['configuration'];cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS);nominal=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or not 0<fps<=240 or not 0<cfg['target_sample_fps']<=fps:raise SystemExit('Invalid sampling settings')
    step=max(1,round(fps/cfg['target_sample_fps']));frames=[];index=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        if index%step==0:frames.append((index,frame))
        index+=1
    cap.release();indices=[i for i,_ in frames]
    if indices!=truth['indices'] or set(map(str,indices))!=set(truth['gt_boxes']) or index!=truth['decoded_frames'] or fps!=truth['fps']:raise SystemExit('Truth does not cover actual fixed samples')
    if not frames:raise SystemExit('No frames')
    a.out.mkdir(parents=True)
    launch={'started_at_utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':plan_hash,'source_sha256':source['sha256'],'truth_sha256':truth_hash,'script_sha256':sha256(Path(__file__)),'models':[{k:m[k] for k in ['label','expected_sha256']} for m in plan['models']]}
    dump(a.out/'launch.json',launch)
    from ultralytics import YOLO
    result={'role':'external short-clip visible-flame probe; not full event or long-duration acceptance','source_page':source['source_page'],'source_sha256':source['sha256'],'plan_sha256':plan_hash,'truth_sha256':truth_hash,'configuration':cfg,'fps':fps,'step':step,'actual_sample_fps':fps/step,'nominal_frames':nominal,'decoded_frames':index,'sampled_frames':len(frames),'truth':'no visible flame in all sampled full frames; burner occluded; not event safe','models':{}}
    for m in plan['models']:
        model=YOLO(m['weights']);rows=[];run=longest=0
        for index,frame in frames:
            if sha256(Path(m['weights']))!=m['expected_sha256']:raise SystemExit('Checkpoint changed during pass')
            pred=model.predict(frame,imgsz=cfg['imgsz'],conf=cfg['conf'],iou=cfg['nms_iou'],device=a.device,verbose=False,save=False)[0]
            boxes=[]
            for b in pred.boxes:
                if int(b.cls.item())!=0:raise SystemExit('Unexpected detection class')
                boxes.append({'confidence':float(b.conf.item()),'xyxy':[float(v) for v in b.xyxy[0].tolist()]})
            score=matches([], [b['xyxy'] for b in boxes],threshold=cfg['match_iou']);rows.append({'frame_index':index,'time_seconds':index/fps,**score,'predictions':boxes})
            run=run+1 if boxes else 0;longest=max(longest,run)
            if boxes:
                overlay=frame.copy()
                for b in boxes:cv2.rectangle(overlay,tuple(map(int,b['xyxy'][:2])),tuple(map(int,b['xyxy'][2:])),(255,0,0),2)
                cv2.imwrite(str(a.out/(m['label']+'_frame_'+str(index)+'.jpg')),overlay)
        dump(a.out/(m['label']+'_local_predictions.json'),rows)
        result['models'][m['label']]={'weights_sha256':m['expected_sha256'],'frames_with_fp':sum(r['fp']>0 for r in rows),'fp_boxes':sum(r['fp'] for r in rows),'longest_consecutive_sampled_fp_frames':longest,'longest_sampled_fp_span_s':max(0,longest-1)*step/fps}
        dump(a.out/'summary_partial.json',result)
    if sha256(video)!=source['sha256'] or sha256(plan_path)!=plan_hash or sha256(a.truth)!=truth_hash:raise SystemExit('Input changed during pass')
    for m in plan['models']:
        if sha256(Path(m['weights']))!=m['expected_sha256']:raise SystemExit('Checkpoint changed during pass')
    result['completed_at_utc']=datetime.now(timezone.utc).isoformat();dump(a.out/'summary.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
