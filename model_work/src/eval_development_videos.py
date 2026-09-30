"""Registered development videos only; predictions and renderings, not accuracy."""
import argparse,csv,json
from pathlib import Path
import cv2
from ultralytics import YOLO
from data_integrity import sha256
ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--videos-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--sample-fps',type=float,default=1)
    args=p.parse_args()
    if args.out.exists(): raise SystemExit('Refusing overwrite')
    if not 0<args.sample_fps<=30: raise SystemExit('Invalid sample fps')
    known={r['sha256'] for r in json.loads((ROOT/'docs/data_exposure_registry.json').read_text(encoding='utf-8'))['videos']}
    files=sorted(args.videos_dir.glob('*.mp4'))
    if not files: raise SystemExit('No videos')
    identities={p:sha256(p) for p in files}
    if any(d not in known for d in identities.values()): raise SystemExit('Unregistered fresh media: use frozen fresh-video protocol')
    args.out.mkdir(parents=True);model=YOLO(str(args.weights));rows=[];clips=[]
    for path in files:
        cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS)
        w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not cap.isOpened() or fps<=0 or min(w,h)<=0: raise SystemExit('Bad video metadata')
        step=max(1,round(fps/args.sample_fps));index=0;local=[]
        writer=cv2.VideoWriter(str(args.out/(path.stem+'_boxed.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),fps/step,(w,h))
        if not writer.isOpened(): raise SystemExit('Cannot create rendered video')
        while True:
            ok,frame=cap.read()
            if not ok: break
            if index%step==0:
                pred=model.predict(frame,imgsz=640,conf=0.25,iou=0.6,device=0,verbose=False)[0]
                boxes=[{'confidence':round(float(b.conf.item()),4),'xyxy':[round(float(v),2) for v in b.xyxy[0].tolist()]} for b in pred.boxes]
                row={'video':path.name,'frame_index':index,'time_seconds':round(index/fps,3),
                     'width':w,'height':h,'boxes':len(boxes),'predictions':json.dumps(boxes)}
                rows.append(row);local.append(row)
                for b in boxes:
                    c=[int(v) for v in b['xyxy']];cv2.rectangle(frame,tuple(c[:2]),tuple(c[2:]),(255,0,0),3)
                    cv2.putText(frame,f'fire {b["confidence"]:.2f}',(c[0],max(25,c[1]-7)),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,0,0),2)
                writer.write(frame)
            index+=1
        cap.release();writer.release()
        clips.append({'video':path.name,'sha256':identities[path],'fps':fps,'step':step,'sampled':len(local),
                      'frames_with_any_box':sum(r['boxes']>0 for r in local)})
    with (args.out/'per_frame.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={'weights_sha256':sha256(args.weights),'config':{'imgsz':640,'conf':0.25,'nms_iou':0.6,'requested_sample_fps':args.sample_fps},
             'clips':clips,'role':'development-exposed predictions; any-box ratios are not accuracy'}
    (args.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__': main()
