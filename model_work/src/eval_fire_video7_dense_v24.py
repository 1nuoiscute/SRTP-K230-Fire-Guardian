"""Paired video-7 development diagnostic: dense windows and historical parity."""
import argparse, hashlib, json, math
from pathlib import Path
import cv2, numpy as np, torch
from ultralytics import YOLO

NAMES=("v16_best", "v18_last", "v21_last", "v22_last", "v24_best", "v24_last")
WINDOWS=((0.5,1.5),(8.5,9.5))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def write(p, x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def main():
 p=argparse.ArgumentParser(); p.add_argument("--v24",type=Path,required=True); p.add_argument("--v22",type=Path,required=True); p.add_argument("--out",type=Path,required=True); a=p.parse_args()
 if a.out.exists(): raise ValueError("Existing output")
 plans={}; old={}; done={}
 for label,d in (("v24",a.v24),("v22",a.v22)):
  plans[label]=read(d/"plan.json"); done[label]=read(d/"comparison_complete.json")
  if sha(d/"plan.json")!=done[label]["plan_sha256"] or sha(d/"per_frame.jsonl")!=done[label]["per_frame_sha256"]: raise ValueError("Reference hash changed")
  old[label]={r["frame_index"]:r for r in (json.loads(s) for s in (d/"per_frame.jsonl").read_text(encoding="utf-8").splitlines()) if r["video_index"]==7}
 if len(old["v24"])!=27 or set(old["v24"])!=set(old["v22"]): raise ValueError("Reference frames differ")
 source=plans["v24"]["videos"][6]; video=Path(source["path"])
 if source["sha256"]!=plans["v22"]["videos"][6]["sha256"] or sha(video)!=source["sha256"]: raise ValueError("Video hash changed")
 models={m["label"]:m for m in plans["v24"]["models"]}; models.update({m["label"]:m for m in plans["v22"]["models"] if m["label"]=="v22_last"}); models={n:models[n] for n in NAMES}
 if any(sha(m["weights"])!=m["expected_sha256"] for m in models.values()): raise ValueError("Model hash changed")
 config=plans["v24"]["configuration"]
 if config!=plans["v22"]["configuration"]: raise ValueError("Inference settings differ")
 cap=cv2.VideoCapture(str(video)); fps=cap.get(cv2.CAP_PROP_FPS); count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 if not cap.isOpened() or fps<=0 or count!=done["v24"]["clips"][6]["decoded_frames"]: raise ValueError("Video metadata changed")
 step=max(1,round(fps/5)); indices=set(range(0,count,step))|set(old["v24"])
 for lo,hi in WINDOWS: indices.update(range(max(0,math.ceil(lo*fps)),min(count,math.floor(hi*fps)+1)))
 review=[round(t*fps) for t in (0.8,1.0,1.2,8.8,9.0,9.2)]; indices.update(review)
 a.out.mkdir(parents=True); plan=dict(role="Exposed development video; no event truth",source=source,models=list(models.values()),configuration=config,windows=WINDOWS,fps=fps,step=step,indices=sorted(indices),reference_v24_sha256=sha(a.v24/"comparison_complete.json"),reference_v22_sha256=sha(a.v22/"comparison_complete.json"),script_sha256=sha(__file__),reserved_sources_used=False); write(a.out/"plan.json",plan)
 torch.set_num_threads(2); nets={n:YOLO(m["weights"]) for n,m in models.items()}
 if any(n.names!={0:"fire"} for n in nets.values()): raise ValueError("Wrong model class")
 records=[]; frames={}; index=0; parity=0
 try:
  while True:
   ok,im=cap.read()
   if not ok: break
   if index in indices:
    pred={}
    for name,net in nets.items():
     result=net.predict(im,**config,verbose=False)[0]; pred[name]=[dict(confidence=float(b.conf.item()),xyxy=b.xyxy[0].tolist()) for b in result.boxes]
    pix=hashlib.sha256(np.ascontiguousarray(im).tobytes()).hexdigest(); row=dict(frame_index=index,time_seconds=index/fps,source_pixel_sha256=pix,predictions=pred); records.append(row)
    if index in old["v24"]:
     if any(old[x][index]["source_pixel_sha256"]!=pix or old[x][index]["video_sha256"]!=source["sha256"] for x in old): raise ValueError("Frame mismatch")
     for name in NAMES:
      ref=old["v22" if name=="v22_last" else "v24"][index]
      if pred[name]!=ref["predictions"][name]: raise ValueError(f"Prediction mismatch {name} {index}")
      parity+=1
    if index in review: frames[index]=im.copy()
    if len(records)%50==0: print(f"Processed {len(records)}/{len(indices)} frames",flush=True)
   index+=1
 finally: cap.release()
 if index!=count or len(records)!=len(indices) or parity!=27*len(NAMES) or sha(video)!=source["sha256"] or any(sha(m["weights"])!=m["expected_sha256"] for m in models.values()): raise ValueError("Incomplete or changed inputs")
 with (a.out/"per_frame.jsonl").open("w",encoding="utf-8") as f:
  for row in records: f.write(json.dumps(row,ensure_ascii=False)+"\n")
 byid={r["frame_index"]:r for r in records}; sheet=[]
 for i in review:
  im=frames[i]; row=byid[i]; h,w=im.shape[:2]; scale=min(460/w,400/h); scaled=cv2.resize(im,(round(w*scale),round(h*scale))); sh,sw=scaled.shape[:2]; tiles=[]
  for name in NAMES:
   tile=np.full((440,480,3),24,np.uint8); left=(480-sw)//2; top=40+(400-sh)//2; tile[top:top+sh,left:left+sw]=scaled
   for box in row["predictions"][name]:
    x1,y1,x2,y2=box["xyxy"]; cv2.rectangle(tile,(round(x1*scale+left),round(y1*scale+top)),(round(x2*scale+left),round(y2*scale+top)),(255,90,0),2)
   cv2.putText(tile,f"{name} frame={i} t={row['time_seconds']:.2f}s",(5,23),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1); tiles.append(tile)
  sheet.append(np.concatenate(tiles,axis=1))
 if not cv2.imwrite(str(a.out/"fixed_window_review.jpg"),np.concatenate(sheet,axis=0)): raise ValueError("Image write failed")
 stats=[]
 for lo,hi in WINDOWS:
  selected=[r for r in records if lo<=r["time_seconds"]<=hi]; stats.append(dict(start=lo,end=hi,frames=len(selected),empty_by_model={n:sum(not r["predictions"][n] for r in selected) for n in NAMES},v24_last_empty_indices=[r["frame_index"] for r in selected if not r["predictions"]["v24_last"]]))
 summary=dict(role=plan["role"],plan_sha256=sha(a.out/"plan.json"),per_frame_sha256=sha(a.out/"per_frame.jsonl"),review_image_sha256=sha(a.out/"fixed_window_review.jpg"),processed_frames=len(records),prior_samples_checked=27,prior_predictions_exact=parity,window_stats=stats,independent_test=False,ground_truth_provided=False,on_board=False)
 write(a.out/"comparison_complete.json",summary); print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
if __name__=="__main__": main()
