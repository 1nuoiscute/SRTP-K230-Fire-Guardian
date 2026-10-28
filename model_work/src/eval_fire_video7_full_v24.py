"""Compare all frames of exposed video 7; reuse identity-checked dense results."""
import argparse, hashlib, json
from pathlib import Path
import cv2, numpy as np, torch
from ultralytics import YOLO

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding="utf8"))
def write(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
def main():
 p=argparse.ArgumentParser(); p.add_argument("--early",type=Path,required=True); p.add_argument("--late",type=Path,required=True); p.add_argument("--out",type=Path,required=True); a=p.parse_args()
 if a.out.exists(): raise ValueError("Existing output")
 plans=[]; caches=[]
 for d in (a.early,a.late):
  plan=read(d/"plan.json"); done=read(d/"comparison_complete.json")
  if sha(d/"plan.json")!=done["plan_sha256"] or sha(d/"per_frame.jsonl")!=done["per_frame_sha256"] or done["prior_predictions_exact"]!=162: raise ValueError("Prior comparison identity mismatch")
  rows=[json.loads(x) for x in (d/"per_frame.jsonl").read_text(encoding="utf8").splitlines()]; cache={r["frame_index"]:r for r in rows}
  if len(rows)!=done["processed_frames"] or len(cache)!=len(rows): raise ValueError("Duplicate or incomplete cache")
  plans.append(plan); caches.append(cache)
 for key in ("source","models","configuration","fps"):
  if plans[0][key]!=plans[1][key]: raise ValueError("Comparison inputs differ")
 common=set(caches[0]) & set(caches[1])
 if any(caches[0][i]!=caches[1][i] for i in common): raise ValueError("Overlapping cached predictions differ")
 cache={**caches[0],**caches[1]}; plan=plans[0]; source=Path(plan["source"]["path"]); models=plan["models"]; config=plan["configuration"]
 if sha(source)!=plan["source"]["sha256"] or any(sha(m["weights"])!=m["expected_sha256"] for m in models): raise ValueError("Video or model hash changed")
 cap=cv2.VideoCapture(str(source)); fps=cap.get(cv2.CAP_PROP_FPS); count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 if not cap.isOpened() or fps!=plan["fps"] or count!=734: raise ValueError("Video metadata differs")
 a.out.mkdir(parents=True); runplan=dict(role="All-frame exposed development video, no event truth",source=plan["source"],models=models,configuration=config,fps=fps,frame_count=count,cached_frames=len(cache),overlap_frames=len(common),early_complete_sha256=sha(a.early/"comparison_complete.json"),late_complete_sha256=sha(a.late/"comparison_complete.json"),script_sha256=sha(__file__),reserved_sources_used=False); write(a.out/"plan.json",runplan)
 torch.set_num_threads(2); nets={m["label"]:YOLO(m["weights"]) for m in models}
 if any(net.names!={0:"fire"} for net in nets.values()): raise ValueError("Wrong class")
 rows=[]; index=0
 try:
  while True:
   ok,im=cap.read()
   if not ok: break
   pix=hashlib.sha256(np.ascontiguousarray(im).tobytes()).hexdigest()
   if index in cache:
    row=cache[index]
    if row["source_pixel_sha256"]!=pix or row["frame_index"]!=index: raise ValueError("Cached decoded frame differs")
   else:
    pred={}
    for name,net in nets.items():
     result=net.predict(im,**config,verbose=False)[0]; pred[name]=[dict(confidence=float(b.conf.item()),xyxy=b.xyxy[0].tolist()) for b in result.boxes]
    row=dict(frame_index=index,time_seconds=index/fps,source_pixel_sha256=pix,predictions=pred)
   rows.append(row); index+=1
   if index%150==0: print(f"Processed {index}/{count} frames",flush=True)
 finally: cap.release()
 if index!=count or sha(source)!=plan["source"]["sha256"] or any(sha(m["weights"])!=m["expected_sha256"] for m in models): raise ValueError("Incomplete or changed input")
 with (a.out/"per_frame.jsonl").open("w",encoding="utf8") as f:
  for row in rows: f.write(json.dumps(row,ensure_ascii=False)+"\n")
 stats={}
 for m in models:
  name=m["label"]; empty=[r["frame_index"] for r in rows if not r["predictions"][name]]; runs=[]
  for i in empty:
   if not runs or i!=runs[-1][-1]+1: runs.append([i])
   else: runs[-1].append(i)
  stats[name]=dict(empty_frames=len(empty),empty_runs=[dict(start=x[0],end=x[-1],frames=len(x),seconds=len(x)/fps) for x in runs])
 summary=dict(role=runplan["role"],plan_sha256=sha(a.out/"plan.json"),per_frame_sha256=sha(a.out/"per_frame.jsonl"),frames=len(rows),cached_frames=len(cache),newly_inferred_frames=len(rows)-len(cache),stats=stats,independent_test=False,ground_truth_provided=False,on_board=False,limits="Empty predictions may mean visible flame, occlusion, out-of-view, or no visible flame; inspect frames before interpreting.")
 write(a.out/"comparison_complete.json",summary); print(json.dumps({k:v for k,v in summary.items() if k!="stats"},indent=2),flush=True)
if __name__=="__main__": main()
