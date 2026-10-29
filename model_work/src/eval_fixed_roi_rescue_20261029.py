"""Evaluate fixed-region rescue on exposed project video frames.

Run full-frame baseline first; only an empty full-frame result triggers a second
model inference on a fixed broad region. Map crop predictions to full-frame pixels.
This is a development diagnostic, not alarm logic or independent testing.
"""
import argparse, hashlib, json, math
from pathlib import Path
import cv2, numpy as np, torch
from ultralytics import YOLO

NAMES=("v16_best","v18_last","v24_last")
FRACTIONS=(0.10,0.25,0.90,0.90)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for block in iter(lambda:f.read(1048576),b""): h.update(block)
 return h.hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding="utf8"))
def write(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
def read_records(d): return [json.loads(s) for s in (d/"per_frame.jsonl").read_text(encoding="utf8").splitlines()]
def validate(d):
 plan=read(d/"plan.json"); done=read(d/"comparison_complete.json")
 if sha(d/"plan.json")!=done["plan_sha256"] or sha(d/"per_frame.jsonl")!=done["per_frame_sha256"]: raise ValueError("Reference identity changed")
 return plan,done
def roi_xyxy(width,height):
 x1=math.floor(width*FRACTIONS[0]); y1=math.floor(height*FRACTIONS[1]); x2=math.ceil(width*FRACTIONS[2]); y2=math.ceil(height*FRACTIONS[3])
 if not 0<=x1<x2<=width or not 0<=y1<y2<=height: raise ValueError("Invalid ROI")
 return x1,y1,x2,y2
def map_boxes(boxes,roi):
 x1,y1,x2,y2=roi; width=x2-x1; height=y2-y1; result=[]
 for b in boxes:
  a,c,d,e=map(float,b.xyxy[0].tolist()); conf=float(b.conf.item())
  if not all(math.isfinite(x) for x in (a,c,d,e,conf)) or a<-.1 or c<-.1 or d>width+.1 or e>height+.1 or a>=d or c>=e: raise ValueError("Invalid crop coordinates")
  result.append(dict(confidence=conf,xyxy=[max(0,a)+x1,max(0,c)+y1,min(width,d)+x1,min(height,e)+y1]))
 return result
def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--samples",type=Path,required=True); p.add_argument("--full",type=Path,required=True); p.add_argument("--out",type=Path,required=True); a=p.parse_args()
 if a.out.exists(): raise ValueError("Existing output")
 sample_plan,sample_done=validate(a.samples); full_plan,full_done=validate(a.full)
 sample=read_records(a.samples); full=read_records(a.full)
 if len(sample)!=307 or len(full)!=734 or full_done["frames"]!=734 or len(sample_plan["videos"])!=9: raise ValueError("Reference inventory mismatch")
 if full_plan["source"]["sha256"]!=sample_plan["videos"][6]["sha256"] or full_plan["configuration"]!=sample_plan["configuration"]: raise ValueError("Reference scope mismatch")
 models={m["label"]:m for m in sample_plan["models"] if m["label"] in NAMES}; full_models={m["label"]:m for m in full_plan["models"]}
 if set(models)!=set(NAMES) or any(models[n]["expected_sha256"]!=full_models[n]["expected_sha256"] for n in NAMES): raise ValueError("Model identity differs")
 if any(sha(m["weights"])!=m["expected_sha256"] for m in models.values()): raise ValueError("Model file changed")
 for r in sample:
  if r["video_index"]==7:
   f=full[r["frame_index"]]
   if r["source_pixel_sha256"]!=f["source_pixel_sha256"] or any(r["predictions"][n]!=f["predictions"][n] for n in NAMES): raise ValueError("Full-video baseline differs")
 selected={}
 for r in sample:
  if r["video_index"]!=7: selected[(r["video_index"],r["frame_index"])]=r
 for r in full: selected[(7,r["frame_index"])]=r
 if len(selected)!=1014: raise ValueError("Unexpected source frame inventory")
 a.out.mkdir(parents=True); plan=dict(role="Fixed crop rescue diagnostic on exposed development media",policy="Only infer crop when full frame has no box",roi_fractions=FRACTIONS,models=[models[n] for n in NAMES],configuration=sample_plan["configuration"],videos=sample_plan["videos"],source_frame_counts="Video7 full 734, other eight videos 280 prior samples",sample_comparison_sha256=sha(a.samples/"comparison_complete.json"),full_comparison_sha256=sha(a.full/"comparison_complete.json"),script_sha256=sha(__file__),reserved_sources_used=False,independent_test=False,on_board=False); write(a.out/"plan.json",plan)
 torch.set_num_threads(2); nets={n:YOLO(models[n]["weights"]) for n in NAMES}
 if any(net.names!={0:"fire"} for net in nets.values()): raise ValueError("Unexpected class")
 results=[]; totals={n:{"full_empty":0,"roi_inferred":0,"roi_any_box":0} for n in NAMES}; seen=set()
 for number,source in enumerate(plan["videos"],1):
  video=Path(source["path"])
  if sha(video)!=source["sha256"]: raise ValueError("Source changed")
  wanted={i:r for (v,i),r in selected.items() if v==number}; cap=cv2.VideoCapture(str(video)); i=0; found=set()
  if not cap.isOpened(): raise ValueError("Video open failed")
  try:
   while True:
    ok,im=cap.read()
    if not ok: break
    if i in wanted:
     ref=wanted[i]; pixel=hashlib.sha256(np.ascontiguousarray(im).tobytes()).hexdigest()
     if pixel!=ref["source_pixel_sha256"]: raise ValueError("Source frame differs")
     height,width=im.shape[:2]; region=roi_xyxy(width,height); x1,y1,x2,y2=region; crop=im[y1:y2,x1:x2]; predictions={}
     for n in NAMES:
      base=ref["predictions"][n]; entry=dict(full_frame=base,crop_attempted=False,crop_predictions=[])
      if not base:
       totals[n]["full_empty"]+=1; totals[n]["roi_inferred"]+=1; entry["crop_attempted"]=True; b=nets[n].predict(crop,**plan["configuration"],verbose=False)[0].boxes; entry["crop_predictions"]=map_boxes(b,region)
       totals[n]["roi_any_box"]+=bool(entry["crop_predictions"])
      predictions[n]=entry
     results.append(dict(video_index=number,frame_index=i,time_seconds=ref["time_seconds"],width=width,height=height,source_pixel_sha256=pixel,roi_xyxy=region,predictions=predictions)); found.add(i)
    i+=1
  finally: cap.release()
  if found!=set(wanted) or sha(video)!=source["sha256"]: raise ValueError("Decoded samples missing or video changed")
  seen.update((number,x) for x in found); print(f"Finished video {number}/9, selected {len(found)} frames",flush=True)
 if seen!=set(selected) or len(results)!=1014 or any(sha(m["weights"])!=m["expected_sha256"] for m in models.values()): raise ValueError("Incomplete result or changed model")
 with (a.out/"per_frame.jsonl").open("w",encoding="utf8") as f:
  for r in results:f.write(json.dumps(r,ensure_ascii=False)+"\n")
 by_video={}
 for number in range(1,10):
  rows=[r for r in results if r["video_index"]==number]; by_video[str(number)]={n:dict(full_empty=sum(not r["predictions"][n]["full_frame"] for r in rows),crop_any=sum(bool(r["predictions"][n]["crop_predictions"]) for r in rows)) for n in NAMES}
 summary=dict(role=plan["role"],plan_sha256=sha(a.out/"plan.json"),per_frame_sha256=sha(a.out/"per_frame.jsonl"),frames=len(results),totals=totals,by_video=by_video,ground_truth_provided=False,independent_test=False,on_board=False,limits="Any-box rescue is not true-positive recovery; inspect false boxes, crop exclusions, and whole-video effects.")
 write(a.out/"comparison_complete.json",summary); print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
if __name__=="__main__": main()
