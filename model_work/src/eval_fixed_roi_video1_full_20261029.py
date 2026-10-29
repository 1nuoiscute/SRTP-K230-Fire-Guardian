"""Full-frame then conditional fixed-ROI inference on exposed video 1.

Records all model decisions and validates old 1-fps reference predictions;
no visual event truth, model adoption, hardware timing or independent test.
"""
import argparse, hashlib, json
from pathlib import Path
import cv2, numpy as np, torch
from ultralytics import YOLO
from eval_fixed_roi_rescue_20261029 import NAMES, FRACTIONS, sha, read, write, validate, read_records, roi_xyxy, map_boxes

def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--samples",type=Path,required=True); p.add_argument("--out",type=Path,required=True); a=p.parse_args()
 if a.out.exists(): raise ValueError("Existing output")
 refplan,refdone=validate(a.samples); ref={r["frame_index"]:r for r in read_records(a.samples) if r["video_index"]==1}; clip=refdone["clips"][0]; entry=refplan["videos"][0]; video=Path(entry["path"])
 if len(ref)!=32 or sha(video)!=entry["sha256"]: raise ValueError("Reference video or frame inventory differs")
 models={m["label"]:m for m in refplan["models"] if m["label"] in NAMES}
 if set(models)!=set(NAMES) or any(sha(m["weights"])!=m["expected_sha256"] for m in models.values()): raise ValueError("Checkpoint changed")
 cap=cv2.VideoCapture(str(video)); fps=cap.get(cv2.CAP_PROP_FPS); count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 if not cap.isOpened() or count!=clip["decoded_frames"] or abs(fps-clip["fps"])>1e-6: raise ValueError("Video metadata changed")
 a.out.mkdir(parents=True); plan=dict(role="All-frame exposed video1 baseline and conditional ROI, without event truth",source=entry,models=[models[n] for n in NAMES],configuration=refplan["configuration"],roi_fractions=FRACTIONS,fps=fps,frame_count=count,sample_comparison_sha256=sha(a.samples/"comparison_complete.json"),script_sha256=sha(__file__),roi_helper_sha256=sha(Path(__file__).with_name("eval_fixed_roi_rescue_20261029.py")),reserved_sources_used=False); write(a.out/"plan.json",plan)
 torch.set_num_threads(2); nets={n:YOLO(models[n]["weights"]) for n in NAMES}
 if any(net.names!={0:"fire"} for net in nets.values()): raise ValueError("Unexpected class")
 rows=[]; review=[]; parity=0; index=0
 try:
  while True:
   ok,im=cap.read()
   if not ok: break
   pixel=hashlib.sha256(np.ascontiguousarray(im).tobytes()).hexdigest(); h,w=im.shape[:2]; roi=roi_xyxy(w,h); x1,y1,x2,y2=roi; crop=im[y1:y2,x1:x2]; predictions={}
   for name,net in nets.items():
    detected=net.predict(im,**plan["configuration"],verbose=False)[0].boxes
    base=[dict(confidence=float(b.conf.item()),xyxy=b.xyxy[0].tolist()) for b in detected]
    if index in ref:
     if ref[index]["source_pixel_sha256"]!=pixel or base!=ref[index]["predictions"][name]: raise ValueError(f"Reference mismatch {name} {index}")
     parity+=1
    extra=[]
    if not base: extra=map_boxes(net.predict(crop,**plan["configuration"],verbose=False)[0].boxes,roi)
    predictions[name]=dict(full_frame=base,crop_attempted=not bool(base),crop_predictions=extra)
   rows.append(dict(frame_index=index,time_seconds=index/fps,width=w,height=h,source_pixel_sha256=pixel,roi_xyxy=roi,predictions=predictions))
   if index in ref:
    scale=min(230/w,340/h); pic=cv2.resize(im,(round(w*scale),round(h*scale))); ph,pw=pic.shape[:2]; tile=np.full((380,240,3),24,np.uint8); left=(240-pw)//2; top=30+(340-ph)//2; tile[top:top+ph,left:left+pw]=pic; cv2.putText(tile,f"frame {index} t={index/fps:.1f}s",(4,20),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1); review.append(tile)
   index+=1
   if index%150==0: print(f"Processed video1 {index}/{count} frames",flush=True)
 finally: cap.release()
 if index!=count or parity!=32*len(NAMES) or len(review)!=32 or sha(video)!=entry["sha256"] or any(sha(m["weights"])!=m["expected_sha256"] for m in models.values()): raise ValueError("Incomplete or changed inputs")
 with (a.out/"per_frame.jsonl").open("w",encoding="utf8") as f:
  for row in rows: f.write(json.dumps(row,ensure_ascii=False)+"\n")
 pages=[]
 for start in range(0,len(review),8):
  path=a.out/f"source_samples_{start//8+1:02}.jpg"
  if not cv2.imwrite(str(path),np.concatenate(review[start:start+8],axis=1)): raise ValueError("Review write failed")
  pages.append(dict(file=path.name,sha256=sha(path)))
 counts={}
 for name in NAMES:
  counts[name]=dict(full_any=sum(bool(r["predictions"][name]["full_frame"]) for r in rows),crop_any=sum(bool(r["predictions"][name]["crop_predictions"]) for r in rows),crop_box_frames=[r["frame_index"] for r in rows if r["predictions"][name]["crop_predictions"]])
 result=dict(role=plan["role"],plan_sha256=sha(a.out/"plan.json"),per_frame_sha256=sha(a.out/"per_frame.jsonl"),frames=len(rows),reference_predictions_exact=parity,models=counts,source_review_pages=pages,ground_truth_provided=False,independent_test=False,on_board=False); write(a.out/"comparison_complete.json",result); print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
if __name__=="__main__": main()
