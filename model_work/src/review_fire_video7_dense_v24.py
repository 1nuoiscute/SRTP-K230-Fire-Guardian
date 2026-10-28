"""Render every v24-last empty sampled frame from a completed dense comparison."""
import argparse, hashlib, json
from pathlib import Path
import cv2, numpy as np

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def main():
 p=argparse.ArgumentParser(); p.add_argument("--comparison",type=Path,required=True); p.add_argument("--out",type=Path,required=True); a=p.parse_args()
 if a.out.exists(): raise ValueError("Existing review")
 plan=read(a.comparison/"plan.json"); done=read(a.comparison/"comparison_complete.json")
 if sha(a.comparison/"plan.json")!=done["plan_sha256"] or sha(a.comparison/"per_frame.jsonl")!=done["per_frame_sha256"]: raise ValueError("Comparison identity changed")
 rows=[json.loads(s) for s in (a.comparison/"per_frame.jsonl").read_text(encoding="utf-8").splitlines()]; selected={r["frame_index"]:r for r in rows if not r["predictions"]["v24_last"]}; labels=[m["label"] for m in plan["models"]]; video=Path(plan["source"]["path"])
 if sha(video)!=plan["source"]["sha256"]: raise ValueError("Source changed")
 cap=cv2.VideoCapture(str(video)); rendered=[]; found=set(); idx=0
 if not cap.isOpened(): raise ValueError("Video not open")
 try:
  while True:
   ok,im=cap.read()
   if not ok: break
   if idx in selected:
    row=selected[idx]; pix=hashlib.sha256(np.ascontiguousarray(im).tobytes()).hexdigest()
    if pix!=row["source_pixel_sha256"]: raise ValueError("Source pixel mismatch")
    h,w=im.shape[:2]; scale=min(460/w,400/h); scaled=cv2.resize(im,(round(w*scale),round(h*scale))); sh,sw=scaled.shape[:2]; tiles=[]
    for label in labels:
     tile=np.full((440,480,3),24,np.uint8); left=(480-sw)//2; top=40+(400-sh)//2; tile[top:top+sh,left:left+sw]=scaled
     for box in row["predictions"][label]:
      x1,y1,x2,y2=box["xyxy"]; cv2.rectangle(tile,(round(x1*scale+left),round(y1*scale+top)),(round(x2*scale+left),round(y2*scale+top)),(255,90,0),2)
     cv2.putText(tile,f"{label} frame={idx} t={row['time_seconds']:.2f}s",(5,23),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1); tiles.append(tile)
    rendered.append(np.concatenate(tiles,axis=1)); found.add(idx)
   idx+=1
 finally: cap.release()
 if found!=set(selected) or sha(video)!=plan["source"]["sha256"]: raise ValueError("Missing frames or source changed")
 a.out.mkdir(parents=True); sheets=[]
 for start in range(0,len(rendered),4):
  path=a.out/f"sheet_{start//4+1:02}.jpg"
  if not cv2.imwrite(str(path),np.concatenate(rendered[start:start+4],axis=0)): raise ValueError("Image write failed")
  sheets.append(dict(file=path.name,sha256=sha(path)))
 write=dict(role="All v24-last empty samples; box absence is not event truth",comparison_sha256=sha(a.comparison/"comparison_complete.json"),script_sha256=sha(__file__),frame_indices=sorted(selected),sheets=sheets); (a.out/"review_inputs.json").write_text(json.dumps(write,indent=2)+"\n",encoding="utf-8"); print(f"Prepared {len(selected)} empty sampled frames on {len(sheets)} pages")
if __name__=="__main__": main()
