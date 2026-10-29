"""Render every sampled frame with a new ROI-only box for manual review."""
import argparse,hashlib,json
from pathlib import Path
import cv2,numpy as np

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""): h.update(b)
 return h.hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding="utf8"))
def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--comparison",type=Path,required=True); p.add_argument("--out",type=Path,required=True); a=p.parse_args()
 if a.out.exists(): raise ValueError("Existing output")
 plan=read(a.comparison/"plan.json"); done=read(a.comparison/"comparison_complete.json")
 if sha(a.comparison/"plan.json")!=done["plan_sha256"] or sha(a.comparison/"per_frame.jsonl")!=done["per_frame_sha256"]: raise ValueError("Comparison changed")
 rows=[json.loads(s) for s in (a.comparison/"per_frame.jsonl").read_text(encoding="utf8").splitlines()]; chosen={(r["video_index"],r["frame_index"]):r for r in rows if any(x["crop_predictions"] for x in r["predictions"].values())}; labels=[m["label"] for m in plan["models"]]; rendered=[]; found=set()
 for video_index,entry in enumerate(plan["videos"],1):
  wanted={i:r for (v,i),r in chosen.items() if v==video_index}
  if not wanted: continue
  source=Path(entry["path"])
  if sha(source)!=entry["sha256"]: raise ValueError("Video changed")
  cap=cv2.VideoCapture(str(source)); index=0
  if not cap.isOpened(): raise ValueError("Video open failed")
  try:
   while True:
    ok,im=cap.read()
    if not ok: break
    if index in wanted:
     r=wanted[index]
     if hashlib.sha256(np.ascontiguousarray(im).tobytes()).hexdigest()!=r["source_pixel_sha256"]: raise ValueError("Frame pixel mismatch")
     h,w=im.shape[:2]; scale=min(460/w,400/h); pic=cv2.resize(im,(round(w*scale),round(h*scale))); ph,pw=pic.shape[:2]; tiles=[]
     for label in ["source_roi"]+labels:
      tile=np.full((440,480,3),24,np.uint8); left=(480-pw)//2; top=40+(400-ph)//2; tile[top:top+ph,left:left+pw]=pic
      if label=="source_roi":
       x1,y1,x2,y2=r["roi_xyxy"]; cv2.rectangle(tile,(round(x1*scale+left),round(y1*scale+top)),(round(x2*scale+left),round(y2*scale+top)),(0,200,255),2)
      else:
       data=r["predictions"][label]; boxes=data["crop_predictions"] if data["crop_attempted"] else data["full_frame"]; color=(0,200,255) if data["crop_attempted"] else (255,100,0)
       for b in boxes:
        x1,y1,x2,y2=b["xyxy"]; cv2.rectangle(tile,(round(x1*scale+left),round(y1*scale+top)),(round(x2*scale+left),round(y2*scale+top)),color,2)
      cv2.putText(tile,f"{label} vid={video_index} frame={index} t={r['time_seconds']:.2f}",(5,24),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1); tiles.append(tile)
     rendered.append((r,np.concatenate(tiles,axis=1))); found.add((video_index,index))
    index+=1
  finally: cap.release()
  if sha(source)!=entry["sha256"]: raise ValueError("Video changed during review")
 if found!=set(chosen): raise ValueError("Missing review frames")
 a.out.mkdir(parents=True); sheets=[]
 for start in range(0,len(rendered),6):
  path=a.out/f"sheet_{start//6+1:02}.jpg"
  if not cv2.imwrite(str(path),np.concatenate([x[1] for x in rendered[start:start+6]],axis=0)): raise ValueError("Image write failed")
  sheets.append(dict(file=path.name,sha256=sha(path),frames=[[x[0]["video_index"],x[0]["frame_index"]] for x in rendered[start:start+6]]))
 data=dict(role="All newly present crop boxes; no correctness implied",comparison_sha256=sha(a.comparison/"comparison_complete.json"),per_frame_sha256=sha(a.comparison/"per_frame.jsonl"),script_sha256=sha(__file__),frame_count=len(rendered),sheets=sheets); (a.out/"review_inputs.json").write_text(json.dumps(data,indent=2)+"\n",encoding="utf8"); print(f"Prepared {len(rendered)} frames / {len(sheets)} review pages")
if __name__=="__main__": main()
