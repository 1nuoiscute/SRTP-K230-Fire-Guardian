"""Render every newly boxed empty-label development image for actual review."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from eval_fixed_roi_rescue_20261029 import sha,read

def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--comparison",type=Path,required=True); p.add_argument("--out",type=Path,required=True); a=p.parse_args()
 if a.out.exists(): raise ValueError("Existing review")
 plan=read(a.comparison/"plan.json"); done=read(a.comparison/"comparison_complete.json")
 if sha(a.comparison/"plan.json")!=done["plan_sha256"] or sha(a.comparison/"per_image.jsonl")!=done["per_image_sha256"]: raise ValueError("Comparison identity changed")
 rows=[json.loads(s) for s in (a.comparison/"per_image.jsonl").read_text(encoding="utf8").splitlines()]; selected=[r for r in rows if any(v["crop_predictions"] for v in r["predictions"].values())]; names=[m["label"] for m in plan["models"]]; rendered=[]
 for index,r in enumerate(selected):
  path=Path(r["image"])
  if sha(path)!=r["image_sha256"]: raise ValueError("Image changed")
  im=cv2.imread(str(path))
  if im is None or [im.shape[1],im.shape[0]]!=[r["width"],r["height"]]: raise ValueError("Image changed")
  h,w=im.shape[:2]; scale=min(460/w,400/h); pic=cv2.resize(im,(round(w*scale),round(h*scale))); ph,pw=pic.shape[:2]; tiles=[]
  for name in ["source_roi"]+names:
   tile=np.full((440,480,3),24,np.uint8); left=(480-pw)//2; top=40+(400-ph)//2; tile[top:top+ph,left:left+pw]=pic
   if name=="source_roi":
    x1,y1,x2,y2=r["roi_xyxy"]; cv2.rectangle(tile,(round(x1*scale+left),round(y1*scale+top)),(round(x2*scale+left),round(y2*scale+top)),(0,200,255),2)
   else:
    pred=r["predictions"][name]; boxes=pred["crop_predictions"] if pred["crop_attempted"] else pred["full_frame"]; color=(0,200,255) if pred["crop_attempted"] else (255,100,0)
    for box in boxes:
     x1,y1,x2,y2=box["xyxy"]; cv2.rectangle(tile,(round(x1*scale+left),round(y1*scale+top)),(round(x2*scale+left),round(y2*scale+top)),color,2)
   cv2.putText(tile,f"{name} image={index+1} {r['provenance']}",(5,23),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1); tiles.append(tile)
  rendered.append((r,np.concatenate(tiles,axis=1)))
 a.out.mkdir(parents=True); sheets=[]
 for start in range(0,len(rendered),4):
  path=a.out/f"sheet_{start//4+1:02}.jpg"
  if not cv2.imwrite(str(path),np.concatenate([x[1] for x in rendered[start:start+4]],axis=0)): raise ValueError("Write failed")
  sheets.append(dict(file=path.name,sha256=sha(path),files=[x[0]["file"] for x in rendered[start:start+4]]))
 result=dict(role="All new crop-only predictions; no visual verdict implied",comparison_sha256=sha(a.comparison/"comparison_complete.json"),per_image_sha256=sha(a.comparison/"per_image.jsonl"),script_sha256=sha(__file__),images=len(rendered),sheets=sheets); (a.out/"review_inputs.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8"); print(f"Prepared {len(rendered)} images / {len(sheets)} pages")
if __name__=="__main__": main()
