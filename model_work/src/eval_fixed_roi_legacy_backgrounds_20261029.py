"""Assess conditional fixed-ROI rescue on exposed legacy empty-label images.

Original empty labels have varying semantic certainty. New boxes require visual
review; counts are not a real-world false-alarm rate. No held-out sources used.
"""
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
import cv2,torch
from ultralytics import YOLO
from eval_fixed_roi_rescue_20261029 import NAMES,FRACTIONS,sha,read,write,roi_xyxy,map_boxes

GROUPS=("ks_nofire_confirmed","object_pan","nofire_real_indoor")
def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--comparison",type=Path,required=True); p.add_argument("--out",type=Path,required=True); a=p.parse_args()
 if a.out.exists(): raise ValueError("Existing output")
 data=a.comparison/"legacy_sources"; complete=read(a.comparison/"comparison_complete.json"); legacy=read(data/"summary.json"); inputs=read(data/"local_inputs.json"); selected=[r for r in inputs if r["provenance"] in GROUPS]
 if len(selected)!=215 or Counter(r["provenance"] for r in selected)!={"ks_nofire_confirmed":75,"object_pan":40,"nofire_real_indoor":100} or any(r["gt"] or int(r["fire_boxes"]) for r in selected): raise ValueError("Unexpected reference inventory")
 listing={m["label"]:m for m in complete["plan"]["models"] if m["label"] in NAMES}
 if set(listing)!=set(NAMES) or any(sha(m["weights"])!=m["expected_sha256"] or legacy["models"][n]["weights_sha256"]!=m["expected_sha256"] for n,m in listing.items()): raise ValueError("Weight identity changed")
 predictions={}
 for n in NAMES:
  details=read(data/(n+"_local_predictions.json")); d={r["image"]:r for r in details}
  if len(d)!=286 or len(details)!=286: raise ValueError("Baseline cache inventory differs")
  for group in GROUPS:
   subset=[r for r in details if r["provenance"]==group]; old=legacy["models"][n]["by_provenance"][group]
   if len(subset)!=old["images"] or sum(bool(r["predictions"]) for r in subset)!=old["images_with_fp"] or sum(len(r["predictions"]) for r in subset)!=old["fp"]: raise ValueError("Baseline totals differ")
  predictions[n]=d
 for item in selected:
  image=Path(item["image"]); label=image.parents[2]/"labels/test"/(image.stem+".txt")
  if sha(image)!=item["sha256"] or sha(label)!=item["label_sha256"] or label.read_text(encoding="utf8").strip(): raise ValueError("Image or label changed")
  if any(predictions[n][item["file"]]["provenance"]!=item["provenance"] for n in NAMES): raise ValueError("Baseline provenance mismatch")
 a.out.mkdir(parents=True); plan=dict(role="Fixed-region rescue on already-exposed empty-label image groups",policy="Infer crop only when full-frame prediction list is empty",roi_fractions=FRACTIONS,models=[listing[n] for n in NAMES],groups=GROUPS,image_count=len(selected),comparison_sha256=sha(a.comparison/"comparison_complete.json"),legacy_summary_sha256=sha(data/"summary.json"),inputs_sha256=sha(data/"local_inputs.json"),prediction_sha256={n:sha(data/(n+"_local_predictions.json")) for n in NAMES},script_sha256=sha(__file__),helper_sha256=sha(Path(__file__).with_name("eval_fixed_roi_rescue_20261029.py")),reserved_sources_used=False,independent_test=False,on_board=False); write(a.out/"plan.json",plan)
 torch.set_num_threads(2); nets={n:YOLO(listing[n]["weights"]) for n in NAMES}
 if any(net.names!={0:"fire"} for net in nets.values()): raise ValueError("Unexpected class")
 rows=[]
 for i,item in enumerate(selected):
  image=Path(item["image"]); label=image.parents[2]/"labels/test"/(image.stem+".txt")
  if sha(image)!=item["sha256"] or sha(label)!=item["label_sha256"]: raise ValueError("Image or label changed during evaluation")
  im=cv2.imread(str(image))
  if im is None: raise ValueError("Image unreadable")
  h,w=im.shape[:2]
  if [w,h]!=item["size"]: raise ValueError("Image dimensions differ")
  region=roi_xyxy(w,h); x1,y1,x2,y2=region; pred={}
  for n in NAMES:
   base=predictions[n][item["file"]]["predictions"]; extra=[]
   if not base: extra=map_boxes(nets[n].predict(im[y1:y2,x1:x2],imgsz=640,conf=.25,iou=.6,device=0,verbose=False)[0].boxes,region)
   pred[n]=dict(full_frame=base,crop_attempted=not bool(base),crop_predictions=extra)
  rows.append(dict(file=item["file"],provenance=item["provenance"],image=item["image"],image_sha256=item["sha256"],label_sha256=item["label_sha256"],width=w,height=h,roi_xyxy=region,predictions=pred))
  if (i+1)%50==0: print(f"Processed {i+1}/{len(selected)} images",flush=True)
 if any(sha(m["weights"])!=m["expected_sha256"] for m in listing.values()) or any(sha(data/(n+"_local_predictions.json"))!=plan["prediction_sha256"][n] for n in NAMES): raise ValueError("Input changed")
 with (a.out/"per_image.jsonl").open("w",encoding="utf8") as f:
  for row in rows:f.write(json.dumps(row,ensure_ascii=False)+"\n")
 counts={}
 for n in NAMES:
  counts[n]={}
  for group in GROUPS:
   group_rows=[r for r in rows if r["provenance"]==group]; counts[n][group]=dict(images=len(group_rows),full_any=sum(bool(r["predictions"][n]["full_frame"]) for r in group_rows),crop_new_images=sum(bool(r["predictions"][n]["crop_predictions"]) for r in group_rows),crop_new_boxes=sum(len(r["predictions"][n]["crop_predictions"]) for r in group_rows))
 result=dict(role=plan["role"],plan_sha256=sha(a.out/"plan.json"),per_image_sha256=sha(a.out/"per_image.jsonl"),models=counts,semantics="Empty fire labels are not complete event truth; inspect all newly appearing boxes",independent_test=False,on_board=False); write(a.out/"comparison_complete.json",result); print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
if __name__=="__main__":main()
