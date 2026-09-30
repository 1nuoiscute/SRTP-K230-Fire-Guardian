"""Paired fixed-threshold localization by provenance on unchanged legacy test images.
Detailed private frame identities/predictions remain local; public summary is aggregate.
"""
import argparse,csv,json,re
from collections import defaultdict
from pathlib import Path
import cv2
from ultralytics import YOLO
from data_integrity import matches,sha256,validate_yolo

def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",type=Path,required=True)
    ap.add_argument("--models",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--device",default="0")
    a=ap.parse_args()
    if a.out.exists():raise SystemExit("Refusing overwrite")
    rows=[r for r in csv.DictReader((a.data/"manifest.csv").open(encoding="utf-8-sig")) if r["split"]=="test"]
    models=json.loads(a.models.read_text(encoding="utf-8"))
    if not rows or not models:raise SystemExit("No test images or models")
    labels=set()
    for m in models:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+",m["label"]) or m["label"] in labels:
            raise SystemExit("Unsafe or duplicate model label")
        labels.add(m["label"])
        if sha256(Path(m["weights"]))!=m["expected_sha256"]:raise SystemExit("Model identity mismatch")
    inputs=[]
    for r in rows:
        image=a.data/"images/test"/r["file"];label=a.data/"labels/test"/(image.stem+".txt")
        if sha256(image)!=r["sha256"]:raise SystemExit("Manifest image identity mismatch")
        text=label.read_text(encoding="utf-8");validate_yolo(text)
        im=cv2.imread(str(image))
        if im is None:raise SystemExit("Unreadable image")
        h,w=im.shape[:2];gt=[]
        for line in text.splitlines():
            if not line.strip():continue
            cls,x,y,bw,bh=map(float,line.split())
            if cls!=0:raise SystemExit("Expected binary fire class")
            gt.append([(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h])
        if len(gt)!=int(r["fire_boxes"]):raise SystemExit("Manifest label count mismatch")
        inputs.append({**r,"image":str(image),"label_sha256":sha256(label),"size":[w,h],"gt":gt})
    a.out.mkdir(parents=True);dump(a.out/"local_inputs.json",inputs)
    summary={"role":"exposed legacy development diagnostic; fixed threshold counts, not mAP or independent acceptance","configuration":{"imgsz":640,"conf":0.25,"nms_iou":0.6,"match_iou":0.5},"images":len(inputs),"models":{}}
    for m in models:
        model=YOLO(m["weights"]);details=[];groups=defaultdict(list);overlays=a.out/(m["label"]+"_errors");overlays.mkdir()
        for r in inputs:
            image=Path(r["image"]);label=a.data/"labels/test"/(image.stem+".txt")
            if sha256(image)!=r["sha256"] or sha256(label)!=r["label_sha256"]:
                raise SystemExit("Test image or label changed during comparison")
            pred=model.predict(str(image),imgsz=640,conf=.25,iou=.6,device=a.device,verbose=False)[0]
            boxes=[{"confidence":round(float(b.conf.item()),4),"xyxy":[float(v) for v in b.xyxy[0].tolist()]} for b in pred.boxes]
            score=matches(r["gt"],[b["xyxy"] for b in boxes])
            record={"image":r["file"],"provenance":r["provenance"],"gt_boxes":len(r["gt"]),**score,"predictions":boxes}
            details.append(record);groups[r["provenance"]].append(record)
            if score["fp"] or score["fn"]:
                im=cv2.imread(str(image))
                for b in r["gt"]:cv2.rectangle(im,tuple(map(int,b[:2])),tuple(map(int,b[2:])),(0,255,0),2)
                for b in boxes:cv2.rectangle(im,tuple(map(int,b["xyxy"][:2])),tuple(map(int,b["xyxy"][2:])),(255,0,0),2)
                cv2.imwrite(str(overlays/r["file"]),im)
        aggregate={}
        for name,rr in groups.items():
            aggregate[name]={"images":len(rr),"gt_boxes":sum(r["gt_boxes"] for r in rr),"images_with_fp":sum(r["fp"]>0 for r in rr),"images_with_fn":sum(r["fn"]>0 for r in rr),**{k:sum(r[k] for r in rr) for k in ("tp","fp","fn")}}
        dump(a.out/(m["label"]+"_local_predictions.json"),details)
        summary["models"][m["label"]]={"weights_sha256":m["expected_sha256"],"by_provenance":aggregate}
        dump(a.out/"summary_partial.json",summary)
    dump(a.out/"summary.json",summary)
    print(json.dumps(summary,indent=2),flush=True)
if __name__=="__main__":main()
