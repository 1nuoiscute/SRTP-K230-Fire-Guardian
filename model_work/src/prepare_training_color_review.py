"""Create deterministic stratified visual review sheets, without assigning color truth.
Only original train images are read. Counts from this sample are not population estimates.
"""
import argparse,csv,hashlib,json,random
from pathlib import Path
from PIL import Image,ImageDraw

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def fit(im,size):
    im=im.copy();im.thumbnail(size)
    canvas=Image.new("RGB",size,"#151515")
    canvas.paste(im,((size[0]-im.width)//2,(size[1]-im.height)//2))
    return canvas

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--seed",type=int,default=20260930)
    a=ap.parse_args()
    if a.out.exists(): raise SystemExit("Refusing overwrite")
    rows=list(csv.DictReader((a.data/"manifest.csv").open(encoding="utf-8-sig")))
    rng=random.Random(a.seed);selected=[]
    for provenance,count in [("kitchen_stove_fire",24),("ks_flame",12),("generic_fire",12)]:
        pool=sorted((r for r in rows if r["split"]=="train" and r["provenance"]==provenance),key=lambda r:r["file"])
        selected.extend(rng.sample(pool,count))
    a.out.mkdir(parents=True);entries=[];sheets=[]
    for page in range(4):
        sheet=Image.new("RGB",(2048,1152),"#303030")
        for j,row in enumerate(selected[page*12:(page+1)*12]):
            idx=page*12+j+1;impath=a.data/"images/train"/row["file"]
            lp=a.data/"labels/train"/(impath.stem+".txt")
            if sha(impath)!=row["sha256"]: raise SystemExit("Original image identity mismatch")
            im=Image.open(impath).convert("RGB");w,h=im.size;boxes=[]
            for line in lp.read_text(encoding="utf-8").splitlines():
                if not line.strip(): continue
                cls,x,y,bw,bh=map(float,line.split())
                if cls!=0: raise SystemExit("Unexpected class")
                boxes.append([max(0,int((x-bw/2)*w)),max(0,int((y-bh/2)*h)),min(w,int((x+bw/2)*w)),min(h,int((y+bh/2)*h))])
            annotated=im.copy();draw=ImageDraw.Draw(annotated)
            for b in boxes: draw.rectangle(b,outline="#ff00ff",width=max(1,w//350))
            union=[min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)] if boxes else [0,0,w,h]
            tile=Image.new("RGB",(512,384),"#202020");td=ImageDraw.Draw(tile)
            td.text((8,8),f"{idx:02d} {row['provenance']} {w}x{h} boxes={len(boxes)}",fill="white")
            tile.paste(fit(annotated,(252,310)),(2,40));tile.paste(fit(im.crop(union),(252,310)),(258,40))
            td.text((8,360),"Full frame / label union (full if empty)",fill="white")
            sheet.paste(tile,((j%4)*512,(j//4)*384))
            entries.append({"index":idx,**row,"image_path":str(impath.resolve()),"label_path":str(lp.resolve()),"label_sha256":sha(lp),"size":[w,h],"boxes_xyxy":boxes,"manual_review":None})
        out=a.out/f"sheet_{page+1:02d}.jpg";sheet.save(out,quality=95)
        sheets.append({"file":out.name,"sha256":sha(out)})
    report={"role":"stratified diagnostic sample, not exhaustive color census or independent evaluation","seed":a.seed,"data_manifest_sha256":sha(a.data/"manifest.csv"),"script_sha256":sha(Path(__file__)),"selection":"random sample within sorted provenance, 24 stove / 12 KS / 12 generic; unequal sampling fractions","sheets":sheets,"images":entries}
    (a.out/"review_pending.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"sample":len(entries),"sheets":sheets},indent=2))
if __name__=="__main__": main()
