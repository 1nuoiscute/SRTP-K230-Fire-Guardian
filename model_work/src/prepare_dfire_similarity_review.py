"""Render fixed sampled cross-split families, known-overlap leads and label strata.

Samples are visual audit leads only, never automatic duplicate/label truth.
"""
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageDraw
from data_integrity import sha256
from hamming_index import HammingIndex


def load(p): return json.loads(p.read_text(encoding="utf-8"))


def panel(path, boxes=None):
    im = Image.open(path).convert("RGB")
    if boxes:
        w,h = im.size; draw=ImageDraw.Draw(im)
        for cls,x,y,bw,bh in boxes:
            box=[int((x-bw/2)*w),int((y-bh/2)*h),int((x+bw/2)*w),int((y+bh/2)*h)]
            color="#ff55ff" if cls==0 else "#ff9933"
            draw.rectangle(box,outline=color,width=max(2,w//250))
            draw.text(box[:2],"Smoke" if cls==0 else "Fire",fill=color)
    im.thumbnail((620,430)); canvas=Image.new("RGB",(620,430),"#151515")
    canvas.paste(im,((620-im.width)//2,(430-im.height)//2)); return canvas


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--inventory-dir",type=Path,required=True)
    p.add_argument("--perceptual-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise SystemExit("Refusing overwrite")
    invpath=a.inventory_dir/"inventory.json"; ps=load(a.perceptual_dir/"summary.json")
    if sha256(invpath)!=ps["inventory_sha256"]:raise SystemExit("Inventory changed")
    records=load(invpath); components=load(a.perceptual_dir/"components.json")
    known=load(a.perceptual_dir/"known_images.json"); leads=load(a.perceptual_dir/"known_similarity_pairs.json")
    for name in ("components.json","known_images.json","known_similarity_pairs.json"):
        if sha256(a.perceptual_dir/name)!=ps[name+"_sha256"]:raise SystemExit("Perceptual results changed")
    rng=random.Random(20261002); reviews=[]
    cross=[c for c in components if len(c["split_counts"])>1]
    selected=rng.sample(cross,min(12,len(cross)))
    for family in selected:
        trees=defaultdict(HammingIndex); best=None
        for i in family["members"]:
            row=records[i]; value=int(row["phash64"],16)
            for split,tree in trees.items():
                if split==row["split"]:continue
                for distance,j in tree.within(value,4):
                    key=(distance,hashlib.sha256((row["sha256"]+records[j]["sha256"]).encode()).hexdigest())
                    if best is None or key<best[0]:best=(key,j,i)
            trees[row["split"]].add(value,i)
        if best is None:raise SystemExit("Cross-split component lacks an actual cross-split edge")
        key,left,right=best
        reviews.append({"kind":"source_cross_split","left":records[left],"right":records[right],
                        "distance":key[0],"component_images":len(family["members"]),"manual_review":None})
    nearest={}
    for pair in leads:
        i=pair["candidate"]
        key=(pair["distance"],known[pair["known"]]["sha256"])
        if i not in nearest or key<nearest[i][0]:nearest[i]=(key,pair["known"])
    chosen=rng.sample(sorted(nearest),min(8,len(nearest)))
    for i in chosen:
        key,j=nearest[i]
        reviews.append({"kind":"known_exposure_lead","left":records[i],"right":known[j],"distance":key[0],"manual_review":None})
    for classes in ([],[0],[1],[0,1]):
        pool=sorted([r for r in records if r["split"]=="train" and not r["errors"] and r["classes"]==classes],key=lambda r:r["sha256"])
        for row in rng.sample(pool,4):
            reviews.append({"kind":"label_stratum","left":row,"right":None,"manual_review":None})
    a.out.mkdir(parents=True); sheets=[]
    for page in range((len(reviews)+3)//4):
        sheet=Image.new("RGB",(1280,1920),"#252525")
        for j,review in enumerate(reviews[page*4:page*4+4]):
            left,right=review["left"],review["right"]
            for r in (left,right):
                if r is not None and sha256(Path(r["image"]))!=r["sha256"]:raise SystemExit("Review image changed")
            idx=page*4+j+1;review["index"]=idx
            tile=Image.new("RGB",(1280,480),"#252525");draw=ImageDraw.Draw(tile)
            draw.text((8,8),f"{idx:02d} {review['kind']} left={left.get('split','known')} classes={left.get('classes','unknown')} distance={review.get('distance','n/a')}",fill="white")
            tile.paste(panel(left["image"],left.get("boxes")),(10,40))
            if right:
                tile.paste(panel(right["image"],right.get("boxes")),(650,40))
                draw.text((650,8),f"right={right.get('split',right.get('scope','known'))}",fill="white")
            sheet.paste(tile,(0,j*480))
        target=a.out/f"sheet_{page+1:02d}.jpg";sheet.save(target,quality=95)
        sheets.append({"file":target.name,"sha256":sha256(target)})
    result={"role":"pending actual visual duplicate/exposure and source-label audit","seed":20261002,
            "inventory_sha256":sha256(invpath),"perceptual_summary_sha256":sha256(a.perceptual_dir/"summary.json"),
            "script_sha256":sha256(Path(__file__)),"selection":{"cross_split_components_random_sample":12,"known_candidate_images_random_sample":8,"train_label_strata":4,"each_stratum":4,
            "scope":"Different audit sampling fractions; not duplicate prevalence/error-rate estimate. Source first-10000 pair file not used (biased toward train)."},
            "sheets":sheets,"reviews":reviews}
    (a.out/"review_pending.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"reviews":len(reviews),"sheets":len(sheets)},indent=2))


if __name__=="__main__":main()
