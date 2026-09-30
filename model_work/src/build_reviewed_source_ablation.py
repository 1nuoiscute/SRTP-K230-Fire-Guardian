"""Derive a source-removal experiment from a fully identity-checked parent.
Never edits original inputs/labels or starts training; weights change only with explicit reviewed negative curation.
"""
import argparse,copy,csv,json,shutil
from collections import Counter
from pathlib import Path
import yaml
from data_integrity import sha256
from dataset_preflight import verify_reviewed_dataset
from reviewed_sampling_changes import validate_sampling_changes

def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--parent",type=Path,required=True)
    mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--exclude-provenance")
    mode.add_argument("--changes-review",type=Path)
    ap.add_argument("--evidence",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args();parent=a.parent.resolve();out=a.out.resolve()
    if out.exists(): raise SystemExit("Refusing overwrite")
    verified=verify_reviewed_dataset(parent,yaml.safe_load((parent/"data.yaml").read_text(encoding="utf-8")))
    meta=json.loads((parent/"build_manifest.json").read_text(encoding="utf-8"))
    base=Path(meta["base"]);originals={r["file"]:r for r in csv.DictReader((base/"manifest.csv").open(encoding="utf-8-sig")) if r["split"]=="train"}
    evidence=json.loads(a.evidence.read_text(encoding="utf-8"))
    if evidence["data_manifest_sha256"]!=sha256(base/"manifest.csv"):
        raise SystemExit("Audit evidence is for another base manifest")
    changes={};reweighted=[]
    if a.changes_review:
        review=json.loads(a.changes_review.read_text(encoding="utf-8"))
        if review["base_manifest_sha256"]!=sha256(base/"manifest.csv"):
            raise SystemExit("Sampling review is for another base")
        for path_key,hash_key in [("sample_manifest","sample_manifest_sha256"),("sample_sheet","sample_sheet_sha256")]:
            if sha256(Path(review[path_key]))!=review[hash_key]:raise SystemExit("Negative sample review changed")
        empty={r["image"] for r in meta["lineage"] if r["origin"]=="legacy_train" and not (base/"labels/train"/(Path(r["image"]).stem+".txt")).read_text(encoding="utf-8").strip()}
        changes=validate_sampling_changes(review,meta["lineage"],sha256(parent/"build_manifest.json"),sha256(a.evidence),empty)
        sampled={r["file"]:r for r in json.loads(Path(review["sample_manifest"]).read_text(encoding="utf-8"))["images"]}
        for name,change in changes.items():
            if change["decision"]=="reweight_negative":
                checked=sampled.get(name,{})
                if checked.get("sha256")!=change["sha256"] or checked.get("label_sha256")!=change["label_sha256"]:
                    raise SystemExit("Reweighted negative was not in the identity-bound visual sample")
    kept=[];removed=[]
    for row in meta["lineage"]:
        if row["origin"]=="legacy_train":
            raw=originals[row["image"]]
            if raw["sha256"]!=row["sha256"]: raise SystemExit("Base/source identity mismatch")
            change=changes.get(row["image"],{})
            if (a.exclude_provenance and raw["provenance"]==a.exclude_provenance) or change.get("decision")=="exclude":
                removed.append({**row,"provenance":raw["provenance"],"source":raw["source"]});continue
        retained=copy.deepcopy(row)
        if changes.get(row["image"],{}).get("decision")=="reweight_negative":
            change=changes[row["image"]];retained["weight"]=change["new_weight"];reweighted.append(change)
        kept.append(retained)
    if not (removed or reweighted) or not kept: raise SystemExit("Ablation must remove an existing source and preserve training inputs")
    out.mkdir(parents=True)
    for folder in ("images/train","labels/train"): (out/folder).mkdir(parents=True)
    for row in kept:
        if row["origin"]=="legacy_train": continue
        srcimage=parent/"images/train"/row["image"];srclabel=parent/"labels/train"/(srcimage.stem+".txt")
        dstimage=out/"images/train"/srcimage.name;dstlabel=out/"labels/train"/srclabel.name
        shutil.copy2(srcimage,dstimage);shutil.copy2(srclabel,dstlabel)
        if sha256(dstimage)!=row["sha256"] or sha256(dstlabel)!=row["label_sha256"]:
            raise SystemExit("Retained derived image/label identity mismatch")
    for name in ("label_review_sheet.jpg","new_source_review_sheet.jpg"):
        if (parent/name).exists(): shutil.copy2(parent/name,out/name)
    for name in ("source_label_review","source_reviews"):
        if (parent/name).exists(): shutil.copytree(parent/name,out/name)
    decision={"type":"reviewed negative curation" if a.changes_review else "whole-source removal, not automatic relabeling or a count of bad images","excluded_provenance":a.exclude_provenance,"removed_unique_images":len(removed),"removed_entries":sum(r["weight"] for r in removed),"removed_images":removed,"reweighted_negative_items":reweighted,"parent_manifest_sha256":sha256(parent/"build_manifest.json"),"parent_approval_sha256":sha256(parent/"review_approval.json"),"parent_preflight":verified,"evidence_sha256":sha256(a.evidence),"scope":"Retained image/label bytes and source review artifacts are inherited unchanged; only explicitly reviewed negative weights may change; no new blind samples."}
    evidence_copy=out/"source_reviews"/"training_quality_evidence.json";shutil.copy2(a.evidence,evidence_copy)
    dump(out/"source_reviews"/"source_ablation_decision.json",decision)
    if a.changes_review:
        shutil.copy2(a.changes_review,out/"source_reviews"/"sampling_changes_review.json")
        shutil.copy2(Path(review["sample_manifest"]),out/"source_reviews"/"negative_sample_review.json")
        shutil.copy2(Path(review["sample_sheet"]),out/"negative_sample_review_sheet.jpg")
    paths=[]
    for row in kept:
        root=base if row["origin"]=="legacy_train" else out
        paths.extend([str(root/"images/train"/row["image"])]*row["weight"])
    (out/"train.txt").write_text("\n".join(paths)+"\n",encoding="utf-8")
    legacy=Counter(originals[r["image"]]["provenance"] for r in kept if r["origin"]=="legacy_train")
    meta.update({"lineage":kept,"legacy_train":dict(legacy),"legacy_unique_images":sum(legacy.values()),"train_entries":len(paths),"unique_paths":len(kept),"derived_images":sum(r["origin"]!="legacy_train" for r in kept),"parent_manifest_sha256":sha256(parent/"build_manifest.json"),"parent_review_sha256":sha256(parent/"review_approval.json"),"review_required":"Approve exact removal, explicit negative weight changes and inherited identities; no new annotations.","source_ablation":{"excluded_provenance":a.exclude_provenance,"reweighted_negative_images":len(reweighted),"additional_sampling_entries":sum(r["new_weight"]-r["old_weight"] for r in reweighted),"removed_unique_images":len(removed),"removed_entries":sum(r["weight"] for r in removed),"decision_sha256":sha256(out/"source_reviews"/"source_ablation_decision.json")}})
    meta["limitations"].append("Negative curation combines quarantine and weight changes; no isolated causal claim for each operation." if a.changes_review else "Whole-source ablation changes per-epoch sample count and gradient update count; it is not proof all excluded labels are wrong.")
    dump(out/"build_manifest.json",meta)
    config={"path":str(out),"train":str(out/"train.txt"),"val":str(base/"images/val"),"test":str(base/"images/test"),"nc":1,"names":{0:"fire"}}
    (out/"data.yaml").write_text(yaml.safe_dump(config,sort_keys=False),encoding="utf-8")
    approval=json.loads((parent/"review_approval.json").read_text(encoding="utf-8"))
    approval.update({"approved":False,"manifest_sha256":sha256(out/"build_manifest.json"),"sheet_sha256":sha256(out/"label_review_sheet.jpg"),"reason":"Pending source-removal review; retained annotations were reviewed in the parent.","source_review_files_sha256":{p.name:sha256(p) for p in (out/"source_reviews").iterdir() if p.is_file()}})
    if a.changes_review:
        approval["negative_sample_sheet_sha256"]=sha256(out/"negative_sample_review_sheet.jpg")
    approval.pop("reviewed_utc",None)
    dump(out/"review_approval_pending.json",approval);dump(out/"review_approval.json",approval)
    print(json.dumps({"retained_unique":len(kept),"train_entries":len(paths),"removed":len(removed),"reweighted":len(reweighted),"approved":False},indent=2))
if __name__=="__main__": main()
