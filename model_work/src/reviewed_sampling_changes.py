"""Validate a bounded, approved revision of existing legacy train samples."""
from pathlib import PurePosixPath,PureWindowsPath

def validate_sampling_changes(review,lineage,parent_manifest_sha256,evidence_sha256,empty_labels):
    if review.get("role")!="development_negative_curation" or review.get("approved") is not True:
        raise ValueError("Sampling changes are not approved development curation")
    if review.get("parent_manifest_sha256")!=parent_manifest_sha256 or review.get("evidence_sha256")!=evidence_sha256:
        raise ValueError("Sampling review parent/evidence identity differs")
    by_name={r["image"]:r for r in lineage};result={}
    if not review.get("changes"):raise ValueError("No reviewed sampling changes")
    for item in review["changes"]:
        name=item["image"]
        if not name or PurePosixPath(name).name!=name or PureWindowsPath(name).name!=name or name in result:
            raise ValueError("Unsafe or duplicate reviewed name")
        if name not in by_name or by_name[name]["origin"]!="legacy_train":
            raise ValueError("Sampling changes must refer to existing legacy train inputs")
        row=by_name[name]
        if item.get("sha256")!=row["sha256"] or item.get("label_sha256")!=row["label_sha256"] or not item.get("reason"):
            raise ValueError("Reviewed image/label identity or rationale differs")
        action=item.get("decision")
        if action=="reweight_negative":
            weight=item.get("new_weight")
            if name not in empty_labels or item.get("old_weight")!=row["weight"]:
                raise ValueError("Reweight requires an actually empty label and the original weight")
            if type(weight) is not int or not 1<=weight<=50:
                raise ValueError("Reviewed weight must be a bounded integer")
        elif action!="exclude":raise ValueError("Unknown sampling decision")
        result[name]=dict(item)
    return result
