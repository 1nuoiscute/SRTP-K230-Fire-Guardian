"""Recalculate v24 preclip anchor gradient ratios from the preserved audit.

The historical trainer's ratio uses the initial distance. This reader never
changes training inputs, logs, checkpoints, or their recorded identities.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def summarize(audit):
    lam = audit["anchor_lambda"]
    if not math.isfinite(lam) or lam <= 0:
        raise ValueError("Invalid anchor lambda")
    rows = [r for r in audit["records"] if r["actual_optimizer_step"]]
    if len(rows) != audit["actual_optimizer_steps"] or not rows:
        raise ValueError("Optimizer-step inventory differs")
    ratios, historical = [], []
    for row in rows:
        det = row["detection_grad_norm"]
        distance = row["current_anchor_distance_norm"]
        if not math.isfinite(det) or det <= 0 or not math.isfinite(distance) or distance < 0:
            raise ValueError("Invalid gradient or current distance")
        ratios.append(lam * distance / det)
        historical.append(row["anchor_to_detection_grad_ratio"])
    ratios.sort()
    return {
        "effective_steps": len(rows),
        "actual_preclip_ratio_median": statistics.median(ratios),
        "actual_preclip_ratio_p95_nearest_lower": ratios[int(.95 * (len(ratios)-1))],
        "actual_preclip_ratio_max": max(ratios),
        "actual_preclip_ratio_below_target": sum(x < audit["target_anchor_gradient_ratio"] for x in ratios),
        "historical_initial_distance_ratio_median": statistics.median(historical),
        "steps_with_combined_preclip_norm_above_10": sum(r["combined_grad_norm_preclip"] > 10 for r in rows),
        "limits": "Norm ratios before global clipping; optimizer updates and postclip component vectors are not reconstructed.",
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--meta", type=Path, required=True)
    args=p.parse_args()
    raw=args.audit.read_bytes()
    meta=json.loads(args.meta.read_text(encoding="utf-8"))
    digest=hashlib.sha256(raw).hexdigest()
    if digest != meta["completed_audits"][args.audit.name]:
        raise ValueError("Preserved training audit hash differs from metadata")
    audit=json.loads(raw)
    result=summarize(audit)
    if result["effective_steps"] != meta["actual_optimizer_steps"]:
        raise ValueError("Metadata step count differs")
    result["audit_sha256"]=digest
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
