"""Publish aggregate PC experiment facts without private filenames or media hashes."""
import argparse
import json
from pathlib import Path

from data_integrity import sha256


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("blend-run", "static-run", "dynamic-run", "static-policy", "dynamic-policy", "static-video", "dynamic-video"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit("Refusing overwrite")
    blend = load(a.blend_run / "comparison_complete.json")
    limits = blend["plan"]["screening_only"]
    base = blend["models"]["v3"]
    base_fp = blend["legacy_sources"]["models"]["v3"]["by_provenance"]["nofire_real_indoor"]["fp"]
    models = {}
    for label, row in blend["models"].items():
        sources = blend["legacy_sources"]["models"][label]["by_provenance"]
        models[label] = {"alpha": row["alpha"], "weights_sha256": row["expected_sha256"],
                         "blue_primary_localized": row["blue_primary"]["primary_localized_iou50"],
                         "legacy_map50": row["legacy286"]["map50"], "legacy_map50_95": row["legacy286"]["map50_95"],
                         "derived_frames_by_origin": row["frames"]["by_origin"], "legacy_by_provenance": sources,
                         "passes_development_screen": base["legacy286"]["map50"] - row["legacy286"]["map50"] <= limits["maximum_map50_drop"]
                         and row["blue_primary"]["primary_localized_iou50"] - base["blue_primary"]["primary_localized_iou50"] >= limits["minimum_blue_gain"]
                         and sources["nofire_real_indoor"]["fp"] - base_fp <= limits["maximum_additional_noflame_fp"]}
    result = {"date": "2026-10-02", "role": "development diagnostics and CPU export compatibility; no independent or board acceptance",
              "decision": "Retain original v3; no interpolated checkpoint passes the frozen development screen.",
              "blend": {"launch_sha256": sha256(a.blend_run / "launch.json"), "manifest_sha256": blend["plan"]["manifest_sha256"],
                        "screening_only": limits, "models": models}, "onnx": {}, "videos": {}}
    for name, run, policy, video in [("static_square", a.static_run, a.static_policy, a.static_video),
                                     ("dynamic_rectangular", a.dynamic_run, a.dynamic_policy, a.dynamic_video)]:
        parity, wrapper = load(run / "summary.json"), load(policy / "summary.json")
        if parity["all_passed"] is not True or wrapper["images"] != 80:
            raise ValueError("Incomplete export verification")
        result["onnx"][name] = {"parity": parity, "wrapper": wrapper}
        rows = load(video / "local_summary.json")
        previews = load(video / "preview_validation.json")
        if len(rows) != 10 or len(previews) != 10 or any(r["preview_frames"] != rows[i]["sampled_frames"] for i, r in enumerate(previews)):
            raise ValueError("Incomplete video/preview verification")
        result["videos"][name] = {"clips": len(rows), "decoded_frames": sum(r["decoded_frames"] for r in rows),
                                 "sampled_frames": sum(r["sampled_frames"] for r in rows), "previews_validated": len(previews),
                                 "per_clip": [{k: r[k] for k in ("clip_index", "decoded_frames", "sampled_frames", "frames_with_any_box", "total_boxes", "nominal_minus_decoded_frames")} for r in rows],
                                 "role": "previously exposed predictions; any-box counts are not accuracy; nominal frame timestamps, not VFR validation"}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Public aggregate saved:", a.out)


if __name__ == "__main__":
    main()
