"""Publish aggregate source-cap diagnostics without media names or private paths."""
import argparse
import json
from pathlib import Path
from data_integrity import sha256

BASE_SHA = "48f284f9094fe334e02c66647f95fed8bfb3396b09f4488753508f514ed38980"


def read(path): return json.loads(path.read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--comparison", type=Path, required=True)
    p.add_argument("--expected-data-sha256", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists(): raise SystemExit("Refusing overwrite")
    manifest = a.data / "build_manifest.json"
    if sha256(manifest) != a.expected_data_sha256: raise SystemExit("Dataset identity differs")
    data, run = read(manifest), read(a.run / "experiment_meta.json")
    comparison = read(a.comparison / "comparison_complete.json")
    if comparison["data_manifest_sha256"] != a.expected_data_sha256 or run["dataset_manifest_sha256"] != a.expected_data_sha256:
        raise SystemExit("Training and comparison dataset identities differ")
    models = comparison["models"]
    if set(models) != {"v3", "v12_best", "v12_last"} or models["v3"]["weights_sha256"] != BASE_SHA or run["start_sha256"] != BASE_SHA:
        raise SystemExit("Unexpected baseline or candidates")
    for name in ("best", "last"):
        if sha256(a.run / "weights" / (name + ".pt")) != run[name + ".pt_sha256"] or models["v12_" + name]["weights_sha256"] != run[name + ".pt_sha256"]:
            raise SystemExit("Candidate checkpoint identity differs")
    sources = comparison["legacy_source_counts"]
    if sources["images"] != 286 or set(sources["models"]) != set(models):
        raise SystemExit("Missing fixed-threshold source comparison")
    public_models = {}
    for label, r in models.items():
        if sources["models"][label]["weights_sha256"] != r["weights_sha256"]:
            raise SystemExit("Source-count checkpoint identity differs")
        by_source = sources["models"][label]["by_provenance"]
        if by_source["nofire_real_indoor"]["images"] != 100:
            raise SystemExit("Unexpected original no-fire diagnostic count")
        scores = {k: r["legacy286"][k] for k in ("precision", "recall", "map50", "map50_95")}
        video_counts = {}
        for kind in ("old_videos", "teammate_videos"):
            video_counts[kind] = [{"clip_index": i, "sampled": v["sampled"], "frames_with_any_box": v["frames_with_any_box"]}
                                 for i, v in enumerate(r[kind]["clips"])]
        public_models[label] = {"weights_sha256": r["weights_sha256"], "blue_primary_localized_iou50": r["blue_primary"]["primary_localized_iou50"],
             "legacy286": scores, "fixed_threshold_by_provenance": by_source, "training_exposed_frames": r["frames"]["by_origin"],
             "commons_source_strata": r["commons13"]["source_exposure"], "video_prediction_counts": video_counts}
    base = public_models["v3"]
    for label, r in public_models.items():
        if label == "v3": continue
        drop = base["legacy286"]["map50"] - r["legacy286"]["map50"]
        gain = r["blue_primary_localized_iou50"] - base["blue_primary_localized_iou50"]
        fp_gain = r["fixed_threshold_by_provenance"]["nofire_real_indoor"]["fp"] - base["fixed_threshold_by_provenance"]["nofire_real_indoor"]["fp"]
        checks = {"map50_drop_at_most_0_01": drop <= .01, "blue_gain_at_least_1": gain >= 1,
                  "nofire_fp_increase_at_most_1": fp_gain <= 1}
        r["development_gate"] = {"map50_drop": drop, "blue_gain": gain, "nofire_fp_increase": fp_gain,
                                 "checks": checks, "pass": all(checks.values()), "scope": "Screening only; no automatic promotion or independent acceptance"}
    plan = data["source_sampling"]
    result = {"role": "exposed development source-sampling ablation; no board deployment or independent acceptance",
              "incumbent": "v3", "data_manifest_sha256": a.expected_data_sha256,
              "comparison_complete_sha256": sha256(a.comparison / "comparison_complete.json"),
              "experiment_meta_sha256": sha256(a.run / "experiment_meta.json"), "summary_script_sha256": sha256(Path(__file__)),
              "source_sampling": {"seed": plan["seed"], "caps": plan["caps"],
                                  "audit": {k: v for k, v in plan["audit"].items() if k != "selection"}},
              "training": {k: run["config"][k] for k in ("epochs", "patience", "imgsz", "batch", "seed", "optimizer", "lr0", "warmup_bias_lr")},
              "models": public_models,
              "v11_same_derived_frames": {k: {"weights_sha256": v["weights_sha256"], "by_origin": v["by_origin"]} for k,v in comparison["frame_references"].items()},
              "limitations": ["Same epoch schedule but fewer optimizer steps after source capping.",
                  "Source caps are not scene-group splitting and do not remove inherited v3 exposure.",
                  "Legacy mAP/P/R and approximate blue boxes are development diagnostics, not independent F1.",
                  "Video any-box counts lack per-frame event truth and are not accuracy or false alarms per hour."]}
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v.get("development_gate") for k,v in public_models.items()}, indent=2))


if __name__ == "__main__": main()
