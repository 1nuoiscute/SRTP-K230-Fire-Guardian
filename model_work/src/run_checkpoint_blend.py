"""Bounded, sequential development sweep with frozen endpoints and alphas.

Never retrains, replaces an incumbent, publishes, or operates a board.
Each subprocess has a time limit and a retained log. Failed output is preserved.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml
from checkpoint_blend import build_blends
from data_integrity import sha256
from dataset_preflight import verify_reviewed_dataset

SRC = Path(__file__).resolve().parent
ROOT = SRC.parents[1]


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--adapted", type=Path, required=True)
    p.add_argument("--base-sha256", required=True)
    p.add_argument("--adapted-sha256", required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out = a.out.resolve()
    a.data = a.data.resolve()
    if a.out.exists():
        raise SystemExit("Refusing overwrite: inspect existing run")
    preflight = verify_reviewed_dataset(a.data, yaml.safe_load((a.data / "data.yaml").read_text(encoding="utf-8")))
    for weights, digest in ((a.base, a.base_sha256), (a.adapted, a.adapted_sha256)):
        if sha256(weights) != digest.lower():
            raise SystemExit("Endpoint checkpoint identity mismatch")
    a.out.mkdir(parents=True)
    script_names = ["checkpoint_blend.py", "run_checkpoint_blend.py", "eval_hardcase_frames.py",
                    "eval_blue_localization.py", "eval_fire_fulltest.py", "eval_legacy_source_localization.py",
                    "data_integrity.py", "dataset_preflight.py"]
    plan = {"role": "all inputs development-exposed; not independent acceptance",
            "alphas": [0., .25, .5, .75, 1.], "dataset_preflight": preflight,
            "manifest_sha256": sha256(a.data / "build_manifest.json"),
            "scripts": {n: sha256(SRC / n) for n in script_names},
            "screening_only": {"maximum_map50_drop": .01, "minimum_blue_gain": 1,
                               "maximum_additional_noflame_fp": 1},
            "limitations": ["BN floating buffers interpolated without recalibration",
                            "No model adoption based on these exposed diagnostics",
                            "Only one endpoint pair and three interior coefficients; no adaptive sweep"]}
    dump(a.out / "launch.json", plan)
    try:
        blends = build_blends(a.base, a.adapted, a.base_sha256, a.adapted_sha256, [.25, .5, .75], a.out / "weights")
        models = [{"label": "v3", "alpha": 0., "weights": str(a.base.resolve()), "expected_sha256": a.base_sha256},
                  *blends,
                  {"label": "v9_last", "alpha": 1., "weights": str(a.adapted.resolve()), "expected_sha256": a.adapted_sha256}]
        dump(a.out / "models.json", models)
        summary = {"plan": plan, "models": {}}

        def run(script, extra, log):
            command = [sys.executable, "-B", str(SRC / script), *extra]
            print("Evaluating", script, *extra, flush=True)
            with log.open("w", encoding="utf-8") as stream:
                subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                               check=True, timeout=1800)

        for m in models:
            label = m["label"]
            summary["models"][label] = {k: m[k] for k in ("alpha", "expected_sha256")}
            for kind, script, extra, filename in [
                ("frames", "eval_hardcase_frames.py", ["--data", str(a.data)], "summary.json"),
                ("blue_primary", "eval_blue_localization.py", [], "summary.json"),
                ("legacy286", "eval_fire_fulltest.py", [], "metrics_summary.json")]:
                if sha256(Path(m["weights"])) != m["expected_sha256"]:
                    raise ValueError("Weights changed during evaluation")
                target = a.out / (label + "_" + kind)
                run(script, ["--weights", m["weights"], "--out", str(target), *extra],
                    a.out / (label + "_" + kind + ".log"))
                summary["models"][label][kind] = json.loads((target / filename).read_text(encoding="utf-8"))
                dump(a.out / "comparison_partial.json", summary)
        target = a.out / "legacy_sources"
        run("eval_legacy_source_localization.py", ["--data", str(Path(preflight["base"]) if "base" in preflight else
                                                             Path(json.loads((a.data / "build_manifest.json").read_text(encoding="utf-8"))["base"])),
                                                 "--models", str(a.out / "models.json"), "--out", str(target)],
            a.out / "legacy_sources.log")
        summary["legacy_sources"] = json.loads((target / "summary.json").read_text(encoding="utf-8"))
        # Identity check after all diagnostics. Never silently accept a partial sweep.
        verify_reviewed_dataset(a.data, yaml.safe_load((a.data / "data.yaml").read_text(encoding="utf-8")))
        for name, digest in plan["scripts"].items():
            if sha256(SRC / name) != digest:
                raise ValueError("Evaluation code changed during run")
        for m in models:
            if sha256(Path(m["weights"])) != m["expected_sha256"]:
                raise ValueError("Weights changed during run")
        dump(a.out / "comparison_complete.json", summary)
        print(json.dumps({label: {"alpha": r["alpha"], "blue": r["blue_primary"]["primary_localized_iou50"],
                                  "map50": r["legacy286"]["map50"], "frames": r["frames"]["by_origin"]}
                          for label, r in summary["models"].items()}, indent=2), flush=True)
    except Exception as error:
        dump(a.out / "failed.json", {"error_type": type(error).__name__, "message": str(error),
                                    "action": "Preserve partial outputs and logs; no automatic restart"})
        raise


if __name__ == "__main__":
    main()
