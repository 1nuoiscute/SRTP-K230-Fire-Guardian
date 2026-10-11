"""Identity-bound interpolation of two related, unfused YOLO checkpoints.

This is a development experiment, not a new training run or an ensemble.
Floating parameters and buffers (including BN statistics) are interpolated in
FP32. Integer buffers come from the base checkpoint. No BN recalibration occurs.
"""
import argparse
import copy
import json
import math
from pathlib import Path

from data_integrity import sha256


def blend_states(base, adapted, alpha):
    import torch
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and in [0, 1]")
    if set(base) != set(adapted):
        raise ValueError("State keys differ")
    result = {}
    counts = {"floating_tensors": 0, "base_integer_buffers": 0}
    for key, left in base.items():
        right = adapted[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"State shape/dtype differs: {key}")
        if left.is_complex() or left.is_quantized:
            raise ValueError(f"Unsupported state tensor: {key}")
        if left.is_floating_point():
            l, r = left.detach().cpu().float(), right.detach().cpu().float()
            if not torch.isfinite(l).all() or not torch.isfinite(r).all():
                raise ValueError(f"Nonfinite input state: {key}")
            result[key] = torch.lerp(l, r, alpha)
            if not torch.isfinite(result[key]).all():
                raise ValueError(f"Nonfinite blended state: {key}")
            counts["floating_tensors"] += 1
        else:
            result[key] = left.detach().cpu().clone()
            counts["base_integer_buffers"] += 1
    return result, counts


def architecture(model):
    return {"modules": [(name, type(module).__module__ + "." + type(module).__qualname__)
                        for name, module in model.named_modules()],
            "yaml": {k: model.yaml.get(k) for k in ("nc", "backbone", "head", "scale", "scales")},
            "names": model.names, "stride": model.stride.tolist()}


def build_blends(base_path, adapted_path, base_digest, adapted_digest, alphas, out):
    import torch
    import ultralytics
    if out.exists():
        raise ValueError(f"Refusing overwrite: {out}")
    if not alphas or len(set(alphas)) != len(alphas) or any(not math.isfinite(a) or not 0 < a < 1 for a in alphas):
        raise ValueError("Specify unique interior alphas")
    for path, digest in ((base_path, base_digest), (adapted_path, adapted_digest)):
        if sha256(path) != digest.lower():
            raise ValueError("Checkpoint identity mismatch")
    # Only explicitly supplied trusted project checkpoints may be unpickled.
    checkpoints = [torch.load(p, map_location="cpu", weights_only=False) for p in (base_path, adapted_path)]
    models = [c.get("ema") if c.get("ema") is not None else c["model"] for c in checkpoints]
    if architecture(models[0]) != architecture(models[1]):
        raise ValueError("Checkpoint architecture or class semantics differ")
    identities = {"base_sha256": base_digest.lower(), "adapted_sha256": adapted_digest.lower(),
                  "script_sha256": sha256(Path(__file__)), "torch": torch.__version__,
                  "ultralytics": ultralytics.__version__, "role": "exposed development interpolation; no training; no deployment",
                  "policy": "FP32 float parameters and buffers; integer buffers from base; no BN recalibration"}
    # Validate all states before any output is created.
    blend_states(models[0].state_dict(), models[1].state_dict(), alphas[0])
    out.mkdir(parents=True)
    records = []
    for index, alpha in enumerate(alphas):
        state, counts = blend_states(models[0].state_dict(), models[1].state_dict(), alpha)
        model = copy.deepcopy(models[0]).cpu().float().eval()
        model.load_state_dict(state, strict=True)
        record = {**identities, "alpha": alpha, "counts": counts}
        path = out / f"blend_{index:02d}.pt"
        torch.save({"model": model, "ema": None, "epoch": -1, "optimizer": None,
                    "train_args": copy.deepcopy(checkpoints[0].get("train_args", {})),
                    "blend_metadata": record, "version": ultralytics.__version__}, path)
        records.append({**record, "label": f"blend_{index:02d}", "weights": str(path.resolve()),
                        "expected_sha256": sha256(path)})
    (out / "blend_identity.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--adapted", type=Path, required=True)
    p.add_argument("--base-sha256", required=True)
    p.add_argument("--adapted-sha256", required=True)
    p.add_argument("--alpha", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(build_blends(a.base, a.adapted, a.base_sha256, a.adapted_sha256, a.alpha, a.out), indent=2))


if __name__ == "__main__":
    main()
