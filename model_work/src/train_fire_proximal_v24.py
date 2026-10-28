"""V24 controlled L2-SP anchor experiment on the audited v22 route.

The source data, starting checkpoint, freeze mask, schedule, and sampling stay
identical to v22. The only intervention is a fixed L2-SP gradient toward v21
last during each real optimizer update. A four-batch calibration run must be
completed before candidate training; its single lambda is then reused exactly.
"""
import argparse
import math
import platform
import statistics
from pathlib import Path

from build_kitchen_source_extension_v20 import read, write, verify
from data_integrity import sha256
import train_fire_source_sampling_v22 as V22
from train_kitchen_partial_features_v21 import (
    FEATURES,
    PREFIX,
    PRIMARY,
    frozen_schedule,
)
from train_scratch_semantic_v16 import verify_trainer_reconstruction


ROOT = Path(__file__).resolve().parents[2]
ROLE = "V24 controlled L2-SP anchor experiment on the audited v22 route"
ANCHOR_PATH = (
    ROOT
    / "model_work/runs/fire-kitchen-partial-features-v21-20261028/weights/last.pt"
)
ANCHOR_SHA = "185a906e277b603e949a1f8b6b0d3693a32d62b87aeedb7b6de736a46248d48c"
REFERENCE_META = (
    ROOT / "model_work/runs/fire-kitchen-partial-features-v21-20261028"
    / "experiment_meta.json"
)
DEPENDENCIES = ["train_fire_source_sampling_v22.py"] + V22.DEPENDENCIES
DEFAULT_TARGET_RATIO = 0.20
DEFAULT_CALIBRATION_BATCHES = 4


class CalibrationComplete(RuntimeError):
    """Stop the calibration run after the requested optimizer attempts."""


def _unit_interval(value, name):
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be finite and inside (0, 1)")
    return float(value)


def select_lambda(detection_norms, anchor_delta_norm, target_ratio):
    """Choose one lambda from pre-evaluation gradient-scale measurements."""
    ratio = _unit_interval(target_ratio, "target_ratio")
    if not math.isfinite(anchor_delta_norm) or anchor_delta_norm <= 0.0:
        raise ValueError("anchor_delta_norm must be finite and positive")
    if not detection_norms:
        raise ValueError("At least one detection-gradient measurement is required")
    candidates = []
    for value in detection_norms:
        norm = float(value)
        if not math.isfinite(norm) or norm <= 0.0:
            raise ValueError("Detection-gradient norms must be finite and positive")
        candidates.append(ratio * norm / anchor_delta_norm)
    selected = float(format(statistics.median(candidates), ".12g"))
    if not math.isfinite(selected) or selected <= 0.0:
        raise ValueError("Selected lambda is invalid")
    return {
        "target_anchor_gradient_ratio": ratio,
        "anchor_delta_norm": float(anchor_delta_norm),
        "detection_gradient_norms": [float(value) for value in detection_norms],
        "candidate_lambdas": candidates,
        "selection_rule": (
            "median(target_ratio * detection_grad_norm / anchor_delta_norm) "
            "over the pre-evaluation calibration batches"
        ),
        "selected_lambda": selected,
    }


def _layer_index(name):
    parts = name.split(".")
    if len(parts) > 2 and parts[0] == "model" and parts[1].isdigit():
        return int(parts[1])
    return None


def _trainable_group(name):
    layer = _layer_index(name)
    if name.startswith("model.23."):
        return "head"
    if layer is not None and PREFIX <= layer < FEATURES:
        return "features"
    return "other"


def torch_is_finite(tensor):
    import torch

    return bool(torch.isfinite(tensor).all())


def _anchor_delta_summary(start_state, anchor_state, trainable_names):
    records = {}
    names_by_group = {"features": [], "head": [], "other": [], "trainable": []}
    for name in trainable_names:
        if name not in anchor_state:
            raise ValueError(f"Anchor lacks trainable parameter: {name}")
        left = start_state[name]
        right = anchor_state[name]
        if left.shape != right.shape:
            raise ValueError(f"Anchor parameter shape differs: {name}")
        group = _trainable_group(name)
        names_by_group[group].append(name)
        names_by_group["trainable"].append(name)
    for group, names in names_by_group.items():
        sum_sq = 0.0
        numel = 0
        max_abs = 0.0
        for name in names:
            delta = (anchor_state[name].float() - start_state[name].float()).double()
            sum_sq += float(delta.square().sum())
            numel += delta.numel()
            max_abs = max(max_abs, float(delta.abs().max()))
        records[group] = {
            "tensors": len(names),
            "numel": numel,
            "sum_sq": sum_sq,
            "rms": (sum_sq / numel) ** 0.5 if numel else 0.0,
            "l2_norm": sum_sq**0.5,
            "max_abs": max_abs,
        }
    return records


def _normalized_config(config):
    return V22._schedule_without_output(config)


def _validate_fixed_anchor_state(start_state, anchor_state):
    import torch
    from backbone_freeze_audit import is_backbone_key

    fixed = {
        name: value
        for name, value in start_state.items()
        if is_backbone_key(name, PREFIX)
        or (
            is_backbone_key(name, FEATURES)
            and name.endswith(("running_mean", "running_var", "num_batches_tracked"))
        )
        or name == "model.23.dfl.conv.weight"
    }
    changed = [
        name
        for name, value in fixed.items()
        if name not in anchor_state or not torch.equal(anchor_state[name], value)
    ]
    if changed:
        raise ValueError(f"v21 last changed fixed v16 state: {changed[:10]}")
    return sorted(fixed)


class ProximalAnchorAudit:
    """Measure, inject, and persist the fixed L2-SP anchor intervention."""

    def __init__(
        self,
        out,
        start_state,
        anchor_state,
        anchor_path,
        target_ratio=None,
        calibration_batches=None,
        anchor_lambda=None,
    ):
        self.out = Path(out)
        self.start_state = start_state
        self.anchor_state = anchor_state
        self.anchor_path = Path(anchor_path)
        self.target_ratio = target_ratio
        self.calibration_batches = calibration_batches
        self.anchor_lambda = anchor_lambda
        self.mode = "calibration" if anchor_lambda is None else "candidate"
        self.params = []
        self.optimizer = None
        self.handle = None
        self.records = []
        self.epochs = []
        self.pending_index = None
        self.anchor_summary = None
        self.fixed_anchor_state_keys = None

    def on_start(self, trainer):
        import torch

        if self.params or self.optimizer is not None:
            raise RuntimeError("Proximal audit was already attached")
        if trainer.accumulate != 1:
            raise RuntimeError("V24 requires one optimizer attempt per batch")
        if self.mode == "calibration":
            if self.anchor_lambda is not None:
                raise ValueError("Calibration must not receive a precomputed lambda")
            if self.calibration_batches < 3:
                raise ValueError("At least three calibration batches are required")
        elif not math.isfinite(self.anchor_lambda) or self.anchor_lambda <= 0.0:
            raise ValueError("Candidate training requires a positive fixed lambda")

        named = dict(trainer.model.named_parameters())
        trainable_names = sorted(name for name, value in named.items() if value.requires_grad)
        if not trainable_names:
            raise RuntimeError("No trainable parameters were found")
        self.anchor_summary = _anchor_delta_summary(
            self.start_state, self.anchor_state, trainable_names
        )
        trainable_ids = {id(named[name]) for name in trainable_names}
        optimizer_ids = {
            id(parameter)
            for group in trainer.optimizer.param_groups
            for parameter in group["params"]
        }
        if not trainable_ids.issubset(optimizer_ids):
            raise RuntimeError("Optimizer does not contain every trainable parameter")
        self.optimizer_parameter_tensors = len(optimizer_ids)
        self.frozen_optimizer_parameter_tensors = len(optimizer_ids - trainable_ids)

        for name in trainable_names:
            parameter = named[name]
            anchor = self.anchor_state[name].to(
                device=parameter.device, dtype=parameter.dtype
            )
            if not torch.isfinite(anchor).all():
                raise RuntimeError(f"Nonfinite anchor tensor: {name}")
            self.params.append((name, parameter, anchor.detach()))

        self.fixed_anchor_state_keys = _validate_fixed_anchor_state(
            self.start_state, self.anchor_state
        )
        self.optimizer = trainer.optimizer
        self.handle = self.optimizer.register_step_post_hook(self._on_step)
        write(
            self.out / "proximal_anchor_identity.json",
            {
                "role": ROLE,
                "mode": self.mode,
                "anchor_path": str(self.anchor_path.resolve()),
                "anchor_sha256": sha256(self.anchor_path),
                "trainable_tensors": len(trainable_names),
                "optimizer_parameter_tensors": self.optimizer_parameter_tensors,
                "frozen_optimizer_parameter_tensors": (
                    self.frozen_optimizer_parameter_tensors
                ),
                "anchor_delta": self.anchor_summary,
                "fixed_state_keys_equal": self.fixed_anchor_state_keys,
                "target_anchor_gradient_ratio": self.target_ratio,
                "anchor_lambda": self.anchor_lambda,
                "calibration_batches": self.calibration_batches,
            },
        )
        print(
            "V24_ANCHOR: "
            f"mode={self.mode}, tensors={len(trainable_names)}, "
            f"delta_l2={self.anchor_summary['trainable']['l2_norm']:.6g}",
            flush=True,
        )

    def _on_step(self, optimizer, args, kwargs):
        if optimizer is not self.optimizer:
            raise RuntimeError("Observed optimizer identity changed")
        if self.pending_index is None:
            raise RuntimeError("Optimizer stepped without a proximal attempt")
        record = self.records[self.pending_index]
        if record["actual_optimizer_step"]:
            raise RuntimeError("One proximal attempt was counted twice")
        record["actual_optimizer_step"] = True

    def _gradient_record(self):
        import torch

        group_sums = {
            "features": 0.0,
            "head": 0.0,
            "other": 0.0,
            "trainable": 0.0,
        }
        parameter_distance_sums = {
            "features": 0.0,
            "head": 0.0,
            "other": 0.0,
            "trainable": 0.0,
        }
        nonfinite = []
        missing = []
        for name, parameter, anchor in self.params:
            group = _trainable_group(name)
            distance_sum = float(
                (parameter.detach().float().double() - anchor.float().double())
                .square()
                .sum()
            )
            parameter_distance_sums[group] += distance_sum
            parameter_distance_sums["trainable"] += distance_sum
            if parameter.grad is None:
                missing.append(name)
                continue
            if not torch.isfinite(parameter.grad).all():
                nonfinite.append(name)
                continue
            grad_sum = float(
                parameter.grad.detach().float().double().square().sum()
            )
            group_sums[group] += grad_sum
            group_sums["trainable"] += grad_sum
        if nonfinite:
            return None, parameter_distance_sums, nonfinite, missing
        return group_sums, parameter_distance_sums, [], missing

    def before_clip(self, cadence, epoch):
        import torch

        if self.optimizer is None:
            raise RuntimeError("Proximal audit is not attached")
        if self.pending_index is not None:
            raise RuntimeError("A prior proximal attempt was not finalized")

        record = {
            "batch": cadence.batches + 1,
            "epoch": int(epoch),
            "actual_optimizer_step": False,
            "scaler_skipped": None,
            "nonfinite_detection_gradient": False,
            "calibration_eligible": False,
        }
        (
            group_sums,
            current_distance_sums,
            nonfinite,
            missing,
        ) = self._gradient_record()
        record.update(
            {
                "missing_detection_gradient_parameter_count": len(missing),
                "missing_detection_gradient_parameters": missing,
            }
        )
        if nonfinite:
            record.update(
                {
                    "nonfinite_detection_gradient": True,
                    "nonfinite_parameter_count": len(nonfinite),
                    "nonfinite_parameters": nonfinite[:10],
                    "detection_grad_norm": None,
                    "anchor_delta_norm_at_start": (
                        self.anchor_summary["trainable"]["l2_norm"]
                    ),
                    "current_anchor_distance_norm": None,
                    "detection_grad_norm_by_group": None,
                    "current_anchor_distance_by_group": None,
                }
            )
            self.records.append(record)
            self.pending_index = len(self.records) - 1
            return

        detection_norm = group_sums["trainable"] ** 0.5
        if detection_norm <= 0.0:
            raise RuntimeError("Detection gradient norm is zero")
        anchor_delta_norm = self.anchor_summary["trainable"]["l2_norm"]
        record.update(
            {
                "detection_grad_norm": detection_norm,
                "anchor_delta_norm_at_start": anchor_delta_norm,
                "current_anchor_distance_norm": (
                    current_distance_sums["trainable"] ** 0.5
                ),
                "detection_grad_norm_by_group": {
                    name: value**0.5 for name, value in group_sums.items()
                },
                "current_anchor_distance_by_group": {
                    name: value**0.5
                    for name, value in current_distance_sums.items()
                },
            }
        )

        if self.mode == "calibration":
            candidate = self.target_ratio * detection_norm / anchor_delta_norm
            record["candidate_lambda"] = candidate
            record["target_anchor_gradient_ratio"] = self.target_ratio
            record["calibration_eligible"] = True
            self.records.append(record)
            eligible = sum(
                row["calibration_eligible"] for row in self.records
            )
            if eligible >= self.calibration_batches:
                self.pending_index = None
                raise CalibrationComplete(
                    f"Collected {eligible} finite calibration batches"
                )
            self.pending_index = len(self.records) - 1
            return

        anchor_gradient_norm = self.anchor_lambda * anchor_delta_norm
        with torch.no_grad():
            for _, parameter, anchor in self.params:
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                parameter.grad.add_(
                    parameter.detach() - anchor, alpha=self.anchor_lambda
                )
        combined_sums, _, nonfinite, combined_missing = self._gradient_record()
        if nonfinite:
            raise RuntimeError(
                f"Nonfinite gradient after anchor injection: {nonfinite[:5]}"
            )
        if combined_missing:
            raise RuntimeError(
                "Missing gradient after anchor injection: "
                f"{combined_missing[:5]}"
            )
        combined_norm = combined_sums["trainable"] ** 0.5
        record.update(
            {
                "anchor_lambda": self.anchor_lambda,
                "anchor_grad_norm_at_start": anchor_gradient_norm,
                "anchor_to_detection_grad_ratio": (
                    anchor_gradient_norm / detection_norm
                ),
                "combined_grad_norm_preclip": combined_norm,
                "combined_grad_norm_by_group_preclip": {
                    name: value**0.5 for name, value in combined_sums.items()
                },
            }
        )
        self.records.append(record)
        self.pending_index = len(self.records) - 1

    def after_clip(self):
        if self.mode == "calibration":
            return
        if self.pending_index is None:
            raise RuntimeError("No pending proximal attempt after clipping")
        if self.records[self.pending_index]["nonfinite_detection_gradient"]:
            self.records[self.pending_index][
                "combined_grad_norm_postclip"
            ] = None
            return
        sum_sq = 0.0
        for _, parameter, _ in self.params:
            if parameter.grad is None:
                raise RuntimeError("Gradient disappeared during clipping")
            sum_sq += float(parameter.grad.detach().float().double().square().sum())
        self.records[self.pending_index]["combined_grad_norm_postclip"] = sum_sq**0.5

    def after_scaler_step(self):
        if self.pending_index is None:
            raise RuntimeError("No pending proximal attempt after scaler step")
        record = self.records[self.pending_index]
        record["scaler_skipped"] = not record["actual_optimizer_step"]
        self.pending_index = None

    def epoch_end(self, trainer):
        if trainer.optimizer is not self.optimizer:
            raise RuntimeError("Trainer replaced the observed optimizer")
        epoch = int(trainer.epoch) + 1
        pending = [row for row in self.records if row["epoch"] == epoch]
        self.epochs.append(
            {
                "epoch": epoch,
                "attempts": len(pending),
                "actual_optimizer_steps": sum(
                    row["actual_optimizer_step"] for row in pending
                ),
                "scaler_skipped_attempts": sum(
                    bool(row["scaler_skipped"]) for row in pending
                ),
            }
        )
        self.save()

    def calibration_result(self, source_sha256, data_manifest_sha256, config):
        if self.mode != "calibration":
            raise ValueError("Calibration summary requested from candidate audit")
        eligible_records = [
            row for row in self.records if row["calibration_eligible"]
        ]
        detection_norms = [
            row["detection_grad_norm"] for row in eligible_records
        ]
        if len(detection_norms) != self.calibration_batches:
            raise RuntimeError("Calibration did not collect the requested batches")
        selected = select_lambda(
            detection_norms,
            self.anchor_summary["trainable"]["l2_norm"],
            self.target_ratio,
        )
        return {
            "role": ROLE,
            "calibration_complete": True,
            "script_sha256": sha256(Path(__file__)),
            "source_sha256": source_sha256,
            "anchor_path": str(self.anchor_path.resolve()),
            "anchor_sha256": sha256(self.anchor_path),
            "data_manifest_sha256": data_manifest_sha256,
            "config": _normalized_config(config),
            "anchor_delta": self.anchor_summary,
            "fixed_state_keys_equal": self.fixed_anchor_state_keys,
            "calibration_batches": self.calibration_batches,
            "attempted_batches": len(self.records),
            "records": eligible_records,
            **selected,
        }

    def save(self):
        write(
            self.out / "proximal_anchor_audit.json",
            {
                "scope": (
                    "Per-optimizer-attempt detection and L2-SP measurements; "
                    "gradient norms are not inference quality claims"
                ),
                "role": ROLE,
                "mode": self.mode,
                "anchor_path": str(self.anchor_path.resolve()),
                "anchor_sha256": sha256(self.anchor_path),
                "anchor_lambda": self.anchor_lambda,
                "target_anchor_gradient_ratio": self.target_ratio,
                "anchor_delta": self.anchor_summary,
                "epochs": self.epochs,
                "attempts": len(self.records),
                "actual_optimizer_steps": sum(
                    row["actual_optimizer_step"] for row in self.records
                ),
                "records": self.records,
            },
        )

    def close(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        if self.anchor_summary is not None:
            self.save()


def _trainer_class(plan, sampling_audit, partial_audit, proximal_audit):
    import torch

    base = V22.trainer_class(plan, sampling_audit)

    class ProximalTrainer(base):
        def optimizer_step(self):
            self.scaler.unscale_(self.optimizer)
            proximal_audit.before_clip(partial_audit.cadence, self.epoch + 1)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            proximal_audit.after_clip()
            self.scaler.step(self.optimizer)
            proximal_audit.after_scaler_step()
            self.scaler.update()
            self.optimizer.zero_grad()
            if self.ema:
                self.ema.update(self.model)

    return ProximalTrainer


def _validate_calibration(calibration, config, data):
    required = {
        "role": ROLE,
        "calibration_complete": True,
        "script_sha256": sha256(Path(__file__)),
        "source_sha256": V22.START_SHA,
        "anchor_sha256": ANCHOR_SHA,
        "data_manifest_sha256": V22.DATA_SHA,
        "config": _normalized_config(config),
    }
    for name, expected in required.items():
        if calibration.get(name) != expected:
            raise ValueError(f"V24 calibration identity differs: {name}")
    if Path(calibration["anchor_path"]).resolve() != ANCHOR_PATH.resolve():
        raise ValueError("V24 calibration used a different anchor path")
    records = calibration.get("records", [])
    if len(records) != calibration.get("calibration_batches"):
        raise ValueError("V24 calibration record count differs")
    if calibration.get("attempted_batches", 0) < len(records):
        raise ValueError("V24 calibration attempted-batch count is invalid")
    selected = select_lambda(
        [row["detection_grad_norm"] for row in records],
        calibration["anchor_delta_norm"],
        calibration["target_anchor_gradient_ratio"],
    )
    if selected["selected_lambda"] != calibration.get("selected_lambda"):
        raise ValueError("V24 calibrated lambda cannot be reconstructed")
    batches = [row.get("batch") for row in records]
    if (
        any(not isinstance(batch, int) for batch in batches)
        or batches != sorted(set(batches))
        or not all(
            row.get("calibration_eligible") is True
            and math.isfinite(row.get("detection_grad_norm", float("nan")))
            and math.isfinite(row.get("candidate_lambda", float("nan")))
            and row.get("missing_detection_gradient_parameter_count")
            == len(row.get("missing_detection_gradient_parameters", []))
            for row in records
        )
    ):
        raise ValueError("V24 calibration contains invalid gradient records")
    if sha256(data.parent / "build_manifest.json") != V22.DATA_SHA:
        raise ValueError("V24 dataset manifest changed")
    return calibration["selected_lambda"]


def _validate_proximal_completion(meta, proximal, cadence, sampling):
    V22.validate_completion(meta, meta["partial_epoch_audit"], cadence, sampling)
    if proximal.get("role") != ROLE or proximal.get("mode") != "candidate":
        raise ValueError("Wrong proximal audit identity")
    if proximal["anchor_sha256"] != ANCHOR_SHA:
        raise ValueError("Completed run used a different anchor")
    if proximal["anchor_lambda"] != meta["proximal"]["anchor_lambda"]:
        raise ValueError("Completed run lambda differs from preflight")
    records = proximal["records"]
    if len(records) != meta["actual_batches"]:
        raise ValueError("Proximal attempt count differs from actual batches")
    if sum(row["actual_optimizer_step"] for row in records) != len(cadence["steps"]):
        raise ValueError("Proximal actual-step count differs from optimizer cadence")
    if any(row["scaler_skipped"] is None for row in records):
        raise ValueError("A proximal optimizer attempt has an unknown scaler result")
    for row in records:
        if row.get("missing_detection_gradient_parameter_count") != len(
            row.get("missing_detection_gradient_parameters", [])
        ):
            raise ValueError("A proximal record has inconsistent missing gradients")
        if row["actual_optimizer_step"]:
            names = (
                "detection_grad_norm",
                "anchor_grad_norm_at_start",
                "combined_grad_norm_preclip",
                "combined_grad_norm_postclip",
            )
            if row["nonfinite_detection_gradient"] or not all(
                math.isfinite(row[name]) for name in names
            ):
                raise ValueError("A real optimizer step lacks finite gradient norms")
        elif not row["scaler_skipped"]:
            raise ValueError("A non-step proximal attempt was not scaler-skipped")


def _build_plan_public(plan):
    manifest_entries = []
    for stratum in V22.STRATA_MASS:
        for name in plan.member_names[stratum]:
            manifest_entries.extend(
                [plan.per_entry_weight[stratum]]
                * plan.path_weights[(plan.train_dir / name).resolve()]
            )
    return plan.public_plan(manifest_entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, default=ANCHOR_PATH)
    parser.add_argument("--reference-meta", type=Path, default=REFERENCE_META)
    parser.add_argument("--calibrate-only", action="store_true")
    parser.add_argument(
        "--calibration-batches", type=int, default=DEFAULT_CALIBRATION_BATCHES
    )
    parser.add_argument(
        "--target-anchor-ratio", type=float, default=DEFAULT_TARGET_RATIO
    )
    parser.add_argument("--calibration-result", type=Path)
    args = parser.parse_args()

    import torch
    import ultralytics
    from ultralytics import YOLO

    start = args.start.resolve()
    data = args.data.resolve()
    out = args.out.resolve()
    anchor = args.anchor.resolve()
    reference_path = args.reference_meta.resolve()
    if out.exists():
        raise ValueError("Existing V24 output directory")
    if ultralytics.__version__ != "8.4.70":
        raise RuntimeError(
            f"V24 requires Ultralytics 8.4.70, got {ultralytics.__version__}"
        )
    if sha256(start) != V22.START_SHA:
        raise ValueError("V24 starting checkpoint changed")
    if anchor != ANCHOR_PATH.resolve() or sha256(anchor) != ANCHOR_SHA:
        raise ValueError("V24 anchor must be the audited v21 last checkpoint")

    dataset = verify(data.parent, V22.DATA_SHA)
    plan = V22.SourceSamplingPlan.from_verified_manifest(data.parent, V22.DATA_SHA)
    reference = read(reference_path)
    V22.validate_reference(reference, data)
    if reference["checkpoints"]["last.pt"] != ANCHOR_SHA:
        raise ValueError("v21 reference metadata disagrees with the anchor")

    model = YOLO(str(start))
    if model.names != {0: "fire"}:
        raise ValueError("Wrong class identity")
    rebuilt = verify_trainer_reconstruction(model, start)
    start_state = {
        name: value.detach().cpu().clone()
        for name, value in model.model.state_dict().items()
    }
    anchor_state = {
        name: value.detach().cpu().clone()
        for name, value in YOLO(str(anchor)).model.state_dict().items()
    }
    if start_state.keys() != anchor_state.keys():
        raise ValueError("Anchor state keys differ from the starting model")

    config = frozen_schedule(data, out, start, PREFIX)
    if config["workers"] != 0 or config["device"] != 0:
        raise ValueError("Frozen v21 schedule has an unexpected runtime configuration")
    reference_config = _normalized_config(reference["config"])
    candidate_config = _normalized_config(config)
    if reference_config != candidate_config:
        raise ValueError("V24 schedule differs from v21 beyond output paths")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    target_ratio = None
    calibration_batches = None
    anchor_lambda = None
    calibration_path = None
    calibration = None
    if args.calibrate_only:
        target_ratio = _unit_interval(
            args.target_anchor_ratio, "target_anchor_ratio"
        )
        calibration_batches = int(args.calibration_batches)
        if args.calibration_result is not None:
            raise ValueError("Calibration output is fixed inside --out")
    else:
        if args.calibration_result is None:
            raise ValueError("Candidate training requires --calibration-result")
        calibration_path = args.calibration_result.resolve()
        calibration = read(calibration_path)
        anchor_lambda = _validate_calibration(calibration, config, data)
        target_ratio = calibration["target_anchor_gradient_ratio"]
        calibration_batches = calibration["calibration_batches"]

    marker = out.parent / (out.name + "_preflight.json")
    if marker.exists():
        raise ValueError("Existing V24 preflight marker")

    audit = V22.PartialFeatureAudit(model.model, out)
    sampling_audit = V22.SourceSamplingAudit(out, plan)
    proximal = ProximalAnchorAudit(
        out,
        start_state,
        anchor_state,
        anchor,
        target_ratio=target_ratio,
        calibration_batches=calibration_batches,
        anchor_lambda=anchor_lambda,
    )
    Trainer = _trainer_class(plan, sampling_audit, audit, proximal)

    plan_public = _build_plan_public(plan)
    plan_value_sha256 = V22._value_sha256(plan_public)
    identity = {
        "role": ROLE,
        "single_intervention": (
            "fixed L2-SP gradient toward v21 last on trainable parameters"
        ),
        "source_sha256": V22.START_SHA,
        "anchor_path": str(anchor),
        "anchor_sha256": ANCHOR_SHA,
        "dataset": dataset,
        "config": config,
        "frozen_prefix_blocks": PREFIX,
        "feature_blocks": FEATURES,
        "feature_bn_statistics_fixed": True,
        "candidate_training": not args.calibrate_only,
        "checkpoint_screen": PRIMARY,
        "teacher_loss": False,
        "actual_trainer_reconstruction": rebuilt,
        "script_sha256": sha256(Path(__file__)),
        "dependencies": {
            name: sha256(Path(__file__).with_name(name))
            for name in DEPENDENCIES
        },
        "prerequisites": {
            str(reference_path.relative_to(ROOT)): sha256(reference_path)
        },
        "v21_schedule_equal_except_output_paths": True,
        "sampling": {
            "masses": V22.STRATA_MASS,
            "expected_unique_images": plan.unique_images,
            "expected_entries": plan.entries,
            "source_sampling_plan_value_sha256": plan_value_sha256,
            "entry_weights_sha256": plan_public["entry_weights_sha256"],
            "image_weight_pairs_sha256": plan_public["image_weight_pairs_sha256"],
            "manifest_entry_weights_sha256": plan_public[
                "manifest_entry_weights_sha256"
            ],
        },
        "proximal": {
            "anchor_lambda": anchor_lambda,
            "target_anchor_gradient_ratio": target_ratio,
            "calibration_batches": calibration_batches,
            "calibration_result_sha256": (
                sha256(calibration_path) if calibration_path is not None else None
            ),
            "calibration_selection_rule": (
                None
                if calibration is None
                else calibration["selection_rule"]
            ),
        },
        "nominal_batches_per_epoch": V22.NOMINAL_BATCHES_PER_EPOCH,
        "nominal_batches": V22.NOMINAL_BATCHES,
        "actual_batches": None,
        "actual_optimizer_steps": None,
        "checkpoints": None,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
        },
    }
    if calibration_path is not None:
        identity["prerequisites"][str(calibration_path)] = sha256(calibration_path)
    if not args.calibrate_only:
        write(marker, identity)

    source_bytes = {
        name: Path(__file__).with_name(name).read_bytes()
        for name in [Path(__file__).name] + DEPENDENCIES
    }

    def capture_sources(trainer):
        if Path(trainer.save_dir).resolve() != out:
            raise RuntimeError("Trainer changed the intended output directory")
        write(out / "source_sampling_plan.json", plan_public)
        if V22._value_sha256(read(out / "source_sampling_plan.json")) != (
            plan_value_sha256
        ):
            raise RuntimeError("Source-sampling plan changed at training launch")
        identity["sampling"]["source_sampling_plan_sha256"] = sha256(
            out / "source_sampling_plan.json"
        )
        snapshot = out / "source_snapshot"
        snapshot.mkdir()
        for name, payload in source_bytes.items():
            if Path(__file__).with_name(name).read_bytes() != payload:
                raise RuntimeError("A source dependency changed at training launch")
            (snapshot / name).write_bytes(payload)
        write(out / "v24_identity_in_progress.json", identity)

    def on_epoch_end(trainer):
        sampling_audit.epoch_end(trainer, audit.cadence)
        audit.epoch_end(trainer)
        proximal.epoch_end(trainer)

    model.add_callback("on_train_start", capture_sources)
    model.add_callback("on_train_start", audit.on_start)
    model.add_callback("on_train_start", proximal.on_start)
    model.add_callback("on_train_batch_start", audit.batch_start)
    model.add_callback("on_train_batch_end", audit.cadence.on_batch_end)
    model.add_callback("on_train_epoch_end", on_epoch_end)

    calibration_done = False
    try:
        model.train(trainer=Trainer, **config)
    except CalibrationComplete:
        if not args.calibrate_only:
            raise
        calibration_done = True
    except Exception as error:
        write(
            out / "failed.json",
            {"error_type": type(error).__name__, "message": str(error)},
        )
        raise
    finally:
        proximal.close()
        audit.cadence.close()

    if args.calibrate_only:
        if not calibration_done:
            raise RuntimeError("Calibration trainer returned without completing")
        result = proximal.calibration_result(V22.START_SHA, V22.DATA_SHA, config)
        write(out / "calibration_result.json", result)
        write(out / "v24_calibration_identity.json", identity)
        print(
            "V24 calibration complete: "
            f"batches={calibration_batches}, "
            f"target_ratio={target_ratio}, "
            f"lambda={result['selected_lambda']:.12g}",
            flush=True,
        )
        return

    fixed = {
        **audit.freeze.reference,
        **audit.feature_bn,
        "model.23.dfl.conv.weight": audit.initial["model.23.dfl.conv.weight"],
    }
    checkpoints, adaptations = V22.checkpoint_adaptation(audit, fixed, torch)
    sampling = read(out / "source_sampling_actual.json")
    cadence = read(out / "optimizer_cadence_audit.json")
    partial_epochs = read(out / "partial_feature_update_audit.json")
    proximal_audit = read(out / "proximal_anchor_audit.json")

    identity.update(
        {
            "checkpoints": checkpoints,
            "actual_batches": audit.cadence.batches,
            "actual_optimizer_steps": len(audit.cadence.steps),
            "sampling": {
                **identity["sampling"],
                **sampling_audit.compact_summary(),
            },
            "serialized_frozen_state_equal": True,
            "serialized_adaptation": adaptations,
            "partial_epoch_audit": partial_epochs,
            "completed_audits": {
                name: sha256(out / name)
                for name in [
                    "source_sampling_actual.json",
                    "actual_initialization_audit.json",
                    "frozen_prefix_audit.json",
                    "partial_feature_update_audit.json",
                    "optimizer_cadence_audit.json",
                    "proximal_anchor_identity.json",
                    "proximal_anchor_audit.json",
                ]
            },
        }
    )
    _validate_proximal_completion(identity, proximal_audit, cadence, sampling)
    write(out / "experiment_meta.json", identity)
    print(
        "V24 completed: "
        f"draws={V22.DRAWS_PER_EPOCH}, batches={identity['actual_batches']}, "
        f"optimizer_steps={identity['actual_optimizer_steps']}, "
        f"lambda={anchor_lambda:.12g}; "
        "best/last still require fixed held-out comparisons",
        flush=True,
    )


if __name__ == "__main__":
    main()
