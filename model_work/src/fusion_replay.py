"""Offline, explainable kitchen-risk state machine for development replay.

This is not a certified alarm and must not control safety actuators. Its
thresholds are supplied by a config file; the bundled config is synthetic and
uncalibrated. Missing evidence produces UNKNOWN rather than an all-clear.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


STATES = ("NORMAL", "WATCH", "WARNING", "ALARM", "UNKNOWN")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def boolean(value: str | bool | int | None, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if value in (1, "1", "true", "True", "TRUE"):
        return True
    if value in (0, "0", "false", "False", "FALSE", ""):
        return False
    raise ValueError(f"{field} must be 0 or 1, got {value!r}")


def optional_float(value: str | None, field: str) -> float | None:
    if value is None or value.strip() == "":
        return None
    result = float(value)
    if not -1e9 < result < 1e9:
        raise ValueError(f"{field} out of supported range")
    return result


@dataclass(frozen=True)
class Config:
    calibrated: bool
    fire_conf_min: float
    outside_frac_min: float
    smoke_watch_index: float
    smoke_alarm_index: float
    temp_watch_delta_c: float
    temp_alarm_delta_c: float
    humidity_steam_hint_delta_pct: float
    warning_persist_s: float
    alarm_persist_s: float
    clear_hold_s: float
    max_gap_s: float

    @classmethod
    def read(cls, path: Path) -> "Config":
        data = json.loads(path.read_text(encoding="utf-8"))
        required = set(cls.__dataclass_fields__)
        if set(data) != required:
            raise ValueError(f"Config keys mismatch: missing={required-set(data)}, extra={set(data)-required}")
        config = cls(**data)
        if not isinstance(config.calibrated, bool):
            raise ValueError("calibrated must be a JSON boolean")
        if not (0 < config.fire_conf_min <= 1 and 0 < config.outside_frac_min <= 1):
            raise ValueError("Invalid visual thresholds")
        if not (0 <= config.smoke_watch_index < config.smoke_alarm_index <= 1):
            raise ValueError("Smoke index thresholds must increase within [0,1]")
        if not (0 <= config.temp_watch_delta_c < config.temp_alarm_delta_c):
            raise ValueError("Temperature thresholds must increase")
        if any(getattr(config, field) <= 0 for field in
               ("warning_persist_s", "alarm_persist_s", "clear_hold_s", "max_gap_s")):
            raise ValueError("Time thresholds must be positive")
        return config


@dataclass(frozen=True)
class Observation:
    clip_id: str
    t_s: float
    fire_conf: float | None
    flame_outside_frac: float | None
    flame_area_frac: float | None
    smoke_index: float | None
    temp_delta_c: float | None
    humidity_delta_pct: float | None
    certified_smoke_alarm: bool
    visual_ok: bool
    smoke_ok: bool
    temp_ok: bool
    reset_requested: bool

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "Observation":
        required = set(cls.__dataclass_fields__)
        if not required.issubset(row):
            raise ValueError(f"Missing input columns: {required-set(row)}")
        obs = cls(
            clip_id=row["clip_id"], t_s=float(row["t_s"]),
            fire_conf=optional_float(row["fire_conf"], "fire_conf"),
            flame_outside_frac=optional_float(row["flame_outside_frac"], "flame_outside_frac"),
            flame_area_frac=optional_float(row["flame_area_frac"], "flame_area_frac"),
            smoke_index=optional_float(row["smoke_index"], "smoke_index"),
            temp_delta_c=optional_float(row["temp_delta_c"], "temp_delta_c"),
            humidity_delta_pct=optional_float(row["humidity_delta_pct"], "humidity_delta_pct"),
            certified_smoke_alarm=boolean(row["certified_smoke_alarm"], "certified_smoke_alarm"),
            visual_ok=boolean(row["visual_ok"], "visual_ok"),
            smoke_ok=boolean(row["smoke_ok"], "smoke_ok"),
            temp_ok=boolean(row["temp_ok"], "temp_ok"),
            reset_requested=boolean(row["reset_requested"], "reset_requested"),
        )
        if not obs.clip_id or obs.t_s < 0:
            raise ValueError("clip_id required and t_s must be nonnegative")
        if obs.visual_ok and obs.fire_conf is None:
            raise ValueError("visual_ok=1 requires fire_conf")
        if obs.visual_ok and (obs.flame_outside_frac is None or obs.flame_area_frac is None):
            raise ValueError("visual_ok=1 requires flame geometry, including zero values")
        if obs.smoke_ok and obs.smoke_index is None:
            raise ValueError("smoke_ok=1 requires smoke_index")
        if obs.temp_ok and obs.temp_delta_c is None:
            raise ValueError("temp_ok=1 requires temp_delta_c")
        for name in ("fire_conf", "flame_outside_frac", "flame_area_frac", "smoke_index"):
            value = getattr(obs, name)
            if value is not None and not 0 <= value <= 1:
                raise ValueError(f"{name} must be within [0,1]")
        return obs


class FusionEngine:
    def __init__(self, config: Config):
        self.config = config
        self.last_t: float | None = None
        self.started: dict[str, float] = {}
        self.alarm_latched = False
        self.warning_hold_until = -1.0

    def _persisted(self, name: str, active: bool, t_s: float, seconds: float) -> bool:
        if not active:
            self.started.pop(name, None)
            return False
        self.started.setdefault(name, t_s)
        return t_s - self.started[name] >= seconds

    def step(self, obs: Observation) -> dict[str, str | float | int]:
        c = self.config
        reasons: list[str] = []
        if self.last_t is not None:
            if obs.t_s <= self.last_t:
                raise ValueError(f"Non-increasing time in {obs.clip_id}: {obs.t_s} after {self.last_t}")
            if obs.t_s - self.last_t > c.max_gap_s:
                self.started.clear()
                self.warning_hold_until = -1.0
                reasons.append("data_gap_streaks_reset")
        self.last_t = obs.t_s

        visual_ready = obs.visual_ok and obs.fire_conf is not None
        smoke_ready = obs.smoke_ok and obs.smoke_index is not None
        temp_ready = obs.temp_ok and obs.temp_delta_c is not None
        missing = [name for name, ready in (("visual", visual_ready),
                                             ("smoke", smoke_ready),
                                             ("temperature", temp_ready)) if not ready]
        fire_visible = visual_ready and obs.fire_conf >= c.fire_conf_min
        outside = (fire_visible and obs.flame_outside_frac is not None
                   and obs.flame_outside_frac >= c.outside_frac_min)
        smoke_watch = smoke_ready and obs.smoke_index >= c.smoke_watch_index
        smoke_high = smoke_ready and obs.smoke_index >= c.smoke_alarm_index
        temp_watch = temp_ready and obs.temp_delta_c >= c.temp_watch_delta_c
        temp_high = temp_ready and obs.temp_delta_c >= c.temp_alarm_delta_c
        steam_hint = (smoke_watch and obs.humidity_delta_pct is not None
                      and obs.humidity_delta_pct >= c.humidity_steam_hint_delta_pct
                      and not temp_watch and not outside)
        watch_any = outside or smoke_watch or temp_watch
        warning_outside = self._persisted("outside_warning", outside, obs.t_s, c.warning_persist_s)
        warning_smoke = self._persisted("smoke_warning", smoke_high, obs.t_s, c.warning_persist_s)
        warning_combo = self._persisted("smoke_temp_warning", smoke_watch and temp_watch,
                                        obs.t_s, c.warning_persist_s)
        alarm_visual = self._persisted("visual_alarm", outside and (smoke_high or temp_high),
                                       obs.t_s, c.alarm_persist_s)
        alarm_combo = self._persisted("smoke_temp_alarm", smoke_high and temp_high,
                                      obs.t_s, c.alarm_persist_s)

        active_risk = watch_any or obs.certified_smoke_alarm
        if obs.certified_smoke_alarm:
            self.alarm_latched = True
            reasons.append("independent_smoke_alarm")
        if alarm_visual:
            self.alarm_latched = True
            reasons.append("outside_flame_with_smoke_or_heat_persistent")
        if alarm_combo:
            self.alarm_latched = True
            reasons.append("smoke_and_heat_persistent")
        if obs.reset_requested:
            if self.alarm_latched and not active_risk and not missing:
                self.alarm_latched = False
                self.started.clear()
                self.warning_hold_until = -1.0
                reasons.append("manual_reset_accepted")
            elif self.alarm_latched:
                reasons.append("manual_reset_rejected_active_or_missing_evidence")

        if self.alarm_latched:
            state = "ALARM"
            if not reasons:
                reasons.append("alarm_latched")
        elif warning_outside or warning_smoke or warning_combo:
            state = "WARNING"
            self.warning_hold_until = obs.t_s + c.clear_hold_s
            if warning_outside:
                reasons.append("outside_flame_persistent")
            if warning_smoke:
                reasons.append("high_smoke_persistent")
            if warning_combo:
                reasons.append("smoke_and_heat_rising_persistent")
        elif obs.t_s < self.warning_hold_until:
            state = "WARNING"
            reasons.append("warning_clear_hold")
        elif watch_any:
            state = "WATCH"
            if outside:
                reasons.append("flame_outside_burner_roi")
            if smoke_watch:
                reasons.append("smoke_rising")
            if temp_watch:
                reasons.append("temperature_rising")
            if steam_hint:
                reasons.append("steam_possible_not_a_safety_clear")
        elif missing:
            state = "UNKNOWN"
        else:
            state = "NORMAL"
            if fire_visible:
                reasons.append("visible_flame_within_burner_roi")
            else:
                reasons.append("no_risk_signal_at_this_sample")
        if missing:
            reasons.append("missing:" + ",".join(missing))
        return {"clip_id": obs.clip_id, "t_s": obs.t_s, "state": state,
                "reason": ";".join(reasons), "fire_visible": int(fire_visible),
                "outside_flame": int(outside), "smoke_watch": int(smoke_watch),
                "smoke_high": int(smoke_high), "temp_watch": int(temp_watch),
                "temp_high": int(temp_high), "alarm_latched": int(self.alarm_latched)}


def replay(input_path: Path, config_path: Path, out_dir: Path) -> None:
    if out_dir.exists():
        raise SystemExit(f"Refusing overwrite: {out_dir}")
    config = Config.read(config_path)
    with input_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SystemExit("Empty input")
    if "visual_prediction_index" in rows[0]:
        metadata_path = input_path.with_suffix(input_path.suffix + ".meta.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["output_sha256"] != sha256(input_path):
            raise ValueError("Converted input identity changed")
        if metadata["config_sha256"] != sha256(config_path):
            raise ValueError("Conversion/replay config differs; reconvert with the intended config")
    engines: dict[str, FusionEngine] = {}
    output = []
    transitions = []
    previous: dict[str, str] = {}
    for row in rows:
        obs = Observation.from_row(row)
        engine = engines.setdefault(obs.clip_id, FusionEngine(config))
        result = engine.step(obs)
        output.append(result)
        if previous.get(obs.clip_id) != result["state"]:
            transitions.append({"clip_id": obs.clip_id, "t_s": obs.t_s,
                                "from": previous.get(obs.clip_id, "START"), "to": result["state"],
                                "reason": result["reason"]})
            previous[obs.clip_id] = str(result["state"])
    out_dir.mkdir(parents=True)
    with (out_dir / "per_sample.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    with (out_dir / "transitions.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(transitions[0]))
        writer.writeheader()
        writer.writerows(transitions)
    summary = {"input": str(input_path.resolve()), "input_sha256": sha256(input_path),
               "config": str(config_path.resolve()), "config_sha256": sha256(config_path),
               "calibrated": config.calibrated, "development_only": True,
               "samples": len(output), "clips": len(engines),
               "state_samples": dict(Counter(r["state"] for r in output)),
               "transitions": len(transitions),
               "warning": "Sample counts are not independent event accuracy. Demo thresholds and synthetic data cannot establish fire-safety performance."}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def outside_fraction(box: list[float], roi: list[float]) -> float:
    x1, y1, x2, y2 = box
    area = max(0.0, x2-x1) * max(0.0, y2-y1)
    ix1, iy1 = max(x1, roi[0]), max(y1, roi[1])
    ix2, iy2 = min(x2, roi[2]), min(y2, roi[3])
    inside = max(0.0, ix2-ix1) * max(0.0, iy2-iy1)
    return max(0.0, min(1.0, 1 - inside/area)) if area else 0.0


def select_visual_evidence(boxes, roi, width, height, confidence_min):
    """Keep confidence and geometry from one box, prioritizing eligible outside boxes."""
    if not (isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0):
        raise ValueError("Frame dimensions must be positive integers")
    if not 0 < confidence_min <= 1:
        raise ValueError("Invalid selection confidence threshold")
    def checked_box(box, name):
        if len(box) != 4 or any(not math.isfinite(float(v)) for v in box):
            raise ValueError(f"{name} needs four finite coordinates")
        x1, y1, x2, y2 = map(float, box)
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"{name} outside frame bounds")
        return [x1, y1, x2, y2]
    roi = checked_box(roi, "ROI")
    candidates = []
    for index, box in enumerate(boxes):
        confidence = float(box["confidence"])
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Prediction confidence must be within [0,1]")
        xyxy = checked_box(box["xyxy"], "Prediction")
        candidates.append({"fire_conf": confidence,
                           "flame_outside_frac": outside_fraction(xyxy, roi),
                           "flame_area_frac": (xyxy[2]-xyxy[0])*(xyxy[3]-xyxy[1])/(width*height),
                           "visual_prediction_index": index})
    eligible = [v for v in candidates if v["fire_conf"] >= confidence_min]
    if eligible:
        chosen = max(eligible, key=lambda v: (v["flame_outside_frac"], v["fire_conf"], v["flame_area_frac"]))
    elif candidates:
        chosen = max(candidates, key=lambda v: v["fire_conf"])
    else:
        chosen = {"fire_conf": 0.0, "flame_outside_frac": 0.0,
                  "flame_area_frac": 0.0, "visual_prediction_index": ""}
    return {**chosen, "visual_qualified_boxes": len(eligible)}


def visual_to_fusion(input_path: Path, roi_path: Path, out_path: Path, config_path: Path) -> None:
    metadata_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    if out_path.exists() or metadata_path.exists():
        raise SystemExit(f"Refusing overwrite: {out_path}")
    config = Config.read(config_path)
    rois = json.loads(roi_path.read_text(encoding="utf-8"))
    with input_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SystemExit("Empty visual input")
    output = []
    for row in rows:
        video = row["video"]
        roi_record = rois.get(video)
        if not isinstance(roi_record, dict):
            raise ValueError(f"ROI for {video} must declare width, height and roi_xyxy")
        width, height = int(row["width"]), int(row["height"])
        if (width, height) != (roi_record["width"], roi_record["height"]):
            raise ValueError(f"ROI/frame dimensions differ for {video}")
        t_s = float(row["time_seconds"])
        if not math.isfinite(t_s) or t_s < 0:
            raise ValueError("Visual sample time must be finite and nonnegative")
        boxes = json.loads(row["predictions"])
        evidence = select_visual_evidence(boxes, roi_record["roi_xyxy"], width, height, config.fire_conf_min)
        output.append({"clip_id": video, "t_s": t_s, **evidence,
                       "visual_frame_width": width, "visual_frame_height": height,
                       "smoke_index": "", "temp_delta_c": "", "humidity_delta_pct": "",
                       "certified_smoke_alarm": 0, "visual_ok": 1, "smoke_ok": 0,
                       "temp_ok": 0, "reset_requested": 0})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    metadata = {"development_only": True, "calibrated": config.calibrated, "samples": len(output),
                "input_sha256": sha256(input_path), "roi_sha256": sha256(roi_path),
                "config_sha256": sha256(config_path), "output_sha256": sha256(out_path),
                "converter_sha256": sha256(Path(__file__)),
                "selection": "Largest outside fraction among confidence-eligible boxes; same box supplies confidence and area. Otherwise highest confidence below threshold, or zeros for an empty detection list.",
                "geometry": "Bounding-box area proxies, not segmented flame area or fire severity.",
                "missing_channels": ["smoke", "temperature"],
                "identity_limit": "Input hashes bind these replay files; they do not prove capture timestamps or video/model provenance."}
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Converted {len(output)} development frames with same-box evidence; smoke and temperature explicitly missing")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("replay")
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    convert = commands.add_parser("visual-to-fusion")
    convert.add_argument("--input", type=Path, required=True)
    convert.add_argument("--roi-json", type=Path, required=True)
    convert.add_argument("--config", type=Path, required=True)
    convert.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "replay":
        replay(args.input, args.config, args.out)
    else:
        visual_to_fusion(args.input, args.roi_json, args.out, args.config)


if __name__ == "__main__":
    main()
