"""Recompute temporal fixed-ROI rescue effects from saved predictions only.

Reads the completed comparison summary, plan, and per-frame JSONL. It never
loads video, weights, model libraries, or writes result files. JSON is stdout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

MODEL_NAMES = ("v16_best", "v18_last", "v24_last")
VIDEO_INDEX = 7
VIDEO7_FRAMES = 734
TOTAL_FRAMES = 1014
VIDEO_COUNT = 9
WINDOW = (708, 719)
POINT_FRAMES = (252, 560)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-standard JSON numeric constant: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json_bytes(data: bytes, label: str) -> Any:
    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid UTF-8 JSON in {label}: {exc}") from exc


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_completion_hashes(
    summary: dict[str, Any], plan_bytes: bytes, per_frame_bytes: bytes
) -> tuple[str, str]:
    plan_expected = summary.get("plan_sha256")
    _require(_is_sha256(plan_expected), "completion summary has invalid plan_sha256")
    plan_actual = _sha256(plan_bytes)
    _require(plan_actual == plan_expected, "plan.json SHA-256 does not match completion summary")

    data_hashes = [
        value
        for key, value in summary.items()
        if key != "plan_sha256" and _is_sha256(value)
    ]
    _require(
        len(data_hashes) == 1,
        "completion summary must contain exactly one per-frame SHA-256 besides plan_sha256",
    )
    records_actual = _sha256(per_frame_bytes)
    _require(
        records_actual == data_hashes[0],
        "per_frame.jsonl SHA-256 does not match completion summary",
    )
    return plan_actual, records_actual


def _finite_number(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be numeric",
    )
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    _require(math.isfinite(number), f"{label} must be finite")
    return number


def _validate_boxes(boxes: Any, label: str) -> None:
    _require(isinstance(boxes, list), f"{label} must be a list")
    for index, box in enumerate(boxes):
        where = f"{label}[{index}]"
        _require(isinstance(box, dict), f"{where} must be an object")
        confidence = _finite_number(box.get("confidence"), f"{where}.confidence")
        _require(0.0 <= confidence <= 1.0, f"{where}.confidence is outside [0, 1]")
        coords = box.get("xyxy")
        _require(isinstance(coords, list) and len(coords) == 4, f"{where}.xyxy must have four coordinates")
        x1, y1, x2, y2 = (
            _finite_number(value, f"{where}.xyxy[{i}]")
            for i, value in enumerate(coords)
        )
        _require(x1 < x2 and y1 < y2, f"{where}.xyxy has non-positive area")


def _validate_record(record: Any, line_number: int) -> tuple[int, int, float]:
    label = f"per_frame.jsonl line {line_number}"
    _require(isinstance(record, dict), f"{label} must be an object")
    for key in ("video_index", "frame_index", "time_seconds", "width", "height",
                "source_pixel_sha256", "roi_xyxy", "predictions"):
        _require(key in record, f"{label} is missing {key}")

    video_index = record["video_index"]
    frame_index = record["frame_index"]
    _require(type(video_index) is int and 1 <= video_index <= VIDEO_COUNT,
             f"{label}.video_index must be an integer from 1 to {VIDEO_COUNT}")
    _require(type(frame_index) is int and frame_index >= 0,
             f"{label}.frame_index must be a non-negative integer")
    timestamp = _finite_number(record["time_seconds"], f"{label}.time_seconds")
    _require(timestamp >= 0, f"{label}.time_seconds must be non-negative")
    for key in ("width", "height"):
        _require(type(record[key]) is int and record[key] > 0,
                 f"{label}.{key} must be a positive integer")
    _require(_is_sha256(record["source_pixel_sha256"]),
             f"{label}.source_pixel_sha256 must be a SHA-256 hex string")

    roi = record["roi_xyxy"]
    _require(isinstance(roi, list) and len(roi) == 4,
             f"{label}.roi_xyxy must have four coordinates")
    rx1, ry1, rx2, ry2 = (
        _finite_number(value, f"{label}.roi_xyxy[{i}]")
        for i, value in enumerate(roi)
    )
    _require(0 <= rx1 < rx2 <= record["width"] and
             0 <= ry1 < ry2 <= record["height"],
             f"{label}.roi_xyxy is outside the frame or has no area")

    predictions = record["predictions"]
    _require(isinstance(predictions, dict), f"{label}.predictions must be an object")
    _require(set(predictions) == set(MODEL_NAMES),
             f"{label}.predictions must contain exactly {', '.join(MODEL_NAMES)}")
    for model_name in MODEL_NAMES:
        pred = predictions[model_name]
        where = f"{label}.predictions.{model_name}"
        _require(isinstance(pred, dict), f"{where} must be an object")
        for key in ("full_frame", "crop_attempted", "crop_predictions"):
            _require(key in pred, f"{where} is missing {key}")
        _validate_boxes(pred["full_frame"], f"{where}.full_frame")
        _validate_boxes(pred["crop_predictions"], f"{where}.crop_predictions")
        _require(type(pred["crop_attempted"]) is bool,
                 f"{where}.crop_attempted must be boolean")
        should_attempt = len(pred["full_frame"]) == 0
        _require(pred["crop_attempted"] is should_attempt,
                 f"{where}.crop_attempted conflicts with full_frame boxes")
        _require(pred["crop_attempted"] or not pred["crop_predictions"],
                 f"{where} has crop boxes although crop inference was not attempted")
    return video_index, frame_index, timestamp


def _empty_runs(
    empty: list[bool], timestamps: list[float]
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    start: int | None = None
    for index in range(len(empty) + 1):
        is_empty = empty[index] if index < len(empty) else False
        if is_empty and start is None:
            start = index
        elif not is_empty and start is not None:
            end = index - 1
            runs.append({
                "start_frame": start,
                "end_frame": end,
                "frames": end - start + 1,
                "start_time_seconds": timestamps[start],
                "end_time_seconds": timestamps[end],
            })
            start = None
    return runs


def _model_frame_state(record: dict[str, Any], model_name: str) -> dict[str, Any]:
    pred = record["predictions"][model_name]
    full_count = len(pred["full_frame"])
    crop_count = len(pred["crop_predictions"])
    return {
        "frame_index": record["frame_index"],
        "time_seconds": record["time_seconds"],
        "full_frame_box_count": full_count,
        "crop_attempted": pred["crop_attempted"],
        "crop_box_count": crop_count,
        "rescued": full_count == 0 and crop_count > 0,
        "post_rescue_empty": full_count == 0 and crop_count == 0,
    }


def analyze_video7_records(
    records: Iterable[dict[str, Any]], expected_frames: int = VIDEO7_FRAMES
) -> dict[str, Any]:
    _require(type(expected_frames) is int and expected_frames > 0,
             "expected_frames must be a positive integer")
    per_video: dict[int, dict[int, tuple[float, dict[str, Any]]]] = defaultdict(dict)
    all_totals = {
        name: {"full_empty": 0, "crop_any": 0}
        for name in MODEL_NAMES
    }

    for line_number, record in enumerate(records, 1):
        video_index, frame_index, timestamp = _validate_record(record, line_number)
        frames = per_video[video_index]
        _require(frame_index not in frames,
                 f"duplicate frame index {frame_index} in video {video_index}")
        frames[frame_index] = (timestamp, record)
        for name in MODEL_NAMES:
            pred = record["predictions"][name]
            if not pred["full_frame"]:
                all_totals[name]["full_empty"] += 1
                if pred["crop_predictions"]:
                    all_totals[name]["crop_any"] += 1

    video7 = per_video.get(VIDEO_INDEX)
    _require(video7 is not None, "cache contains no video_index 7 records")
    expected_indices = set(range(expected_frames))
    actual_indices = set(video7)
    _require(
        actual_indices == expected_indices,
        f"video 7 frame indices must be continuous 0..{expected_frames - 1}; "
        f"missing={sorted(expected_indices - actual_indices)[:12]}, "
        f"unexpected={sorted(actual_indices - expected_indices)[:12]}",
    )
    ordered = [video7[index][1] for index in range(expected_frames)]
    timestamps = [video7[index][0] for index in range(expected_frames)]
    _require(
        all(timestamps[i] < timestamps[i + 1] for i in range(expected_frames - 1)),
        "video 7 time_seconds must strictly increase with frame_index",
    )

    model_results: dict[str, Any] = {}
    for name in MODEL_NAMES:
        before = [not row["predictions"][name]["full_frame"] for row in ordered]
        after = [
            not row["predictions"][name]["full_frame"]
            and not row["predictions"][name]["crop_predictions"]
            for row in ordered
        ]
        rescued = [i for i, row in enumerate(ordered)
                   if not row["predictions"][name]["full_frame"]
                   and bool(row["predictions"][name]["crop_predictions"])]
        before_runs = _empty_runs(before, timestamps)
        after_runs = _empty_runs(after, timestamps)
        before_count = sum(before)
        model_results[name] = {
            "frames": expected_frames,
            "empty_before_frames": before_count,
            "rescued_frames": len(rescued),
            "empty_after_frames": sum(after),
            "empty_frame_reduction_fraction": (
                len(rescued) / before_count if before_count else None
            ),
            "empty_runs_before_count": len(before_runs),
            "empty_runs_after_count": len(after_runs),
            "longest_empty_run_before_frames": max(
                (run["frames"] for run in before_runs), default=0
            ),
            "longest_empty_run_after_frames": max(
                (run["frames"] for run in after_runs), default=0
            ),
            "rescued_frame_indices": rescued,
            "empty_runs_before": before_runs,
            "empty_runs_after": after_runs,
        }

    window_start, window_end = WINDOW
    _require(window_end < expected_frames,
             f"video 7 does not contain required window {window_start}-{window_end}")
    window_frames = ordered[window_start:window_end + 1]
    late_window: dict[str, Any] = {
        "start_frame": window_start,
        "end_frame": window_end,
        "frames": len(window_frames),
        "models": {},
    }
    for name in MODEL_NAMES:
        states = [_model_frame_state(row, name) for row in window_frames]
        late_window["models"][name] = {
            "empty_before_frames": [
                state["frame_index"] for state in states
                if state["full_frame_box_count"] == 0
            ],
            "rescued_frame_indices": [
                state["frame_index"] for state in states if state["rescued"]
            ],
            "empty_after_frames": [
                state["frame_index"] for state in states if state["post_rescue_empty"]
            ],
            "all_frames_have_box_after": all(
                not state["post_rescue_empty"] for state in states
            ),
        }

    selected_frames: dict[str, Any] = {}
    for frame_index in POINT_FRAMES:
        _require(frame_index < expected_frames,
                 f"video 7 does not contain required frame {frame_index}")
        row = ordered[frame_index]
        selected_frames[str(frame_index)] = {
            "frame_index": frame_index,
            "time_seconds": row["time_seconds"],
            "models": {
                name: _model_frame_state(row, name) for name in MODEL_NAMES
            },
        }

    return {
        "video_index": VIDEO_INDEX,
        "frames": expected_frames,
        "models": model_results,
        "window_708_719": late_window,
        "selected_frames_252_560": selected_frames,
        "cache_totals_recomputed": all_totals,
    }


def analyze_comparison_dir(comparison_dir: Path) -> dict[str, Any]:
    summary_path = comparison_dir / "comparison_complete.json"
    plan_path = comparison_dir / "plan.json"
    records_path = comparison_dir / "per_frame.jsonl"
    try:
        summary_bytes = summary_path.read_bytes()
        plan_bytes = plan_path.read_bytes()
        records_bytes = records_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Cannot read complete comparison cache: {exc}") from exc

    summary = _load_json_bytes(summary_bytes, str(summary_path))
    _require(isinstance(summary, dict), "comparison_complete.json must be an object")
    plan_hash, records_hash = verify_completion_hashes(
        summary, plan_bytes, records_bytes
    )
    plan = _load_json_bytes(plan_bytes, str(plan_path))
    _require(isinstance(plan, dict), "plan.json must be an object")
    videos = plan.get("videos")
    _require(isinstance(videos, list) and len(videos) == VIDEO_COUNT,
             f"plan.json must declare exactly {VIDEO_COUNT} videos")
    plan_models = plan.get("models")
    _require(isinstance(plan_models, list), "plan.json models must be a list")
    labels = [model.get("label") for model in plan_models if isinstance(model, dict)]
    _require(len(labels) == len(MODEL_NAMES) and set(labels) == set(MODEL_NAMES),
             "plan.json must declare exactly the three expected models")

    _require(summary.get("role") == "Fixed crop rescue diagnostic on exposed development media",
             "completion summary role is unexpected")
    _require(type(summary.get("frames")) is int and summary["frames"] == TOTAL_FRAMES,
             f"completion summary must report {TOTAL_FRAMES} frames")
    for flag in ("ground_truth_provided", "independent_test", "on_board"):
        _require(summary.get(flag) is False,
                 f"completion summary {flag} must be false for this cache")

    records: list[dict[str, Any]] = []
    try:
        text = records_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"per_frame.jsonl is not UTF-8: {exc}") from exc
    for line_number, line in enumerate(text.splitlines(), 1):
        _require(bool(line.strip()), f"per_frame.jsonl contains a blank line at {line_number}")
        records.append(_load_json_bytes(line.encode("utf-8"), f"{records_path}:{line_number}"))
    _require(len(records) == summary["frames"],
             "per_frame.jsonl row count does not match completion summary")

    analysis = analyze_video7_records(records)
    recomputed_totals = analysis["cache_totals_recomputed"]
    declared_totals = summary.get("totals")
    _require(isinstance(declared_totals, dict), "completion summary totals must be an object")
    declared_by_video = summary.get("by_video")
    _require(isinstance(declared_by_video, dict), "completion summary by_video must be an object")
    for name in MODEL_NAMES:
        declared = declared_totals.get(name)
        _require(isinstance(declared, dict), f"completion summary totals missing {name}")
        _require(declared.get("full_empty") == recomputed_totals[name]["full_empty"],
                 f"completion full_empty total differs from cache for {name}")
        _require(declared.get("roi_inferred") == recomputed_totals[name]["full_empty"],
                 f"completion roi_inferred total differs from cache for {name}")
        _require(declared.get("roi_any_box") == recomputed_totals[name]["crop_any"],
                 f"completion roi_any_box total differs from cache for {name}")

        video_stats: dict[str, dict[str, int]] = {}
        for video_index in range(1, VIDEO_COUNT + 1):
            rows = [
                row for row in records if row["video_index"] == video_index
            ]
            video_stats[str(video_index)] = {
                "full_empty": sum(not row["predictions"][name]["full_frame"] for row in rows),
                "crop_any": sum(
                    not row["predictions"][name]["full_frame"]
                    and bool(row["predictions"][name]["crop_predictions"])
                    for row in rows
                ),
            }
            declared_video = declared_by_video.get(str(video_index))
            _require(isinstance(declared_video, dict),
                     f"completion summary missing video {video_index}")
            declared_model = declared_video.get(name)
            _require(isinstance(declared_model, dict),
                     f"completion summary missing video {video_index} model {name}")
            _require(declared_model.get("full_empty") == video_stats[str(video_index)]["full_empty"],
                     f"completion video {video_index} full_empty differs for {name}")
            _require(declared_model.get("crop_any") == video_stats[str(video_index)]["crop_any"],
                     f"completion video {video_index} crop_any differs for {name}")

    analysis.pop("cache_totals_recomputed")
    analysis["source"] = {
        "comparison_dir": str(comparison_dir),
        "completion_summary_sha256": _sha256(summary_bytes),
        "plan_sha256": plan_hash,
        "per_frame_jsonl_sha256": records_hash,
        "cached_rows": len(records),
        "ground_truth_provided": summary["ground_truth_provided"],
        "independent_test": summary["independent_test"],
        "on_board": summary["on_board"],
    }
    return analysis


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_dir = Path(__file__).resolve().parents[1] / "out" / "fixed_roi_rescue_20261029"
    parser.add_argument("--comparison-dir", type=Path, default=default_dir)
    args = parser.parse_args(argv)
    try:
        result = analyze_comparison_dir(args.comparison_dir)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2, allow_nan=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
