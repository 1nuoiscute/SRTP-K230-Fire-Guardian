#!/usr/bin/env python3
"""Development-only nested fire-box comparison from cached predictions."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from data_integrity import iou, matches, sha256


MODELS = ("v16_best", "v18_last", "v21_best", "v21_last")
POLICIES = ("original", "nested_r2", "nested_r3", "nested_r4")
RATIOS = {"original": None, "nested_r2": 2.0, "nested_r3": 3.0, "nested_r4": 4.0}
CONFIDENCE = 0.25
NMS_IOU = 0.60
MATCH_IOU = 0.50
CHILD_COVERAGE = 0.95
CHILD_AREA_MIN = 1e-9


class CheckFailure(ValueError):
    pass


def fail(message: str) -> None:
    raise CheckFailure(message)


def finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{name} is not numeric: {value!r}")
    number = float(value)
    if not math.isfinite(number):
        fail(f"{name} is non-finite: {value!r}")
    return number


def load_json(path: Path, read_files: dict[str, str]) -> Any:
    if not path.is_file():
        fail(f"Missing JSON input: {path}")
    read_files[str(path)] = sha256(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"Invalid JSON {path}: {exc}")


def load_jsonl(path: Path, read_files: dict[str, str]) -> list[Any]:
    if not path.is_file():
        fail(f"Missing JSONL input: {path}")
    read_files[str(path)] = sha256(path)
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception as exc:
            fail(f"Invalid JSONL {path}:{line_no}: {exc}")
    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def xyxy_from(value: Any, context: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        fail(f"{context}: xyxy must be a four-number list")
    box = [finite_number(v, f"{context}[{i}]") for i, v in enumerate(value)]
    if box[2] <= box[0] or box[3] <= box[1]:
        fail(f"{context}: invalid geometry {box}")
    return box


def boxes_from(value: Any, context: str) -> list[list[float]]:
    if not isinstance(value, list):
        fail(f"{context}: boxes must be a list")
    if len(value) == 4 and all(isinstance(v, (int, float)) for v in value):
        return [xyxy_from(value, context)]
    return [xyxy_from(item, f"{context}[{i}]") for i, item in enumerate(value)]


def normalize_prediction(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or "xyxy" not in value:
        fail(f"{context}: prediction object with xyxy required")
    confidence = value.get("confidence", value.get("conf"))
    if confidence is None:
        fail(f"{context}: prediction has no confidence")
    cls = value.get("cls", value.get("class", 0))
    if isinstance(cls, list) and len(cls) == 1:
        cls = cls[0]
    return {
        "confidence": finite_number(confidence, f"{context}.confidence"),
        "xyxy": xyxy_from(value["xyxy"], context),
        "cls": cls,
    }


def normalize_predictions(values: Any, context: str) -> list[dict[str, Any]]:
    if values is None:
        fail(f"{context}: missing predictions")
    if not isinstance(values, list):
        fail(f"{context}: predictions must be a list")
    return [
        normalize_prediction(item, f"{context}[{index}]")
        for index, item in enumerate(values)
    ]


def area(box: list[float]) -> float:
    value = (box[2] - box[0]) * (box[3] - box[1])
    if not math.isfinite(value) or value <= 0:
        fail(f"Invalid box area: {box}")
    return value


def intersection_area(a: list[float], b: list[float]) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )


def policy_result(predictions: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    if policy == "original":
        return {
            "original": predictions,
            "retained": predictions,
            "deleted_indices": [],
            "deletion_child_indices": {},
        }

    boxes = [pred["xyxy"] for pred in predictions]
    classes = [pred["cls"] for pred in predictions]
    areas = [area(box) for box in boxes]
    causes: dict[int, int] = {}

    # Every decision uses the complete, untouched original prediction list.
    for outer in range(len(predictions)):
        for child in range(len(predictions)):
            if outer == child or classes[outer] != classes[child]:
                continue
            if not areas[child] < areas[outer]:
                continue
            if areas[child] < CHILD_AREA_MIN:
                fail(f"Child box area below {CHILD_AREA_MIN}: {boxes[child]}")
            covered = intersection_area(boxes[outer], boxes[child]) / areas[child]
            if covered < CHILD_COVERAGE:
                continue
            if areas[outer] / areas[child] < RATIOS[policy]:
                continue
            causes.setdefault(outer, child)

    return {
        "original": predictions,
        "retained": [pred for index, pred in enumerate(predictions) if index not in causes],
        "deleted_indices": sorted(causes),
        "deletion_child_indices": {str(k): v for k, v in sorted(causes.items())},
    }


def add_metric(total: dict[str, int], metric: dict[str, int]) -> None:
    for key in ("tp", "fp", "fn"):
        if key not in metric:
            fail(f"Metric missing field {key}")
        total[key] += int(metric[key])


def require_int(parent: dict[str, Any], key: str, context: str) -> int:
    if key not in parent:
        fail(f"{context} lacks {key}")
    try:
        return int(parent[key])
    except (TypeError, ValueError):
        fail(f"{context}.{key} is not an integer")


def check_expected(
    actual: dict[str, int], expected: dict[str, Any], keys: tuple[str, ...], context: str
) -> None:
    for key in keys:
        old_value = require_int(expected, key, context)
        if actual[key] != old_value:
            fail(f"{context}.{key} mismatch: recomputed={actual[key]}, old={old_value}")


def unique_rows(rows: list[Any], key: str, expected_count: int, context: str) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != expected_count:
        fail(f"{context} must contain {expected_count} rows")
    output: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or key not in row:
            fail(f"{context}[{index}] lacks {key}")
        name = row[key]
        if not isinstance(name, str) or name in output:
            fail(f"{context} has invalid or duplicate {key}: {name!r}")
        output[name] = row
    return output


def score(gt: list[list[float]], predictions: list[dict[str, Any]]) -> dict[str, int]:
    return matches(gt, [pred["xyxy"] for pred in predictions], MATCH_IOU)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--videos", required=True, type=Path)
    parser.add_argument("--blue-truth", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    if args.out.exists():
        fail(f"Refusing existing output directory: {args.out}")
    if not args.comparison.is_dir() or not args.videos.is_dir():
        fail("Comparison and videos must be existing directories")

    read_files: dict[str, str] = {}
    legacy_dir = args.comparison / "legacy_sources"
    visible_dir = args.comparison / "visible_flame"

    local_rows = load_json(legacy_dir / "local_inputs.json", read_files)
    legacy_summary = load_json(legacy_dir / "summary.json", read_files)
    old_visible = load_json(visible_dir / "summary.json", read_files)
    comparison_complete = load_json(args.comparison / "comparison_complete.json", read_files)
    blue_truth = load_json(args.blue_truth, read_files)
    video_plan = load_json(args.videos / "plan.json", read_files)
    video_rows = load_jsonl(args.videos / "per_frame.jsonl", read_files)

    local_by_file = unique_rows(local_rows, "file", 286, "local_inputs.json")
    image_details: list[dict[str, Any]] = []
    legacy_original: dict[str, Any] = {}

    for model in MODELS:
        path = legacy_dir / f"{model}_local_predictions.json"
        rows = load_json(path, read_files)
        predictions_by_file = unique_rows(rows, "image", 286, str(path))
        if set(predictions_by_file) != set(local_by_file):
            fail(f"{model}: prediction filenames do not exactly match local inputs")

        grouped: dict[str, list[tuple[list[list[float]], list[dict[str, Any]]]]] = {}
        for filename, row in predictions_by_file.items():
            provenance = row.get("provenance")
            if provenance is None:
                fail(f"{model}/{filename}: missing provenance")
            gt = boxes_from(local_by_file[filename].get("gt"), f"local_inputs[{filename}].gt")
            predictions = normalize_predictions(
                row.get("predictions"), f"{model}/{filename}.predictions"
            )
            grouped.setdefault(provenance, []).append((gt, predictions))
            for policy in POLICIES:
                result = policy_result(predictions, policy)
                image_details.append(
                    {
                        "group": "legacy_local",
                        "model": model,
                        "image": filename,
                        "provenance": provenance,
                        "policy": policy,
                        "gt": gt,
                        **result,
                        "metrics": score(gt, result["retained"]),
                    }
                )

        old_model = legacy_summary.get("models", {}).get(model, {}).get("by_provenance", {})
        legacy_original[model] = {}
        for provenance, entries in grouped.items():
            total = {"tp": 0, "fp": 0, "fn": 0, "images_with_fp": 0}
            for gt, predictions in entries:
                metric = score(gt, predictions)
                add_metric(total, metric)
                total["images_with_fp"] += int(metric["fp"] > 0)
            if provenance not in old_model:
                fail(f"Old legacy summary lacks {model}/{provenance}")
            check_expected(
                total,
                old_model[provenance],
                ("tp", "fp", "fn", "images_with_fp"),
                f"legacy original check {model}/{provenance}",
            )
            legacy_original[model][provenance] = total

    visible_summary: dict[str, Any] = {}
    for model in MODELS:
        rows = load_json(visible_dir / f"{model}_details.json", read_files)
        if not isinstance(rows, list) or len(rows) != 10:
            fail(f"{model} visible details must contain 10 scored images")
        visible_summary[model] = {
            policy: {
                "tp": 0, "fp": 0, "fn": 0, "images_with_fp": 0, "deleted_boxes": 0
            }
            for policy in POLICIES
        }
        for index, row in enumerate(rows):
            gt = boxes_from(
                row.get("visible_flame_boxes"),
                f"{model}/visible[{index}].visible_flame_boxes",
            )
            predictions = normalize_predictions(
                row.get("predictions"),
                f"{model}/visible[{index}].predictions",
            )
            for policy in POLICIES:
                result = policy_result(predictions, policy)
                metric = score(gt, result["retained"])
                target = visible_summary[model][policy]
                add_metric(target, metric)
                target["images_with_fp"] += int(metric["fp"] > 0)
                target["deleted_boxes"] += len(result["deleted_indices"])
                image_details.append(
                    {
                        "group": "visible_flame",
                        "model": model,
                        "image": row.get("image"),
                        "index": row.get("index", index),
                        "policy": policy,
                        "gt": gt,
                        **result,
                        "metrics": metric,
                    }
                )

        old = (
            old_visible.get("models", {})
            .get(model, {})
            .get("supplemental", {})
            .get("0.5", {})
        )
        if not old:
            fail(f"Old visible summary lacks supplemental 0.5 for {model}")
        check_expected(
            visible_summary[model]["original"],
            old,
            ("tp", "fp", "fn"),
            f"visible original check {model}/supplemental[0.5]",
        )

    if not isinstance(blue_truth, list) or len(blue_truth) != 5:
        fail("blue-truth JSON must be a five-row list")
    truth: dict[str, list[float]] = {}
    for index, row in enumerate(blue_truth):
        if not isinstance(row, dict) or not isinstance(row.get("slug"), str):
            fail(f"blue-truth[{index}] lacks slug")
        truth[row["slug"]] = xyxy_from(
            row.get("primary_flame_xyxy"),
            f"blue-truth[{index}].primary_flame_xyxy",
        )

    blue_summary: dict[str, Any] = {}
    for model in MODELS:
        blue = comparison_complete.get("models", {}).get(model, {}).get("blue", {})
        per_image = blue.get("per_image")
        if not isinstance(per_image, list) or len(per_image) != 5:
            fail(f"comparison_complete blue.per_image invalid for {model}")
        blue_summary[model] = {
            policy: {"hits": 0, "per_image": []} for policy in POLICIES
        }
        seen_slugs = set()
        for index, row in enumerate(per_image):
            slug = row.get("slug")
            if slug not in truth or slug in seen_slugs:
                fail(f"{model}: invalid or duplicate blue slug {slug!r}")
            seen_slugs.add(slug)
            predictions = normalize_predictions(
                row.get("predictions"), f"blue[{model}][{index}].predictions"
            )
            for policy in POLICIES:
                retained = policy_result(predictions, policy)["retained"]
                best_iou = max(
                    (iou(truth[slug], pred["xyxy"]) for pred in retained), default=0.0
                )
                hit = best_iou >= MATCH_IOU
                blue_summary[model][policy]["hits"] += int(hit)
                blue_summary[model][policy]["per_image"].append(
                    {"slug": slug, "best_iou": best_iou, "iou50_hit": hit}
                )
        old_hits = require_int(blue, "primary_localized_iou50", f"blue original check {model}")
        if blue_summary[model]["original"]["hits"] != old_hits:
            fail(
                f"blue original check {model}: recomputed="
                f"{blue_summary[model]['original']['hits']}, old={old_hits}"
            )

    if len(video_rows) != 307:
        fail(f"per_frame.jsonl must contain 307 shared samples, got {len(video_rows)}")
    video_summary: dict[str, Any] = {}
    video_details: list[dict[str, Any]] = []
    for model in MODELS:
        video_summary[model] = {
            policy: {
                "frames": 0,
                "frames_with_deletion": 0,
                "original_boxes": 0,
                "retained_boxes": 0,
                "deleted_boxes": 0,
                "by_video": {},
            }
            for policy in POLICIES
        }
        for index, row in enumerate(video_rows):
            predictions_by_model = row.get("predictions")
            if not isinstance(predictions_by_model, dict):
                fail(f"video row {index}: predictions must be an object")
            predictions = normalize_predictions(
                predictions_by_model.get(model), f"video[{index}]/{model}"
            )
            for policy in POLICIES:
                result = policy_result(predictions, policy)
                target = video_summary[model][policy]
                deleted = len(result["deleted_indices"])
                target["frames"] += 1
                target["frames_with_deletion"] += int(deleted > 0)
                target["original_boxes"] += len(predictions)
                target["retained_boxes"] += len(result["retained"])
                target["deleted_boxes"] += deleted
                by_video = target["by_video"].setdefault(
                    str(row.get("video_index")),
                    {"frames": 0, "frames_with_deletion": 0, "deleted_boxes": 0},
                )
                by_video["frames"] += 1
                by_video["frames_with_deletion"] += int(deleted > 0)
                by_video["deleted_boxes"] += deleted
                video_details.append(
                    {
                        "video_index": row.get("video_index"),
                        "frame_index": row.get("frame_index"),
                        "sample_index": row.get("sample_index"),
                        "model": model,
                        "policy": policy,
                        **result,
                    }
                )

    # All inputs and original-cache rechecks succeeded before creating output.
    args.out.mkdir(parents=True, exist_ok=False)
    plan = {
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "models": list(MODELS),
        "policies": list(POLICIES),
        "fixed_parameters": {
            "confidence": CONFIDENCE,
            "nms_iou": NMS_IOU,
            "match_iou": MATCH_IOU,
            "child_coverage": CHILD_COVERAGE,
            "child_area_min": CHILD_AREA_MIN,
            "ratios": {key: value for key, value in RATIOS.items() if value is not None},
            "decision_source": "original prediction list",
            "same_class_only": True,
            "strict_smaller_child_area": True,
            "equal_area_boxes_do_not_delete_each_other": True,
        },
        "input_sha256": read_files,
        "video_plan": {"path": str(args.videos / "plan.json"), "sha256": sha256(args.videos / "plan.json")},
        "roles_and_boundaries": {
            "legacy_local": "Exposure development diagnostic; original cache metrics rechecked by provenance.",
            "visible_flame": "Approved approximate physical flame boxes; original supplemental IoU-0.5 metrics rechecked.",
            "blue": "Primary target best-IoU and hit count only; no all-FP inference.",
            "new_training_sources": "Not evaluated because no supplied GT is used or fabricated.",
            "videos": "Deletion geometry and counts only; not recognition accuracy or false-positive rates.",
            "semantic_warning": "Geometric nesting does not prove the outer box is semantically wrong; legitimate whole-flame boxes may be deleted.",
            "status": "Development-exposed diagnostic comparison; no new mAP, independent test, or deployment conclusion.",
        },
    }
    summary = {
        "models": list(MODELS),
        "policies": list(POLICIES),
        "legacy_original_recheck": legacy_original,
        "visible_flame": visible_summary,
        "blue": blue_summary,
        "videos": video_summary,
        "new_training_sources": {
            "evaluated": False,
            "reason": "No GT supplied; no metrics fabricated.",
        },
        "interpretation_boundary": "Nested geometry is not proof of semantic error; diagnostics only.",
    }

    write_json(args.out / "comparison_plan.json", plan)
    write_json(args.out / "summary.json", summary)
    write_json(args.out / "image_details.json", image_details)
    with (args.out / "video_details.jsonl").open("w", encoding="utf-8") as stream:
        for row in video_details:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [
        "# Nested fire-box development comparison",
        "",
        "Development-exposed cache diagnostic only. Nested geometry is not proof of semantic error.",
        "",
        "## Visible-flame metrics",
        "",
        "| model | policy | TP | FP | FN | images_with_fp | deleted_boxes |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for policy in POLICIES:
            row = visible_summary[model][policy]
            lines.append(
                f"| {model} | {policy} | {row['tp']} | {row['fp']} | {row['fn']} | "
                f"{row['images_with_fp']} | {row['deleted_boxes']} |"
            )
    lines.extend(
        [
            "",
            "## Blue primary-target IoU-0.5 hits",
            "",
            "| model | original | nested_r2 | nested_r3 | nested_r4 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model in MODELS:
        lines.append(
            "| "
            + model
            + " | "
            + " | ".join(str(blue_summary[model][policy]["hits"]) for policy in POLICIES)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Video geometry counts",
            "",
            "| model | policy | frames_with_deletion | deleted_boxes |",
            "|---|---|---:|---:|",
        ]
    )
    for model in MODELS:
        for policy in POLICIES:
            row = video_summary[model][policy]
            lines.append(
                f"| {model} | {policy} | {row['frames_with_deletion']} | {row['deleted_boxes']} |"
            )
    lines.extend(
        [
            "",
            "Video box counts are geometry diagnostics, not accuracy or false-positive rates.",
        ]
    )
    (args.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CheckFailure as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
