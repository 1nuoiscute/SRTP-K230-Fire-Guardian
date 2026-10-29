"""Structural checks for future blind fire-event annotation manifests.

A successful check does not prove source independence, provenance truth,
annotation truth, blind handling, or evaluation readiness.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Mapping
from typing import Any

_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_USAGE = {"development", "candidate_independent_test", "quarantine"}
_PHYSICAL = {"on", "off", "unknown"}
_VISIBILITY = {"flame_visible", "clear_no_flame", "occluded", "unknown"}
_ROI_RELATION = {"inside", "outside", "partial", "indeterminate"}


class ManifestValidationError(ValueError):
    """A structural constraint in the manifest was violated."""


def _fail(where: str, message: str) -> None:
    raise ManifestValidationError(f"{where}: {message}")


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _rect(value: Any, where: str, width: int, height: int) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4 or not all(_finite(v) for v in value):
        _fail(where, "must contain four finite xyxy coordinates")
    x1, y1, x2, y2 = map(float, value)
    if x2 <= x1 or y2 <= y1:
        _fail(where, "must have positive area")
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        _fail(where, "must lie within original image bounds")
    return x1, y1, x2, y2


def _computed_relation(box: tuple[float, float, float, float], roi: tuple[float, float, float, float]) -> str:
    x1, y1, x2, y2 = box
    a1, b1, a2, b2 = roi
    if x1 >= a1 and y1 >= b1 and x2 <= a2 and y2 <= b2:
        return "inside"
    if x2 <= a1 or x1 >= a2 or y2 <= b1 or y1 >= b2:
        return "outside"
    return "partial"


def validate_manifest(manifest: Mapping[str, Any]) -> list[str]:
    """Validate structural constraints and return review warnings.

    Schema v1 uses event ``intervals`` only for continuous physical/visibility
    states. Detection boxes belong only to ``samples``, each tied to an explicit
    original ``frame_number`` and ``timestamp_s``. Uncovered time is warned.
    """
    if not isinstance(manifest, Mapping):
        _fail("manifest", "must be an object")
    if manifest.get("schema_version") != 1:
        _fail("manifest.schema_version", "must equal 1")
    events = manifest.get("events")
    if not isinstance(events, list) or not events:
        _fail("manifest.events", "must be a non-empty list")

    event_ids: set[str] = set()
    group_usage: dict[str, str] = {}
    digest_identity: dict[str, tuple[str, str]] = {}
    warnings: list[str] = []

    for ei, event in enumerate(events):
        at = f"events[{ei}]"
        if not isinstance(event, Mapping):
            _fail(at, "must be an object")
        for key in ("event_id", "source_group_id"):
            if not _nonempty(event.get(key)):
                _fail(f"{at}.{key}", "must be a non-empty string")
        event_id, group_id = event["event_id"], event["source_group_id"]
        if event_id in event_ids:
            _fail(f"{at}.event_id", "must be unique")
        event_ids.add(event_id)

        usage = event.get("usage")
        if not isinstance(usage, str) or usage not in _USAGE:
            _fail(f"{at}.usage", f"must be one of {sorted(_USAGE)}")
        source = event.get("source")
        if not isinstance(source, Mapping):
            _fail(f"{at}.source", "must be an object")
        if not _nonempty(source.get("path")):
            _fail(f"{at}.source.path", "must be a non-empty string")
        digest = source.get("sha256")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            _fail(f"{at}.source.sha256", "must be 64 hexadecimal characters")
        digest = digest.lower()
        prior_identity = digest_identity.get(digest)
        if prior_identity is not None and prior_identity != (group_id, usage):
            _fail(f"{at}.source.sha256", "same original SHA-256 cannot belong to different source groups or uses")
        if prior_identity is not None:
            warnings.append(f"{event_id}: original SHA-256 also appears in another event within the same group/use; possible duplicate event counting requires review")
        digest_identity[digest] = (group_id, usage)
        if group_id in group_usage and group_usage[group_id] != usage:
            _fail(f"{at}.usage", "events in one source group must share frozen usage")
        group_usage[group_id] = usage
        relation = source.get("relation")
        if not isinstance(relation, str) or relation not in {"screened", "unknown"}:
            _fail(f"{at}.source.relation", "must be screened or unknown")
        if relation == "unknown":
            warnings.append(f"{event_id}: source relation unknown; independence is unestablished and event is not acceptance-ready")

        duration = event.get("duration_s")
        if not _finite(duration) or duration <= 0:
            _fail(f"{at}.duration_s", "must be finite and positive")
        image = event.get("image")
        if not isinstance(image, Mapping):
            _fail(f"{at}.image", "must be an object")
        width, height = image.get("width"), image.get("height")
        if any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in (width, height)):
            _fail(f"{at}.image", "width and height must be positive integers")
        roi_data = event.get("roi")
        if not isinstance(roi_data, Mapping) or not _nonempty(roi_data.get("version")):
            _fail(f"{at}.roi", "must include a non-empty calibration version")
        roi = _rect(roi_data.get("xyxy"), f"{at}.roi.xyxy", width, height)

        intervals = event.get("intervals")
        if not isinstance(intervals, list):
            _fail(f"{at}.intervals", "must be a list; use intervals for states and samples for frame boxes")
        covered: list[tuple[float, float]] = []
        for ii, interval in enumerate(intervals):
            it = f"{at}.intervals[{ii}]"
            if not isinstance(interval, Mapping):
                _fail(it, "must be an object")
            start, end = interval.get("start_s"), interval.get("end_s")
            if not _finite(start) or not _finite(end) or start < 0 or end <= start or end > duration:
                _fail(it, "must satisfy 0 <= start_s < end_s <= duration_s")
            physical, visibility = interval.get("physical_state"), interval.get("visibility")
            if not isinstance(physical, str) or physical not in _PHYSICAL:
                _fail(f"{it}.physical_state", f"must be one of {sorted(_PHYSICAL)}")
            if not isinstance(visibility, str) or visibility not in _VISIBILITY:
                _fail(f"{it}.visibility", f"must be one of {sorted(_VISIBILITY)}")
            if "boxes" in interval:
                _fail(f"{it}.boxes", "interval boxes imply frame truth; put boxes in timestamped samples")
            negative = interval.get("negative_example", False)
            if not isinstance(negative, bool):
                _fail(f"{it}.negative_example", "must be a boolean")
            if negative and (physical != "off" or visibility != "clear_no_flame"):
                _fail(f"{it}.negative_example", "negative examples require physical_state=off and visibility=clear_no_flame")
            if negative:
                for evidence_key in ("physical_evidence", "visual_evidence"):
                    if not _nonempty(interval.get(evidence_key)):
                        _fail(f"{it}.{evidence_key}", "negative examples require a non-empty evidence reference")
            elif physical == "off" and visibility == "clear_no_flame":
                warnings.append(f"{event_id}: interval {ii} is off/clear but is not explicitly marked as a reviewed negative")
            if physical == "unknown":
                warnings.append(f"{event_id}: interval {ii} has unknown physical state; do not treat as negative")
            covered.append((float(start), float(end)))
        covered.sort()
        cursor = 0.0
        for start, end in covered:
            if start < cursor:
                _fail(f"{at}.intervals", "time intervals must not overlap")
            if start > cursor:
                warnings.append(f"{event_id}: uncovered interval [{cursor:g}, {start:g}) s; timeline is incomplete")
            cursor = end
        if cursor < duration:
            warnings.append(f"{event_id}: uncovered interval [{cursor:g}, {float(duration):g}) s; timeline is incomplete")
        if not intervals:
            warnings.append(f"{event_id}: no state intervals supplied; timeline is incomplete")

        samples = event.get("samples")
        if not isinstance(samples, list):
            _fail(f"{at}.samples", "must be a list of explicit sampled-frame annotations")
        if not samples:
            warnings.append(f"{event_id}: samples is empty; there are no sampled-frame labels to review")
        seen_frames: set[int] = set()
        seen_times: set[float] = set()
        for si, sample in enumerate(samples):
            st = f"{at}.samples[{si}]"
            if not isinstance(sample, Mapping):
                _fail(st, "must be an object")
            frame = sample.get("frame_number")
            timestamp = sample.get("timestamp_s")
            if not isinstance(frame, int) or isinstance(frame, bool) or frame < 0:
                _fail(f"{st}.frame_number", "must be a non-negative original frame number")
            if not _finite(timestamp) or timestamp < 0 or timestamp > duration:
                _fail(f"{st}.timestamp_s", "must be a finite time within the event")
            if frame in seen_frames or float(timestamp) in seen_times:
                _fail(st, "sample frame numbers and timestamps must be unique within an event")
            seen_frames.add(frame)
            seen_times.add(float(timestamp))
            physical, visibility = sample.get("physical_state"), sample.get("visibility")
            if not isinstance(physical, str) or physical not in _PHYSICAL:
                _fail(f"{st}.physical_state", f"must be one of {sorted(_PHYSICAL)}")
            if not isinstance(visibility, str) or visibility not in _VISIBILITY:
                _fail(f"{st}.visibility", f"must be one of {sorted(_VISIBILITY)}")
            negative = sample.get("negative_example", False)
            if not isinstance(negative, bool):
                _fail(f"{st}.negative_example", "must be a boolean")
            if negative and (physical != "off" or visibility != "clear_no_flame"):
                _fail(f"{st}.negative_example", "negative examples require physical_state=off and visibility=clear_no_flame")
            if negative:
                for evidence_key in ("physical_evidence", "visual_evidence"):
                    if not _nonempty(sample.get(evidence_key)):
                        _fail(f"{st}.{evidence_key}", "negative examples require a non-empty evidence reference")
            elif physical == "off" and visibility == "clear_no_flame":
                warnings.append(f"{event_id}: sampled frame {frame} is off/clear but is not explicitly marked as a reviewed negative")
            if physical == "unknown":
                warnings.append(f"{event_id}: sampled frame {frame} has unknown physical state; do not treat as negative")
            matching_intervals = [
                interval for interval in intervals
                if interval["start_s"] <= timestamp < interval["end_s"]
            ]
            if not matching_intervals:
                warnings.append(f"{event_id}: sampled frame {frame} at {float(timestamp):g} s has no covering state interval")
            else:
                state_interval = matching_intervals[0]
                interval_physical = state_interval["physical_state"]
                interval_visibility = state_interval["visibility"]
                if physical != "unknown" and interval_physical != "unknown" and physical != interval_physical:
                    warnings.append(f"{event_id}: sampled frame {frame} physical_state={physical} conflicts with its covering interval state={interval_physical}")
                if visibility != "unknown" and interval_visibility != "unknown" and visibility != interval_visibility:
                    partial_flame_occlusion = {visibility, interval_visibility} == {"flame_visible", "occluded"}
                    if not partial_flame_occlusion:
                        warnings.append(f"{event_id}: sampled frame {frame} visibility={visibility} conflicts with its covering interval visibility={interval_visibility}")
            if physical == "off" and visibility == "flame_visible":
                warnings.append(f"{event_id}: sampled frame {frame} records physical_state=off with visible flame; review transition timing and evidence")
            boxes = sample.get("boxes")
            if not isinstance(boxes, list):
                _fail(f"{st}.boxes", "must be a list")
            partial_visible = sample.get("partially_visible_flame", False)
            if not isinstance(partial_visible, bool):
                _fail(f"{st}.partially_visible_flame", "must be a boolean")
            if visibility == "flame_visible" and not boxes:
                _fail(f"{st}.boxes", "visible flame sample requires at least one localized box")
            if visibility == "occluded" and boxes and not partial_visible:
                _fail(f"{st}.partially_visible_flame", "occluded samples with boxes must explicitly mark partially_visible_flame=true")
            if visibility not in {"flame_visible", "occluded"} and boxes:
                _fail(f"{st}.boxes", "boxes require visible flame; for occluded samples set partially_visible_flame=true")
            if partial_visible and visibility != "occluded":
                _fail(f"{st}.partially_visible_flame", "is only meaningful with visibility=occluded")
            for bi, box in enumerate(boxes):
                bt = f"{st}.boxes[{bi}]"
                if not isinstance(box, Mapping):
                    _fail(bt, "must be an object")
                coords = _rect(box.get("xyxy"), f"{bt}.xyxy", width, height)
                box_relation = box.get("roi_relation")
                if not isinstance(box_relation, str) or box_relation not in _ROI_RELATION:
                    _fail(f"{bt}.roi_relation", f"must be one of {sorted(_ROI_RELATION)}")
                computed = _computed_relation(coords, roi)
                if box_relation != "indeterminate" and box_relation != computed:
                    _fail(f"{bt}.roi_relation", f"does not match box/ROI geometry (expected {computed})")

    return warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Structurally validate a fire-event JSON manifest. This cannot prove source independence, provenance or label truth."
    )
    parser.add_argument("manifest_json", help="path to UTF-8 JSON manifest, or - to read stdin")
    args = parser.parse_args(argv)
    try:
        if args.manifest_json == "-":
            manifest = json.load(sys.stdin)
        else:
            with open(args.manifest_json, "r", encoding="utf-8") as stream:
                manifest = json.load(stream)
        warnings = validate_manifest(manifest)
        report = {
            "status": "structurally_valid_with_warnings" if warnings else "structurally_valid",
            "warnings": warnings,
            "acceptance_ready": False,
            "limitations": [
                "Structural validation does not prove source independence.",
                "Structural validation does not prove provenance, blind handling, or annotation truth.",
                "Evidence references being present does not prove evidence is authentic, sufficient, or complete.",
                "Sampled-frame labels do not establish truth for unsampled frames.",
                "Repeated boxes across frames may be genuine; this validator does not detect copied or duplicated annotations.",
            ],
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (ManifestValidationError, json.JSONDecodeError, OSError) as exc:
        print(json.dumps({"status": "invalid", "error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
