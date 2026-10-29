"""Filter saved fixed-ROI prediction boxes at preset confidence cutoffs.

Reads only the pinned historical JSON caches. It does not load images, videos,
weights, or inference libraries and prints results to stdout without writing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

MODELS = ("v16_best", "v18_last", "v24_last")
THRESHOLDS = (0.25, 0.30, 0.40, 0.50)
VIDEO_FRAMES = (560, *range(708, 720))
VIDEO_PLAN_SHA = "881d2bbcadced211a13c45d4f92893366c6ffef64b35108fcd56a49a73fb4707"
VIDEO_DONE_SHA = "2019422ace2a809312722bd086e49bca731ff7d4c8d7de6f2d05fcb1465bd222"
VIDEO_ROWS_SHA = "77a039fd52301707f92aaed292ac7530fb2afff8fc3068b282e7312308e3bc48"
IMAGE_PLAN_SHA = "aca91faeec2a17a0f92d139ee7c475da02c93f7d7e0b8b47ac8796e5161e5985"
IMAGE_DONE_SHA = "5ada1e3791dd894fe447049fc29fbae0c2eac2e7feff5e2d24d083df2ef23679"
IMAGE_ROWS_SHA = "9c2e1ac8763c7037269aa544dfdcd56888393b07d2cbe47607550ee49720d7d3"
VIDEO_PIXEL_SHA = {
    560: "5d6ffcd2841686467891cd018f71a1da761ccc93b13af564752609907dbbbbd9",
    708: "1fad4ca13aeb20952febd6556afde27146faa595562765a458bb603183d2c400",
    709: "4de1b17ab432eb6b14fc25b703275c3200cfda88e5659b06532368f563036990",
    710: "51c0e7ea3e4fdf8154259038015a3c64a4dd6159437cca697454d1a702fe26ab",
    711: "bc102857bc07dda3c8d106b4ba84ade47f8d89bcff9b4b098aa2acf46fcb9b4e",
    712: "12559f255170270c7eec1e2d3c1a8606cef3ce5ba93cee906f7f02c9fccdc2e0",
    713: "8e3a1bc35c31f66a493b62e24eb6f8100cad281123c97ff997e95af5c5984b78",
    714: "7ebd554b95a7113a28e1c0cbb6fd85ec3f6c0380834696a885682a7e00cd59c6",
    715: "6054a9ed894c51eb945580a928e978cbf00740db4cfd72b60aa84e376a3326e9",
    716: "73d33b402f98ccb68fbb5d01fa76695ba209495214056981c14d0432ba01d3a5",
    717: "74ce6f9422ee3bee52792e95620dcc4e998d5793f44634e36b1beaeb5773b14b",
    718: "8118f40403e15c6b53cce39bc29cd996b5565541cd37532baf627f50697da5a4",
    719: "4580768ef9a2ed592ec5dc22c0bf7d664176a5c86ff53e2f6fc9b6f329753ab7",
}
IMAGE_IDS = {
    "ks_nofire_confirmed__PXL_20240915_091533721_0021_png.rf.da6f8f9853289045678c1b6933eb5356.jpg": "bb5cc902cc97d3766d9985fc507ae1b7a6518208b52942a9f03df5c190c081f0",
    "nofire_real_indoor__val_55.jpg": "c75c76cc1f75cdf1f88672432b9ce9653b739e53b457f8140ee03e86d6ca93c7",
    "nofire_real_indoor__val_1025.jpg": "07a4d0bbd8b00222d987a1f45e057d11efd3f57c9441e6283437b13b380ee88a",
}
EMPTY_LABEL_SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
MODEL_SHA = {
    "v16_best": "e770cf68d12cd43b4494a1439dc9c65f14d1d5fc570b2e2e379cc084384418fd",
    "v18_last": "2534a8c727157d0333bf7e0b785cd09e950f5e6e0192831947ba26cfe865b3ce",
    "v24_last": "d7d0c0e1823e48dbcc5b57c29bc418500e12cf6639ef82d84ece2804b1f2f68d",
}
VIDEO7_SHA = "f768e372ec81704a31e5bad7834b523b93be05a9b9c88f7040f2178261fe99fc"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(data: bytes, label: str) -> Any:
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON in {label}: {exc}") from exc


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_bound_cache(directory: Path, kind: str) -> tuple[dict, dict, list[dict], dict]:
    """Verify pinned file hashes, completion bindings, source plan, and JSONL."""
    if kind == "video":
        names = ("plan.json", "comparison_complete.json", "per_frame.jsonl")
        expected = (VIDEO_PLAN_SHA, VIDEO_DONE_SHA, VIDEO_ROWS_SHA)
    else:
        names = ("plan.json", "comparison_complete.json", "per_image.jsonl")
        expected = (IMAGE_PLAN_SHA, IMAGE_DONE_SHA, IMAGE_ROWS_SHA)
    blobs = [(directory / name).read_bytes() for name in names]
    actual = tuple(sha256(blob) for blob in blobs)
    _assert(actual == expected, f"{kind} cache identity SHA-256 mismatch: {actual}")
    plan = read_json(blobs[0], names[0])
    done = read_json(blobs[1], names[1])
    _assert(done.get("plan_sha256") == actual[0], f"{kind} completion does not bind plan.json")
    # Historical video completion uses a generated key; identify its one rows hash
    # by the pinned value, while still requiring exactly one matching digest field.
    row_hash_values = [value for key, value in done.items() if key != "plan_sha256" and isinstance(value, str) and len(value) == 64]
    _assert(row_hash_values == [actual[2]], f"{kind} completion does not bind prediction JSONL")
    if kind == "image":
        _assert(done.get("per_image_sha256") == actual[2], "image completion per_image_sha256 mismatch")
        _assert(plan.get("image_count") == 215 and plan.get("reserved_sources_used") is False,
                "unexpected historical image plan scope")
        _assert(len(plan.get("inputs_sha256", "")) == 64 and
                set(plan.get("prediction_sha256", {})) == set(MODELS),
                "image plan input/prediction SHA bindings are incomplete")
    else:
        _assert(done.get("frames") == 1014 and done.get("ground_truth_provided") is False,
                "unexpected historical video completion scope")
        _assert(len(plan.get("videos", [])) == 9 and plan.get("reserved_sources_used") is False,
                "unexpected historical video plan scope")
        _assert(plan["videos"][6].get("sha256") == VIDEO7_SHA,
                "video7 source SHA binding differs from the selected cache source")
    model_entries = {entry.get("label"): entry for entry in plan.get("models", []) if isinstance(entry, dict)}
    _assert(set(model_entries) == set(MODELS) and
            all(model_entries[name].get("expected_sha256") == digest for name, digest in MODEL_SHA.items()),
            f"{kind} plan model identities differ")
    rows = [read_json(line.encode("utf-8"), f"{names[2]}:{n}")
            for n, line in enumerate(blobs[2].decode("utf-8").splitlines(), 1) if line.strip()]
    _assert(len(rows) == (1014 if kind == "video" else 215), f"unexpected {kind} JSONL row count")
    return plan, done, rows, dict(zip(names, actual))


def validate_conditional(prediction: dict, label: str) -> str:
    """Return the cache state while preserving the meaning of empty lists."""
    full = prediction.get("full_frame")
    attempted = prediction.get("crop_attempted")
    crops = prediction.get("crop_predictions")
    _assert(isinstance(full, list) and isinstance(crops, list), f"{label}: box fields must be lists")
    _assert(type(attempted) is bool, f"{label}: crop_attempted must be boolean")
    _assert(attempted == (len(full) == 0), f"{label}: crop_attempted conflicts with full-frame boxes")
    for i, box in enumerate((*full, *crops)):
        score = box.get("confidence") if isinstance(box, dict) else None
        _assert(isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score)
                and 0 <= score <= 1, f"{label}: invalid confidence in box {i}")
    if full:
        _assert(not crops, f"{label}: crop boxes exist despite full-frame detection")
        return "full_frame_existing_crop_not_triggered"
    if not attempted:
        return "not_triggered"
    return "crop_attempted_with_boxes" if crops else "crop_attempted_no_boxes"


def filter_saved_boxes(boxes: list[dict], threshold: float) -> list[dict]:
    """Keep saved boxes at or above threshold; equality is retained."""
    return [box for box in boxes if float(box["confidence"]) >= threshold]


def selected_records(video_rows: list[dict], image_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    _assert(len({(r.get("video_index"), r.get("frame_index")) for r in video_rows}) == len(video_rows),
            "duplicate video frame identity in cache")
    _assert(len({r.get("file") for r in image_rows}) == len(image_rows),
            "duplicate image identity in cache")
    vmap = {(r.get("video_index"), r.get("frame_index")): r for r in video_rows}
    selected_video = []
    for frame in VIDEO_FRAMES:
        row = vmap.get((7, frame))
        _assert(row is not None, f"missing selected video7 frame {frame}")
        _assert(row.get("source_pixel_sha256") == VIDEO_PIXEL_SHA[frame], f"video7 frame {frame} pixel identity mismatch")
        selected_video.append(row)
    imap = {r.get("file"): r for r in image_rows}
    selected_images = []
    for filename, digest in IMAGE_IDS.items():
        row = imap.get(filename)
        _assert(row is not None, f"missing selected image {filename}")
        _assert(row.get("image_sha256") == digest, f"selected image identity mismatch: {filename}")
        _assert(row.get("label_sha256") == EMPTY_LABEL_SHA, f"selected image label hash mismatch: {filename}")
        expected_group = "ks_nofire_confirmed" if filename.startswith("ks_nofire") else "nofire_real_indoor"
        _assert(row.get("provenance") == expected_group, f"selected image provenance mismatch: {filename}")
        selected_images.append(row)
    return selected_video, selected_images


def summarize_entity(row: dict, kind: str, key: str) -> dict:
    result = {key: row["frame_index"] if kind == "video" else row["file"]}
    if kind == "image":
        result["provenance"] = row["provenance"]
    result["models"] = {}
    for model in MODELS:
        pred = row["predictions"][model]
        state = validate_conditional(pred, f"{key}={result[key]} {model}")
        kept = {str(t): len(filter_saved_boxes(pred["crop_predictions"], t)) for t in THRESHOLDS}
        result["models"][model] = {
            "state": state,
            "full_frame_box_count": len(pred["full_frame"]),
            "crop_saved_box_count": len(pred["crop_predictions"]),
            "crop_saved_scores": [box["confidence"] for box in pred["crop_predictions"]],
            "retained_crop_boxes_by_threshold": kept,
            "has_retained_crop_box_by_threshold": {t: n > 0 for t, n in kept.items()},
        }
    return result


def analyze(video_dir: Path, image_dir: Path) -> dict:
    vp, vd, vr, vh = verify_bound_cache(video_dir, "video")
    ip, idone, ir, ih = verify_bound_cache(image_dir, "image")
    vrows, irows = selected_records(vr, ir)
    summaries = [summarize_entity(r, "video", "frame_index") for r in vrows]
    summaries += [summarize_entity(r, "image", "file") for r in irows]
    aggregate = {}
    for model in MODELS:
        aggregate[model] = {}
        for threshold in THRESHOLDS:
            t = str(threshold)
            frame_rows = [x for x in summaries if "frame_index" in x]
            image_rows = [x for x in summaries if "file" in x]
            def totals(rows: list[dict]) -> dict:
                values = [x["models"][model]["retained_crop_boxes_by_threshold"][t] for x in rows]
                return {"entities": len(rows), "boxes": sum(values), "entities_with_box": sum(v > 0 for v in values)}
            aggregate[model][t] = {"video_frames": totals(frame_rows), "images": totals(image_rows)}
    return {
        "scope": "saved ROI crop_predictions only; no inference",
        "threshold_rule": "confidence >= threshold",
        "thresholds": list(THRESHOLDS),
        "video_frames": summaries[:len(vrows)],
        "images": summaries[len(vrows):],
        "aggregate": aggregate,
        "source_hashes": {"video": vh, "images": ih},
        "limitations": [
            "Full-frame existing boxes are reported as state and are not part of crop-box threshold totals.",
            "Empty crop_predictions means no saved crop box; it does not imply an inference failure.",
            "No full-frame ground truth, NMS analysis, score-distribution overlap analysis, or false-positive rate is available.",
            "Empty labels are not complete event truth; no threshold is recommended or adopted.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    out = Path(__file__).resolve().parents[1] / "out"
    parser.add_argument("--video-cache", type=Path, default=out / "fixed_roi_rescue_20261029")
    parser.add_argument("--image-cache", type=Path, default=out / "fixed_roi_legacy_backgrounds_20261029")
    args = parser.parse_args(argv)
    try:
        result = analyze(args.video_cache, args.image_cache)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2, allow_nan=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
