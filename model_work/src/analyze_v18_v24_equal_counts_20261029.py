#!/usr/bin/env python3
"""Rank saved v18_last/v24_last boxes at equal-count samples.

No inference or image output. IoU measures agreement between model boxes, not
accuracy against ground truth. Multi-box samples use exhaustive maximum-sum
one-to-one assignment.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "model_work" / "out" / "project_v24_videos_20261028"
COMPARISON = DATA / "comparison_complete.json"
PLAN = DATA / "plan.json"
FRAME_CACHE = DATA / "per_frame.jsonl"
REVIEW_CACHE = DATA / "review_manifest.json"

EXPECTED_SHA256 = {
    "comparison_complete.json": "3e9eddd3b4bc306e618ab91791b327e9606a8e0e7eff0e670d20fc32a43416c7",
    "plan.json": "4ce9eb770d0cc40e74de38b84b3424084c1a9ba7a956587642f6ab3ceab32172",
    "per_frame.jsonl": "fea1bf5ad52d50fbcb01198537145d545d832a01ab3e9605fca8fdcc57455a6d",
    "review_manifest.json": "6c822f4f608534373b6e5fbdda29264de3aaef8dd8460276c54dfb9be05a5daf",
}
MODEL_A = "v18_last"
MODEL_B = "v24_last"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def box_iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def classify_box_counts(count_a: int, count_b: int) -> str:
    """Classify a sample by saved positive-box counts."""
    if count_a == 0 and count_b == 0:
        return "both_zero"
    if count_a != count_b:
        return "count_mismatch"
    return "equal_positive"

def exact_assignment_ious(boxes_a: list[list[float]],
                         boxes_b: list[list[float]]) -> tuple[float, ...]:
    """Return IoUs for the perfect assignment with greatest total IoU."""
    require(len(boxes_a) == len(boxes_b) and len(boxes_a) > 0,
            "Exact assignment requires equal, nonzero box counts.")
    matrix = [[box_iou(a, b) for b in boxes_b] for a in boxes_a]
    best_perm = None
    best_sum = -1.0
    for permutation in itertools.permutations(range(len(boxes_b))):
        total = sum(matrix[i][permutation[i]] for i in range(len(boxes_a)))
        if total > best_sum:
            best_sum, best_perm = total, permutation
    assert best_perm is not None
    return tuple(matrix[i][best_perm[i]] for i in range(len(boxes_a)))


def validate_artifact_hashes() -> dict[str, str]:
    actual = {}
    for name, expected in EXPECTED_SHA256.items():
        path = DATA / name
        require(path.is_file(), f"Missing required saved artifact: {path}")
        digest = sha256_file(path)
        require(digest == expected,
                f"{name} SHA-256 mismatch: expected {expected}, got {digest}")
        actual[name] = digest
    comparison = load_json(COMPARISON)
    require(comparison.get("plan_sha256") == actual["plan.json"],
            "comparison_complete.json does not bind to the current plan.json")
    require(comparison.get("review_manifest_sha256") == actual["review_manifest.json"],
            "comparison_complete.json does not bind to the current review_manifest.json")
    return actual


def validate_cache_identity(plan: dict, comparison: dict,
                            rows: list[dict], review_manifest: dict) -> dict[int, dict]:
    clips = {int(item["video_index"]): item for item in comparison["clips"]}
    require(len(clips) == len(plan["videos"]) == 9,
            "Expected exactly nine plan videos and nine comparison clips.")
    by_video: dict[int, list[dict]] = {i: [] for i in range(1, 10)}
    seen = set()
    for row in rows:
        vi = int(row["video_index"])
        require(vi in by_video, f"Unexpected video index: {vi}")
        identity = (vi, int(row["sample_index"]), int(row["frame_index"]))
        require(identity not in seen, f"Duplicate per-frame identity: {identity}")
        seen.add(identity)
        by_video[vi].append(row)

    source_hashes = {}
    for vi, source in enumerate(plan["videos"], start=1):
        clip = clips[vi]
        source_path = Path(source["path"])
        expected_video_sha = source["sha256"]
        require(clip["source_sha256"] == expected_video_sha,
                f"Comparison/plan source SHA mismatch for video {vi}.")
        require(source_path.is_file(), f"Missing planned development video: {source_path}")
        actual_video_sha = sha256_file(source_path)
        require(actual_video_sha == expected_video_sha,
                f"Source video SHA-256 mismatch for video {vi}: {actual_video_sha}")
        source_hashes[vi] = actual_video_sha

        video_rows = sorted(by_video[vi], key=lambda r: int(r["sample_index"]))
        step = int(clip["step"])
        expected_frames = list(range(0, int(clip["decoded_frames"]), step))
        actual_frames = [int(r["frame_index"]) for r in video_rows]
        expected_samples = list(range(len(expected_frames)))
        actual_samples = [int(r["sample_index"]) for r in video_rows]
        require(actual_frames == expected_frames,
                f"Sampled frame sequence mismatch for video {vi}.")
        require(actual_samples == expected_samples,
                f"Sample index sequence mismatch for video {vi}.")
        require(len(video_rows) == int(clip["sampled_frames"]),
                f"Sample count mismatch for video {vi}.")
        for row in video_rows:
            require(row["video_sha256"] == expected_video_sha,
                    f"Per-frame source SHA mismatch at video {vi}, sample {row['sample_index']}.")
            require(int(row["frame_index"]) == int(row["sample_index"]) * step,
                    f"Frame/sample mapping mismatch at video {vi}, sample {row['sample_index']}.")
            require(row["width"] > 0 and row["height"] > 0,
                    f"Invalid frame dimensions at video {vi}, sample {row['sample_index']}.")
            pixel_sha = row.get("source_pixel_sha256", "")
            require(len(pixel_sha) == 64 and all(c in "0123456789abcdef" for c in pixel_sha.lower()),
                    f"Invalid saved pixel SHA at video {vi}, sample {row['sample_index']}.")

    # Cross-check retained fixed review-frame identities against the prediction
    # cache. Pixel digests are compared as metadata; this script does not decode images.
    review_count = 0
    review_videos = review_manifest.get("videos", [])
    require(len(review_videos) == 9, "Review manifest does not contain nine videos.")
    rows_by_identity = {
        (int(r["video_index"]), int(r["sample_index"]), int(r["frame_index"])): r
        for r in rows
    }
    for video in review_videos:
        for cached in video.get("frames", []):
            vi = int(cached["video_index"])
            identity = (vi, int(cached["sample_index"]), int(cached["frame_index"]))
            row = rows_by_identity.get(identity)
            require(row is not None, f"Review cache identity absent from per-frame cache: {identity}")
            require(cached["video_sha256"] == row["video_sha256"] == source_hashes[vi],
                    f"Review cache source identity mismatch: {identity}")
            require(cached["source_pixel_sha256"] == row["source_pixel_sha256"],
                    f"Review cache pixel digest mismatch: {identity}")
            review_count += 1
    require(review_count == 27, f"Expected 27 fixed review-cache identities, got {review_count}.")
    require(len(rows) == int(comparison["total_samples"]),
            "Expected 307 unique sampled prediction records.")
    return clips


def recompute_clip_summaries(rows: list[dict], clips: dict[int, dict]) -> None:
    rows_by_video = {
        vi: [r for r in rows if int(r["video_index"]) == vi]
        for vi in clips
    }
    for vi, clip in clips.items():
        video_rows = rows_by_video[vi]
        for model, summary in clip["model_counts"].items():
            counts = [len(r["predictions"][model]) for r in video_rows]
            histogram = dict(sorted(Counter(counts).items()))
            expected_histogram = {
                str(k): int(v) for k, v in summary["box_count_histogram"].items()
            }
            require({str(k): v for k, v in histogram.items()} == expected_histogram,
                    f"Box-count histogram mismatch for video {vi}, model {model}.")
            frames_with_boxes = sum(count > 0 for count in counts)
            total_boxes = sum(counts)
            longest_empty = current_empty = 0
            for count in counts:
                current_empty = current_empty + 1 if count == 0 else 0
                longest_empty = max(longest_empty, current_empty)
            require(frames_with_boxes == summary["frames_with_any_box"],
                    f"Frames-with-boxes mismatch for video {vi}, model {model}.")
            require(total_boxes == summary["boxes"],
                    f"Total-box count mismatch for video {vi}, model {model}.")
            require(longest_empty == summary["longest_empty_sample_run"],
                    f"Empty-run mismatch for video {vi}, model {model}.")


def main() -> int:
    hashes = validate_artifact_hashes()
    comparison = load_json(COMPARISON)
    plan = load_json(PLAN)
    require(int(comparison["total_samples"]) == 307, "Expected 307 unique sampled prediction records.")
    review_manifest = load_json(REVIEW_CACHE)
    rows = [json.loads(line) for line in FRAME_CACHE.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    clips = validate_cache_identity(plan, comparison, rows, review_manifest)
    recompute_clip_summaries(rows, clips)

    ranked = []
    both_zero = count_mismatch = equal_positive = 0
    for row in rows:
        a = row["predictions"][MODEL_A]
        b = row["predictions"][MODEL_B]
        category = classify_box_counts(len(a), len(b))
        if category == "both_zero":
            both_zero += 1
            continue
        if category == "count_mismatch":
            count_mismatch += 1
            continue
        equal_positive += 1
        ious = exact_assignment_ious(
            [x["xyxy"] for x in a], [x["xyxy"] for x in b]
        )
        mean_iou = sum(ious) / len(ious)
        ranked.append({
            "video": int(row["video_index"]),
            "sample": int(row["sample_index"]),
            "frame": int(row["frame_index"]),
            "time": float(row["time_seconds"]),
            "boxes": len(a),
            "mean_iou": mean_iou,
            "difference": 1.0 - mean_iou,
            "pair_ious": ious,
        })
    require((equal_positive, both_zero, count_mismatch) == (256, 36, 15),
            "Equal-count partition differs from the expected 256/36/15.")
    ranked.sort(key=lambda item: (-item["difference"], item["video"], item["sample"]))

    print("Scope: saved v18_last/v24_last predictions on 307 development-video samples.")
    print("IoU measures model-box agreement, not accuracy against ground truth.")
    print("Identity checks: artifact hashes; all 9 source-video SHA-256;")
    print("  all 307 sample/frame/source links; 27 retained review-cache identities;")
    print("  per-video/model box-count summaries.")
    print("Pixel-hash limit: the 27 cached review pixel digests are cross-checked as metadata;")
    print("  this script does not decode frames or recompute pixel hashes for all 307 samples.")
    print("Verified SHA-256:")
    for name, digest in hashes.items():
        print(f"  {name}: {digest}")
    print(f"Partition: equal positive box counts={equal_positive}, both zero={both_zero}, "
          f"counts differ={count_mismatch}")
    print("Ranking: descending difference (1 - mean matched IoU); multi-box pairs use")
    print("  exhaustive maximum-total-IoU one-to-one assignment.")
    print("rank video sample frame time_s boxes mean_IoU difference pair_IoUs")
    for rank, item in enumerate(ranked, start=1):
        pairs = ",".join(f"{value:.6f}" for value in item["pair_ious"])
        print(f"{rank:03d} {item['video']:02d} {item['sample']:03d} "
              f"{item['frame']:04d} {item['time']:.3f} {item['boxes']} "
              f"{item['mean_iou']:.6f} {item['difference']:.6f} [{pairs}]")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
