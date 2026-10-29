from __future__ import annotations
import hashlib
import json
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "model_work" / "out" / "project_v24_videos_20261028"
PLAN_PATH = DATA / "plan.json"
PRED_PATH = DATA / "per_frame.jsonl"
OUT = ROOT / "model_work" / "out" / "v18_v24_equal_count_visual_review_20261029"
TARGETS = [
    {"video_index": 6, "frame_index": 150, "expected_source_sha256": "dbd18d5fb2587b4dafc224d12bc47984ce6e97abbdfacf0e33a7b83520b02cb1", "expected_pixel_sha256": "3a9cde27aa4fb53d548564633827fa490abf29fb5917fe53c6234d28ab7f0cf8"},
    {"video_index": 6, "frame_index": 60, "expected_source_sha256": "dbd18d5fb2587b4dafc224d12bc47984ce6e97abbdfacf0e33a7b83520b02cb1", "expected_pixel_sha256": "fd6623054c9993288f8ad2172ed6747b6e4795f16111168f077889d15417ec97"},
    {"video_index": 9, "frame_index": 840, "expected_source_sha256": "fe87820c64636d5362a35fa98e6d969de892ddebf9a4dc0b326aa6896a97fa6a", "expected_pixel_sha256": "bae6b6ba072901afa7b9f639a564ce4e4b8254e06da3d6eea0bf025be5e3d6df"},
]

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def box_iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0

def panel(image, title, prediction=None, color=(0, 220, 255)):
    canvas = image.copy()
    if prediction is not None:
        coords = [int(round(v)) for v in prediction["xyxy"]]
        cv2.rectangle(canvas, tuple(coords[:2]), tuple(coords[2:]), color, 3, cv2.LINE_AA)
        cv2.putText(canvas, f"{title} {prediction['confidence']:.3f}", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    else:
        cv2.putText(canvas, title, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas

def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {OUT}")
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in PRED_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    script_hash = sha256_file(Path(__file__))
    OUT.mkdir(parents=True, exist_ok=False)
    results = []
    for target in TARGETS:
        vi, fi = target["video_index"], target["frame_index"]
        row = next((r for r in rows if r["video_index"] == vi and r["frame_index"] == fi), None)
        if row is None:
            raise RuntimeError(f"Missing saved prediction identity for video {vi}, frame {fi}")
        clip = plan["videos"][vi - 1]
        src = Path(clip["path"])
        expected_src = target["expected_source_sha256"]
        if clip["sha256"] != expected_src or row["video_sha256"] != expected_src:
            raise RuntimeError(f"Source identity metadata mismatch for video {vi}")
        actual_src = sha256_file(src)
        if actual_src != expected_src:
            raise RuntimeError(f"Source file SHA-256 mismatch for video {vi}: {actual_src}")
        v18 = row["predictions"]["v18_last"]
        v24 = row["predictions"]["v24_last"]
        if len(v18) != 1 or len(v24) != 1:
            raise RuntimeError(f"Expected one positive box per model at video {vi}, frame {fi}")
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open source video {vi}")
        frame = None
        for decoded_index in range(fi + 1):
            ok, decoded = cap.read()
            if not ok:
                cap.release()
                raise RuntimeError(f"Sequential decode failed at frame {decoded_index} for video {vi}")
            if decoded_index == fi:
                frame = decoded
        cap.release()
        pixel_sha = hashlib.sha256(frame.tobytes()).hexdigest()
        if pixel_sha != row["source_pixel_sha256"] or pixel_sha != target["expected_pixel_sha256"]:
            raise RuntimeError(f"Decoded pixel SHA-256 mismatch for video {vi}, frame {fi}: {pixel_sha}")
        if frame.shape[1] != row["width"] or frame.shape[0] != row["height"]:
            raise RuntimeError(f"Decoded dimensions mismatch for video {vi}, frame {fi}")
        triptych = cv2.hconcat([
            panel(frame, "Original"),
            panel(frame, "v18_last", v18[0], (40, 220, 40)),
            panel(frame, "v24_last", v24[0], (255, 180, 40)),
        ])
        image_name = f"video_{vi:02d}_frame_{fi:06d}_comparison.png"
        image_path = OUT / image_name
        if not cv2.imwrite(str(image_path), triptych):
            raise RuntimeError(f"Could not write image: {image_path}")
        results.append({
            "video_index": vi, "video_path": str(src), "video_sha256": actual_src,
            "frame_index": fi, "sample_index": row["sample_index"], "time_seconds": row["time_seconds"],
            "width": row["width"], "height": row["height"],
            "decoded_pixel_sha256_bgr": pixel_sha, "prediction_pixel_sha256": row["source_pixel_sha256"],
            "v18_last": v18[0], "v24_last": v24[0], "one_to_one_iou": box_iou(v18[0]["xyxy"], v24[0]["xyxy"]),
            "comparison_image": image_name, "comparison_image_sha256": sha256_file(image_path),
        })
    manifest = {
        "purpose": "Visual review of three preselected equal-count single-box development samples; no inference performed",
        "source_plan": str(PLAN_PATH.relative_to(ROOT)), "source_predictions": str(PRED_PATH.relative_to(ROOT)),
        "source_predictions_sha256": sha256_file(PRED_PATH), "script_path": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": script_hash, "matching_rule": "single-box exact IoU; difference = 1 - IoU",
        "limitations": ["These are development-video frames, not independent ground truth.", "Visual inspection cannot establish accuracy or a general model winner."],
        "samples": results,
    }
    (OUT / "review_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(OUT), "script_sha256": script_hash, "samples": [{"video_index": x["video_index"], "frame_index": x["frame_index"], "pixel_sha256": x["decoded_pixel_sha256_bgr"], "iou": x["one_to_one_iou"], "image": x["comparison_image"]} for x in results]}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
