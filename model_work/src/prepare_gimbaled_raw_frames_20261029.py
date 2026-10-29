"""Export verified raw gimbaled video frames and unaltered burner crops for blind review."""
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "model_work/data/public_video_probe_20260930/gimbaled.webm"
OUT = ROOT / "model_work/out/gimbaled_raw_frames_20261029"
LEGACY_RUNTIME = ROOT / "model_work/out/gimbaled_grid_review_20261029/prepare_gimbaled_grid_review_20261029_runtime.py"
INTAKE = ROOT / "model_work/out/gimbaled_truth_intake_20261029/intake.json"
EXPECTED_SOURCE_SHA256 = "bdb375cb67d9ddbc3b4521c7fb48c00e2db14a75c5fae5e1cbef7a1455f96cd8"
EXPECTED_FRAME_COUNT = 838
EXPECTED_FPS = 30.0
EXPECTED_SIZE = (748, 420)
TARGET_FRAMES = (540, 570, 600, 630, 660, 690, 720)
CROP_XYXY = (300, 190, 700, 380)
EXPECTED_FRAME_SHA256 = {
    540: "3f769b1425cda60d29d6ef9faabfe68f03d0abfaffdca8612b3250dbbdc53668",
    570: "a860c07df70ed75a7ef043046d455cfb20470fb7f4aa794e507abbd3d01a323a",
    600: "7ed83dd752d30c76b7531a3d24f3d00357e37b950c2b89090de5ca22b202564a",
    630: "34ff4c701f8c08c824038b0d7aebdb149c1371d8f5112a7aee9352f294c9b61e",
    660: "126d1e8b0eaab6908c6a932cfbbe69d4bcc303d4b28925429e705a5a0f00bfba",
    690: "e2a4d2af01abeda1ffe65cb5557d47fd3f759a0ce4b5edada93d3f99d755f9c2",
    720: "48c1ee1bed6e531cda8efdbbe2691ec0c06324066d47d4bb4501df7fe50e9348",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_pixels(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite output: {OUT}")
    if not SOURCE.is_file() or not LEGACY_RUNTIME.is_file() or not INTAKE.is_file():
        raise FileNotFoundError("Required source, original runtime script, or intake is missing")
    intake = json.loads(INTAKE.read_text(encoding="utf-8"))
    intake_frames = {int(row["frame_index"]): row["pixel_sha256"] for row in intake["samples"]}
    for index, expected in EXPECTED_FRAME_SHA256.items():
        if intake_frames.get(index) != expected:
            raise ValueError(f"Existing intake does not match embedded frame hash for {index}")

    source_before = sha256_file(SOURCE)
    if source_before != EXPECTED_SOURCE_SHA256:
        raise ValueError(f"Unexpected source SHA-256: {source_before}")
    cap = cv2.VideoCapture(str(SOURCE))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source video: {SOURCE}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if abs(fps - EXPECTED_FPS) > 1e-6 or (width, height) != EXPECTED_SIZE:
        cap.release()
        raise ValueError(f"Unexpected metadata: fps={fps}, size={width}x{height}")
    x1, y1, x2, y2 = CROP_XYXY
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        cap.release()
        raise ValueError("Crop is outside source frame bounds")

    records = []
    decoded = 0
    wanted = set(TARGET_FRAMES)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape != (height, width, 3):
                raise ValueError(f"Frame {decoded} has unexpected dimensions: {frame.shape}")
            if decoded in wanted:
                frame = np.ascontiguousarray(frame)
                digest = sha256_pixels(frame)
                if digest != EXPECTED_FRAME_SHA256[decoded] or digest != intake_frames[decoded]:
                    raise ValueError(f"Frame {decoded} pixel hash differs from existing intake: {digest}")
                crop = np.ascontiguousarray(frame[y1:y2, x1:x2])
                records.append((decoded, frame.copy(), crop.copy(), digest, sha256_pixels(crop)))
            decoded += 1
    finally:
        cap.release()
    if decoded != EXPECTED_FRAME_COUNT:
        raise ValueError(f"Unexpected decoded frame count: {decoded}")
    if [row[0] for row in records] != list(TARGET_FRAMES):
        raise ValueError(f"Target frame sequence differs: {[row[0] for row in records]}")
    source_after = sha256_file(SOURCE)
    if source_after != source_before or source_after != EXPECTED_SOURCE_SHA256:
        raise ValueError("Source video changed during extraction")

    # Create output only after source, sequential decode, intake, and pixel checks pass.
    OUT.mkdir(parents=True, exist_ok=False)
    runtime_copy = OUT / "prepare_gimbaled_grid_review_20261029_runtime.py"
    shutil.copyfile(LEGACY_RUNTIME, runtime_copy)
    old_runtime_hash = sha256_file(LEGACY_RUNTIME)
    if sha256_file(runtime_copy) != old_runtime_hash:
        raise RuntimeError("Original runtime script byte-copy validation failed")

    outputs = []
    for index, frame, crop, frame_hash, crop_hash in records:
        for suffix, pixels, pixel_hash in (("full", frame, frame_hash), ("crop_x300-700_y190-380", crop, crop_hash)):
            name = f"frame_{index:06d}_{suffix}.png"
            path = OUT / name
            if not cv2.imwrite(str(path), pixels, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                raise RuntimeError(f"Failed to write {path}")
            check = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if check is None or check.shape != pixels.shape or sha256_pixels(check) != pixel_hash:
                raise RuntimeError(f"PNG decoded-pixel validation failed: {path}")
            outputs.append({
                "file": name,
                "frame_index_zero_based": index,
                "kind": "full_resolution_source_frame" if suffix == "full" else "unaltered_source_pixel_crop",
                "crop_xyxy_absolute_pixels_half_open": None if suffix == "full" else list(CROP_XYXY),
                "width": int(check.shape[1]),
                "height": int(check.shape[0]),
                "source_frame_pixel_sha256": frame_hash,
                "output_decoded_pixel_sha256": sha256_pixels(check),
                "png_file_sha256": sha256_file(path),
                "overlay_or_text_added": False,
            })

    script_hash = sha256_file(Path(__file__))
    manifest = {
        "role": "unlabelled raw-pixel source frames for blind human viewing",
        "source_file": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256_expected": EXPECTED_SOURCE_SHA256,
        "source_sha256_before": source_before,
        "source_sha256_after": source_after,
        "fps_metadata": fps,
        "source_width": width,
        "source_height": height,
        "decoded_frame_count": decoded,
        "frame_index_convention": "zero-based sequential decode order",
        "requested_frames": list(TARGET_FRAMES),
        "crop_xyxy_absolute_pixels_half_open": list(CROP_XYXY),
        "existing_intake_file": str(INTAKE.relative_to(ROOT)).replace("\\", "/"),
        "existing_intake_frame_pixel_sha256_matches": {str(k): intake_frames[k] == EXPECTED_FRAME_SHA256[k] for k in TARGET_FRAMES},
        "original_grid_runtime_script_byte_copy": runtime_copy.name,
        "original_grid_runtime_script_sha256": old_runtime_hash,
        "script_file": str(Path(__file__).relative_to(ROOT)).replace("\\", "/"),
        "script_sha256": script_hash,
        "training_run": False,
        "predictions_run": False,
        "boxes_or_labels_created": False,
        "reserved_sources_accessed": False,
        "outputs": outputs,
        "limitations": [
            "Frame numbers are zero-based sequential decode indices; presentation timestamps were not checked.",
            "The crop reproduces the established [300,190,700,380) source-pixel region without overlays; its boundary may exclude relevant context.",
            "Raw pixels and hashes support visual review but do not establish flame truth, event continuity, or source independence.",
        ],
    }
    manifest_path = OUT / "intake.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    reread = json.loads(manifest_path.read_text(encoding="utf-8"))
    if reread["script_sha256"] != sha256_file(Path(__file__)):
        raise RuntimeError("Script hash read-back validation failed")
    if sha256_file(runtime_copy) != old_runtime_hash:
        raise RuntimeError("Runtime-script byte-copy hash changed")
    for item in reread["outputs"]:
        path = OUT / item["file"]
        if sha256_file(path) != item["png_file_sha256"]:
            raise RuntimeError(f"PNG file hash read-back failed: {path}")
        check = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if check is None or sha256_pixels(check) != item["output_decoded_pixel_sha256"]:
            raise RuntimeError(f"PNG source-pixel read-back failed: {path}")
    print(json.dumps({"output_directory": str(OUT), "source_sha256": source_after,
                      "decoded_frames": decoded, "target_frames": list(TARGET_FRAMES),
                      "png_count": len(outputs), "script_sha256": script_hash,
                      "copied_original_runtime_sha256": old_runtime_hash,
                      "manifest_sha256": sha256_file(manifest_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
