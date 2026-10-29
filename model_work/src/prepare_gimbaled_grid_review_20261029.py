"""Create source-only absolute-coordinate grid crops for held gimbaled frames."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "model_work/data/public_video_probe_20260930/gimbaled.webm"
OUT = ROOT / "model_work/out/gimbaled_grid_review_20261029"
EXPECTED_SOURCE_SHA256 = "bdb375cb67d9ddbc3b4521c7fb48c00e2db14a75c5fae5e1cbef7a1455f96cd8"
EXPECTED_FRAME_COUNT = 838
EXPECTED_FPS = 30.0
EXPECTED_SIZE = (748, 420)  # width, height from the accepted source intake
TARGET_FRAMES = (540, 570, 600, 630, 660, 690, 720)
CROP_XYXY = (300, 190, 700, 380)  # established burner review crop, absolute source pixels
GRID_SPACING = 25


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def draw_absolute_grid(crop, x0, y0, spacing):
    """Return a same-size crop with absolute source-pixel grid and edge labels."""
    out = crop.copy()
    height, width = out.shape[:2]
    # Thin, high-contrast grid: deliberately visual only; source pixels are hashed separately.
    for abs_x in range(((x0 + spacing - 1) // spacing) * spacing, x0 + width, spacing):
        local_x = abs_x - x0
        cv2.line(out, (local_x, 0), (local_x, height - 1), (0, 255, 255), 1, cv2.LINE_8)
        cv2.putText(out, str(abs_x), (min(local_x + 2, width - 34), 14),
                    cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0), 2, cv2.LINE_8)
        cv2.putText(out, str(abs_x), (min(local_x + 2, width - 34), 14),
                    cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255), 1, cv2.LINE_8)
    for abs_y in range(((y0 + spacing - 1) // spacing) * spacing, y0 + height, spacing):
        local_y = abs_y - y0
        cv2.line(out, (0, local_y), (width - 1, local_y), (0, 255, 255), 1, cv2.LINE_8)
        label_y = min(max(local_y - 2, 12), height - 2)
        cv2.putText(out, str(abs_y), (2, label_y), cv2.FONT_HERSHEY_PLAIN,
                    0.8, (0, 0, 0), 2, cv2.LINE_8)
        cv2.putText(out, str(abs_y), (2, label_y), cv2.FONT_HERSHEY_PLAIN,
                    0.8, (255, 255, 255), 1, cv2.LINE_8)
    return out


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite output: {OUT}")
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    source_sha_before = sha256_file(SOURCE)
    if source_sha_before != EXPECTED_SOURCE_SHA256:
        raise ValueError(f"Unexpected source SHA-256: {source_sha_before}")

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

    wanted = set(TARGET_FRAMES)
    records = []
    decoded = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[:2] != (height, width):
            cap.release()
            raise ValueError(f"Frame {decoded} has unexpected dimensions: {frame.shape}")
        if decoded in wanted:
            source_pixels = np.ascontiguousarray(frame)
            crop = source_pixels[y1:y2, x1:x2].copy()
            image = draw_absolute_grid(crop, x1, y1, GRID_SPACING)
            records.append((decoded, source_pixels, crop, image))
        decoded += 1
    cap.release()
    if decoded != EXPECTED_FRAME_COUNT:
        raise ValueError(f"Unexpected decoded frame count: {decoded}")
    got = [record[0] for record in records]
    if got != list(TARGET_FRAMES):
        raise ValueError(f"Decoded target frames differ: {got}")
    source_sha_after = sha256_file(SOURCE)
    if source_sha_after != EXPECTED_SOURCE_SHA256 or source_sha_after != source_sha_before:
        raise ValueError("Source changed during extraction")

    # The directory is created only after all source checks and extraction succeed.
    OUT.mkdir(parents=True, exist_ok=False)
    outputs = []
    for frame_index, source_pixels, crop, image in records:
        name = f"frame_{frame_index:06d}_crop_x{x1}-{x2}_y{y1}-{y2}_absolute_grid.png"
        path = OUT / name
        if not cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            raise RuntimeError(f"Failed to write {path}")
        # Re-open to validate the encoded image and exact raw-resolution crop dimensions.
        check = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if check is None or check.shape != crop.shape:
            raise RuntimeError(f"Output validation failed: {path}")
        outputs.append({
            "file": name,
            "frame_index_zero_based": frame_index,
            "source_frame_pixel_sha256": sha256_bytes(source_pixels.tobytes()),
            "uncorrected_crop_pixel_sha256": sha256_bytes(np.ascontiguousarray(crop).tobytes()),
            "output_width": int(check.shape[1]),
            "output_height": int(check.shape[0]),
            "output_sha256": sha256_file(path),
        })

    script_sha = sha256_file(Path(__file__))
    manifest = {
        "role": "source-only coordinate-grid review material; no labels or predictions",
        "source_file": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256_expected": EXPECTED_SOURCE_SHA256,
        "source_sha256_before": source_sha_before,
        "source_sha256_after": source_sha_after,
        "metadata_fps": fps,
        "source_width": width,
        "source_height": height,
        "decoded_frame_count": decoded,
        "frame_index_convention": "zero-based sequential decode order",
        "requested_frames": list(TARGET_FRAMES),
        "crop_xyxy_absolute_pixels_half_open": [x1, y1, x2, y2],
        "grid_spacing_source_pixels": GRID_SPACING,
        "output_is_raw_resolution_crop_with_visual_grid_overlay": True,
        "script_file": str(Path(__file__).relative_to(ROOT)).replace("\\", "/"),
        "script_sha256": script_sha,
        "predictions_run": False,
        "training_run": False,
        "annotations_created": False,
        "reserved_sources_accessed": False,
        "outputs": outputs,
        "limitations": [
            "Grid lines and coordinate numerals are visual overlays; source-frame and uncorrected-crop pixel hashes are recorded separately.",
            "Frame numbers identify zero-based sequential decoded frames; presentation timestamps were not checked.",
            "This preparation creates no flame boxes, event labels, or truth approval."
        ]
    }
    manifest_path = OUT / "intake.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Read-back validation for manifest and every output digest.
    reread = json.loads(manifest_path.read_text(encoding="utf-8"))
    if reread["script_sha256"] != sha256_file(Path(__file__)):
        raise RuntimeError("Script identity read-back failed")
    for item in reread["outputs"]:
        out_path = OUT / item["file"]
        if sha256_file(out_path) != item["output_sha256"]:
            raise RuntimeError(f"Output hash read-back failed: {out_path}")
    print(json.dumps({
        "output_directory": str(OUT),
        "source_sha256": source_sha_after,
        "decoded_frame_count": decoded,
        "target_frames": got,
        "outputs": len(outputs),
        "manifest_sha256": sha256_file(manifest_path),
        "script_sha256": script_sha,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
