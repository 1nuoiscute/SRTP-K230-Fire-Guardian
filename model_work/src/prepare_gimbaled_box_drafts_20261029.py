"""Prepare unapproved visual box drafts from already verified gimbaled full-frame PNGs.

No source video, model, reserved source, inference, or training is accessed.
"""
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "model_work/out/gimbaled_raw_frames_20261029"
SOURCE_INTAKE = SOURCE_DIR / "intake.json"
TRUTH_INTAKE = ROOT / "model_work/out/gimbaled_truth_intake_20261029/intake.json"
OUT = ROOT / "model_work/out/gimbaled_box_drafts_20261029"
FRAMES = {
    540: (520, 195, 585, 214),
    570: (515, 222, 580, 243),
    600: (510, 238, 575, 258),
    630: (510, 280, 578, 301),
    660: (510, 291, 578, 311),
    690: (510, 313, 578, 335),
    720: (510, 307, 578, 329),
}
EXPECTED_FULL_PIXEL_SHA256 = {
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


def sha256_pixels(image):
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite output: {OUT}")
    if not SOURCE_INTAKE.is_file() or not TRUTH_INTAKE.is_file():
        raise FileNotFoundError("Required existing intake is missing")

    source_intake = json.loads(SOURCE_INTAKE.read_text(encoding="utf-8"))
    truth_intake = json.loads(TRUTH_INTAKE.read_text(encoding="utf-8"))
    source_rows = {
        row["frame_index_zero_based"]: row
        for row in source_intake["outputs"]
        if row["kind"] == "full_resolution_source_frame"
    }
    truth_rows = {row["frame_index"]: row for row in truth_intake["samples"]}
    validated = []

    # Finish every input/hash/geometry check before creating the output directory.
    for index, box in FRAMES.items():
        x1, y1, x2, y2 = box
        name = f"frame_{index:06d}_full.png"
        path = SOURCE_DIR / name
        if not path.is_file() or index not in source_rows or index not in truth_rows:
            raise FileNotFoundError(f"Missing full-frame PNG or intake row for {index}")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Cannot decode color PNG for frame {index}")
        height, width = image.shape[:2]
        if not (0 <= x1 < x2 < width and 0 <= y1 < y2 < height):
            raise ValueError(f"Box is outside frame {index}: {box}, dimensions={width}x{height}")
        pixels_sha = sha256_pixels(image)
        expected = EXPECTED_FULL_PIXEL_SHA256[index]
        raw_row = source_rows[index]
        truth_row = truth_rows[index]
        if pixels_sha != expected:
            raise ValueError(f"Decoded pixels differ from embedded identity for frame {index}")
        if pixels_sha != raw_row["source_frame_pixel_sha256"] or pixels_sha != raw_row["output_decoded_pixel_sha256"]:
            raise ValueError(f"Decoded pixels differ from raw-frame intake for frame {index}")
        if pixels_sha != truth_row["pixel_sha256"]:
            raise ValueError(f"Decoded pixels differ from truth intake for frame {index}")
        if (width, height) != (raw_row["width"], raw_row["height"]):
            raise ValueError(f"Dimensions differ from raw-frame intake for frame {index}")
        if (width, height) != (truth_row["width"], truth_row["height"]):
            raise ValueError(f"Dimensions differ from truth intake for frame {index}")
        png_sha = sha256_file(path)
        if png_sha != raw_row["png_file_sha256"]:
            raise ValueError(f"PNG file SHA differs from raw-frame intake for frame {index}")
        validated.append((index, box, path, image, width, height, pixels_sha, png_sha))

    OUT.mkdir(parents=True, exist_ok=False)
    outputs = []
    for index, box, source_path, image, width, height, pixels_sha, source_png_sha in validated:
        raw_name = f"frame_{index:06d}_unlabelled.png"
        overlay_name = f"frame_{index:06d}_box_draft.png"
        raw_out = OUT / raw_name
        overlay_out = OUT / overlay_name
        shutil.copyfile(source_path, raw_out)
        if sha256_file(raw_out) != source_png_sha:
            raise RuntimeError(f"Unlabelled PNG byte-copy verification failed for {index}")

        overlay = image.copy()
        x1, y1, x2, y2 = box
        # Coordinates are preserved literally; OpenCV draws to the supplied endpoint.
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_8)
        cv2.putText(overlay, "UNAPPROVED DRAFT - POSSIBLE REFLECTION/OCCLUSION",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_8)
        cv2.putText(overlay, "UNAPPROVED DRAFT - POSSIBLE REFLECTION/OCCLUSION",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_8)
        if not cv2.imwrite(str(overlay_out), overlay, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            raise RuntimeError(f"Could not write overlay for frame {index}")
        readback = cv2.imread(str(overlay_out), cv2.IMREAD_COLOR)
        if readback is None or readback.shape != image.shape:
            raise RuntimeError(f"Overlay PNG dimensions/readback failed for frame {index}")
        outputs.append({
            "frame_index_zero_based": index,
            "unlabelled_png": raw_name,
            "unlabelled_png_sha256": sha256_file(raw_out),
            "overlay_png": overlay_name,
            "overlay_png_sha256": sha256_file(overlay_out),
            "source_frame_pixel_sha256": pixels_sha,
            "source_full_png_sha256": source_png_sha,
            "dimensions_width_height": [width, height],
            "bbox_xyxy_as_supplied": list(box),
            "bbox_coordinate_note": "Verbatim controller visual rough estimate; drawn on full-frame image; not a truth annotation.",
            "bbox_boundary_validation": {
                "inside_image": True,
                "condition": "0 <= x1 < x2 < width and 0 <= y1 < y2 < height",
            },
            "review_status": "UNAPPROVED",
            "caveat": "Possible reflection and occlusion; requires human review.",
        })

    script_path = Path(__file__).resolve()
    script_sha = sha256_file(script_path)
    script_copy_name = "prepare_gimbaled_box_drafts_20261029.py"
    script_copy_path = OUT / script_copy_name
    shutil.copyfile(script_path, script_copy_path)
    if sha256_file(script_copy_path) != script_sha:
        raise RuntimeError("Original script byte snapshot SHA-256 mismatch")

    manifest = {
        "role": "unapproved visual box drafts for human inspection only",
        "approval_status": "UNAPPROVED",
        "coordinates_are": "verbatim controller visual rough estimates, not verified truth labels",
        "caveat": "Possible reflection and occlusion; reviewers must decide whether each box is acceptable.",
        "source_frame_directory": "model_work/out/gimbaled_raw_frames_20261029",
        "source_intake": "model_work/out/gimbaled_raw_frames_20261029/intake.json",
        "source_intake_sha256": sha256_file(SOURCE_INTAKE),
        "existing_truth_intake": "model_work/out/gimbaled_truth_intake_20261029/intake.json",
        "existing_truth_intake_sha256": sha256_file(TRUTH_INTAKE),
        "script_file": "model_work/src/prepare_gimbaled_box_drafts_20261029.py",
        "script_sha256": script_sha,
        "original_script_byte_snapshot": script_copy_name,
        "original_script_byte_snapshot_sha256": sha256_file(script_copy_path),
        "training_run": False,
        "inference_run": False,
        "reserved_sources_accessed": False,
        "source_video_accessed": False,
        "frames": outputs,
        "limitations": [
            "Drafts are not approved annotations or ground truth.",
            "Possible reflection and occlusion make object boundaries uncertain.",
            "Only the seven named full-frame PNGs and their two existing intake files were read.",
        ],
    }
    manifest_path = OUT / "draft_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    reread = json.loads(manifest_path.read_text(encoding="utf-8"))
    if reread["script_sha256"] != sha256_file(script_path):
        raise RuntimeError("Script identity readback failed")
    for item in reread["frames"]:
        for field, hash_field in (("unlabelled_png", "unlabelled_png_sha256"),
                                  ("overlay_png", "overlay_png_sha256")):
            if sha256_file(OUT / item[field]) != item[hash_field]:
                raise RuntimeError(f"Output PNG hash readback failed: {item[field]}")
    print(json.dumps({
        "output_directory": str(OUT),
        "frames": len(outputs),
        "unlabelled_pngs": len(outputs),
        "overlay_pngs": len(outputs),
        "all_boxes_inside_frame": all(x["bbox_boundary_validation"]["inside_image"] for x in outputs),
        "approval_status": "UNAPPROVED",
        "script_sha256": script_sha,
        "manifest_sha256": sha256_file(manifest_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
