"""Bounded identity and pHash triage for four Pexels 9685496 frames vs v20 train lineage."""
import hashlib
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "model_work/out/pexels_9685496_source_review_20261029"
FRAME_DIR = REVIEW / "frames"
SAMPLE_MANIFEST = REVIEW / "sampling_manifest.json"
SOURCE_VIDEO = ROOT / "model_work/data/pexels_ignition_quarantine_20261029/9685496.mp4"
DATA = Path("D:/SRTP_Datasets/fire_kitchen_source_v20_20261003")
TRAIN_MANIFEST = DATA / "build_manifest.json"
OUT = ROOT / "model_work/out/pexels_9685496_v20_dedupe_audit_20261029"
EXPECTED_SOURCE_SHA256 = "2fec6c7dc9b0578e75d59d01a01b5f159f7bf4614451ad3f87cf0288d9341b96"
EXPECTED_SAMPLE_MANIFEST_SHA256 = "a9949a4982eeefce7acd4ffcc2f14474a85d8fab8c62f09186a102c7fc2c3e1f"
EXPECTED_TRAIN_MANIFEST_SHA256 = "15a509075be0d2ff062894cd7de1160699bbdc7b657d8199bfba77cb80bf36fa"
FRAME_INDICES = (88, 92, 460, 464)
NEAREST_COUNT = 10
LIMITATIONS = [
    "Coverage is limited to four sampled frames and the 1,445 images in the available v20 train lineage.",
    "No generic pretraining corpus, unavailable source, unsampled video frame, or other exposure history was audited.",
    "No exact pixel match does not establish source or event independence; crops, resizing and lossy encoding can evade pixel identity.",
    "pHash distances are review leads only and do not establish identity or independence.",
    "No model inference, training, source admission, or reserved-source access was performed.",
]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def validate_sampling_manifest(manifest, manifest_sha256, validated_source_sha256):
    if manifest_sha256 != EXPECTED_SAMPLE_MANIFEST_SHA256:
        raise ValueError("sampling_manifest SHA-256 mismatch")
    for field in ("source_sha256_before_decode", "source_sha256_after_decode"):
        if manifest.get(field) != validated_source_sha256:
            raise ValueError(f"sampling_manifest {field} does not match validated source video SHA-256")


def fingerprint_bgr(bgr):
    """Match the repository pHash: OpenCV BGR -> gray -> 32x32 DCT -> 8x8 median bits."""
    import cv2

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    coeff = cv2.dct(cv2.resize(gray, (32, 32)).astype("float32"))[:8, :8].flatten()
    bits = coeff > __import__("numpy").median(coeff[1:])
    bits[0] = False
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return value


def write_new(path, data):
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()


def copy_new(source, destination):
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
    return sha256_file(destination)


def write_json_new(path, value):
    write_new(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite output: {OUT}")
    OUT.mkdir(parents=True, exist_ok=False)
    started = utcnow()
    written = []
    try:
        for path in (SAMPLE_MANIFEST, SOURCE_VIDEO, TRAIN_MANIFEST):
            if not path.is_file():
                raise FileNotFoundError(path)
        sample_manifest_hash = sha256_file(SAMPLE_MANIFEST)
        source_hash = sha256_file(SOURCE_VIDEO)
        train_manifest_hash = sha256_file(TRAIN_MANIFEST)
        if source_hash != EXPECTED_SOURCE_SHA256:
            raise ValueError("Quarantined source video SHA-256 mismatch")
        if train_manifest_hash != EXPECTED_TRAIN_MANIFEST_SHA256:
            raise ValueError("v20 build_manifest SHA-256 mismatch")

        sample_manifest = json.loads(SAMPLE_MANIFEST.read_text(encoding="utf-8"))
        validate_sampling_manifest(sample_manifest, sample_manifest_hash, source_hash)
        train_manifest = json.loads(TRAIN_MANIFEST.read_text(encoding="utf-8"))
        rows = train_manifest["lineage"]
        if len(rows) != 1445 or len({row["sha256"] for row in rows}) != 1445:
            raise ValueError("Expected 1,445 unique v20 lineage rows")
        by_index = {row["frame_index_zero_based"]: row for row in sample_manifest["frames"]}
        if any(index not in by_index for index in FRAME_INDICES):
            raise ValueError("Required frame absent from sampling_manifest")
        # Visual dependencies are needed only after pure manifest identity checks pass.
        import cv2

        source_frames = {}
        frame_inputs = []
        for index in FRAME_INDICES:
            rec = by_index[index]
            path = REVIEW / rec["png_file"]
            png_hash = sha256_file(path)
            if png_hash != rec["png_file_sha256"]:
                raise ValueError(f"Sample PNG SHA-256 mismatch at frame {index}")
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"OpenCV could not decode sample PNG at frame {index}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb_hash = sha256_bytes(rgb.tobytes(order="C"))
            if list(rgb.shape[:2]) != [rec["height"], rec["width"]]:
                raise ValueError(f"Sample dimensions mismatch at frame {index}")
            if rgb_hash != rec["pixel_sha256_rgb_row_major"]:
                raise ValueError(f"RGB row-major pixel SHA-256 mismatch at frame {index}")
            source_frames[index] = {
                "frame_index_zero_based": index,
                "width": int(rgb.shape[1]), "height": int(rgb.shape[0]),
                "png_file": rec["png_file"], "png_file_sha256_expected": rec["png_file_sha256"],
                "png_file_sha256_actual": png_hash,
                "pixel_sha256_rgb_row_major_expected": rec["pixel_sha256_rgb_row_major"],
                "pixel_sha256_rgb_row_major_actual": rgb_hash,
                "phash_64": fingerprint_bgr(bgr),
                "exact_pixel_matches": [],
            }
            frame_inputs.append({"frame_index_zero_based": index, "path": str(path),
                                 "png_sha256": png_hash, "pixel_sha256_rgb_row_major": rgb_hash})

        input_record = {
            "role": "bounded source and manifest identity record",
            "source_video": str(SOURCE_VIDEO), "source_video_bytes": SOURCE_VIDEO.stat().st_size,
            "source_video_sha256_expected": EXPECTED_SOURCE_SHA256, "source_video_sha256_actual": source_hash,
            "sampling_manifest": str(SAMPLE_MANIFEST), "sampling_manifest_sha256": sample_manifest_hash,
            "v20_build_manifest": str(TRAIN_MANIFEST),
            "v20_build_manifest_sha256_expected": EXPECTED_TRAIN_MANIFEST_SHA256,
            "v20_build_manifest_sha256_actual": train_manifest_hash,
            "v20_lineage_rows": len(rows), "v20_unique_manifest_sha256_count": len({r["sha256"] for r in rows}),
            "selected_frames": frame_inputs,
            "pixel_hash_contract": "SHA-256 of RGB uint8 row-major bytes, checked against sampling_manifest; dimensions checked separately.",
            "phash_contract": "Repository audit_video_probe_overlap.fingerprint algorithm; all inputs decoded with OpenCV as BGR, then converted internally to gray.",
        }
        archived_copies = []
        for original, filename, expected_hash in (
            (SOURCE_VIDEO, "source_9685496.mp4", source_hash),
            (SAMPLE_MANIFEST, "sampling_manifest.json", sample_manifest_hash),
            (TRAIN_MANIFEST, "v20_build_manifest.json", train_manifest_hash),
        ):
            archived_path = OUT / filename
            archived_hash = copy_new(original, archived_path)
            if archived_hash != expected_hash:
                raise RuntimeError(f"Archived input copy hash mismatch: {filename}")
            archived_copies.append({"file": filename, "bytes": archived_path.stat().st_size,
                                    "sha256": archived_hash, "matches_original": True})
        input_record["archived_input_copies"] = archived_copies
        write_json_new(OUT / "source_and_manifest_identities.json", input_record)
        written.append("source_and_manifest_identities.json")

        nearest = {index: [] for index in FRAME_INDICES}
        exact_counts = {index: 0 for index in FRAME_INDICES}
        checks_path = OUT / "training_image_hash_and_phash_checks.jsonl"
        with checks_path.open("xb") as checks:
            for row_index, row in enumerate(rows, start=1):
                relative = row["image"]
                path = DATA / "images/train" / relative
                if not path.is_file():
                    raise FileNotFoundError(f"Missing v20 lineage image: {path}")
                actual_sha = sha256_file(path)
                if actual_sha != row["sha256"]:
                    raise ValueError(f"v20 lineage SHA-256 mismatch: {relative}")
                bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError(f"Unreadable v20 lineage image: {relative}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                pixel_hash = sha256_bytes(rgb.tobytes(order="C"))
                phash = fingerprint_bgr(bgr)
                distances = {}
                for index, frame in source_frames.items():
                    distance = (phash ^ frame["phash_64"]).bit_count()
                    distances[str(index)] = distance
                    nearest[index].append((distance, relative, row["sha256"]))
                    if rgb.shape[:2] == (frame["height"], frame["width"]) and pixel_hash == frame["pixel_sha256_rgb_row_major_actual"]:
                        exact_counts[index] += 1
                        frame["exact_pixel_matches"].append({"image": relative, "sha256": actual_sha})
                record = {
                    "manifest_row": row_index, "image": relative,
                    "expected_sha256": row["sha256"], "actual_sha256": actual_sha,
                    "sha256_verified": True, "decoded_width": int(rgb.shape[1]), "decoded_height": int(rgb.shape[0]),
                    "decoded_pixel_sha256_rgb_row_major": pixel_hash, "phash_64": phash,
                    "phash_hamming_distance_by_frame_index": distances,
                }
                checks.write((json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
                if row_index % 250 == 0:
                    checks.flush()
            checks.flush()
        written.append("training_image_hash_and_phash_checks.jsonl")

        nearest_records = {}
        for index, entries in nearest.items():
            nearest_records[str(index)] = [
                {"rank": rank, "hamming_distance": distance, "image": image, "sha256": digest}
                for rank, (distance, image, digest) in enumerate(sorted(entries)[:NEAREST_COUNT], start=1)
            ]
            source_frames[index]["nearest_top_10"] = nearest_records[str(index)]
            source_frames[index].pop("phash_64")
        write_json_new(OUT / "per_frame_nearest_phash.json", source_frames)
        written.append("per_frame_nearest_phash.json")
        summary = {
            "status": "completed", "started_utc": started, "completed_utc": utcnow(),
            "role": "bounded identity and similarity triage only; no source-independence conclusion",
            "inputs_record": "source_and_manifest_identities.json",
            "training_image_checks": "training_image_hash_and_phash_checks.jsonl",
            "training_images_manifested": len(rows), "training_images_sha256_verified": len(rows),
            "sampled_frames": list(FRAME_INDICES), "frames_with_exact_rgb_pixel_matches": sum(bool(source_frames[i]["exact_pixel_matches"]) for i in FRAME_INDICES),
            "exact_rgb_pixel_match_count_by_frame": {str(i): exact_counts[i] for i in FRAME_INDICES},
            "minimum_phash_distance_by_frame": {str(i): min(d for d, _, _ in nearest[i]) for i in FRAME_INDICES},
            "nearest_results": "Top 10 per sampled frame; all 1,445 x 4 distances are in JSONL.",
            "limitations": LIMITATIONS,
            "script_sha256": sha256_file(Path(__file__)),
            "runtime": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "opencv": cv2.__version__},
        }
        # Recheck immutable source metadata and all four sample artifacts before closing the audit.
        if sha256_file(SOURCE_VIDEO) != source_hash or sha256_file(SAMPLE_MANIFEST) != sample_manifest_hash or sha256_file(TRAIN_MANIFEST) != train_manifest_hash:
            raise RuntimeError("An input identity changed during audit")
        for index in FRAME_INDICES:
            rec = by_index[index]
            if sha256_file(REVIEW / rec["png_file"]) != rec["png_file_sha256"]:
                raise RuntimeError(f"Sample PNG changed during audit at frame {index}")
        write_json_new(OUT / "summary.json", summary)
        written.append("summary.json")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except Exception as exc:
        failure = {"status": "failed", "started_utc": started, "failed_utc": utcnow(),
                   "error_type": type(exc).__name__, "error": str(exc), "artifacts_written": written,
                   "script_sha256": sha256_file(Path(__file__))}
        failure_path = OUT / "failure.json"
        if not failure_path.exists():
            write_json_new(failure_path, failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2), file=sys.stderr)
        raise


if __name__ == "__main__":
    main()