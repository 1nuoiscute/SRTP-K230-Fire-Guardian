"""Source overlap triage for the held boat clip against v24's available v20 lineage."""
import hashlib
import json
from pathlib import Path

import cv2

from audit_video_probe_overlap import fingerprint
from source_exposure import commons_source_key, training_source_keys

ROOT = Path(__file__).resolve().parents[2]
DATA = Path("D:/SRTP_Datasets/fire_kitchen_source_v20_20261003")
MANIFEST = DATA / "build_manifest.json"
VIDEO = ROOT / "model_work/data/public_video_probe_20260930/gimbaled.webm"
OLD = ROOT / "model_work/out/public_video_probe_overlap_fullchain_20260930/local_overlap.json"
OUT = ROOT / "model_work/out/gimbaled_v24_lineage_audit_20261029"
EXPECTED = {"manifest": "15a509075be0d2ff062894cd7de1160699bbdc7b657d8199bfba77cb80bf36fa",
            "video": "bdb375cb67d9ddbc3b4521c7fb48c00e2db14a75c5fae5e1cbef7a1455f96cd8",
            "old": "c311469082b58800513bf707d683a60b5f5fee5c7ce938d05b4e060d14c08f99"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise ValueError("Refusing to overwrite output")
    for name, path in (("manifest", MANIFEST), ("video", VIDEO), ("old", OLD)):
        if sha(path) != EXPECTED[name]:
            raise ValueError(f"{name} identity mismatch")
    manifest = json.loads(MANIFEST.read_text(encoding="utf8"))
    rows = manifest["lineage"]
    if len(rows) != 1445 or len({row["sha256"] for row in rows}) != 1445:
        raise ValueError("Unexpected v24 lineage inventory")
    old = json.loads(OLD.read_text(encoding="utf8"))
    historical = {row["frame_index"]: row for row in old["frames"] if row["clip"] == "gimbaled"}
    if len(historical) != 140:
        raise ValueError("Historical sample count mismatch")
    cap = cv2.VideoCapture(str(VIDEO))
    decoded, frames = 0, []
    while True:
        ok, image = cap.read()
        if not ok:
            break
        if decoded in historical:
            ph, pixel = fingerprint(image)
            previous = historical[decoded]
            if ph != previous["phash"] or pixel != previous["pixel_sha256"]:
                raise ValueError("Historical sampled frame changed")
            frames.append({"frame_index": decoded, "phash": ph, "pixel_sha256": pixel,
                           "nearest": [], "exact_matches": []})
        decoded += 1
    cap.release()
    if decoded != 838 or len(frames) != 140 or sha(VIDEO) != EXPECTED["video"]:
        raise ValueError("Unexpected video decoding or changed source")
    for row in rows:
        path = DATA / "images/train" / row["image"]
        if not path.is_file() or sha(path) != row["sha256"]:
            raise ValueError(f"Missing or changed lineage image: {path}")
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Unreadable lineage image: {path}")
        ph, pixel = fingerprint(image)
        for frame in frames:
            entry = [(frame["phash"] ^ ph).bit_count(), row["image"], row["sha256"]]
            frame["nearest"] = sorted(frame["nearest"] + [entry])[:3]
            if pixel == frame["pixel_sha256"]:
                frame["exact_matches"].append(entry)
    for name, path in (("manifest", MANIFEST), ("video", VIDEO), ("old", OLD)):
        if sha(path) != EXPECTED[name]:
            raise ValueError(f"{name} changed during audit")
    source = commons_source_key("https://commons.wikimedia.org/wiki/File:Gimbaled_stove.WebM")
    summary = {"role": "Held source overlap triage; no inference, no independence conclusion",
               "inputs_sha256": EXPECTED, "script_sha256": sha(Path(__file__)),
               "lineage_images_verified": len(rows), "sampled_frames": len(frames),
               "sample_identity_matches_earlier_audit": True,
               "original_work_seen_in_v20_source_keys": source in training_source_keys(manifest),
               "frames_with_exact_decoded_pixel_match": sum(bool(frame["exact_matches"]) for frame in frames),
               "minimum_phash_distance": min(frame["nearest"][0][0] for frame in frames),
               "limitations": ["v20 lineage only; generic pretraining and unavailable sources are not covered",
                               "pixel equality misses crops and lossy changes; pHash is only a review lead",
                               "same author, recording event or kitchen cannot be ruled out by hashes",
                               "other prediction exposure history has not been completely audited"]}
    OUT.mkdir(parents=True)
    (OUT / "per_frame_local.json").write_text(json.dumps(frames, indent=2) + "\n", encoding="utf8")
    summary["frame_records_sha256"] = sha(OUT / "per_frame_local.json")
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
