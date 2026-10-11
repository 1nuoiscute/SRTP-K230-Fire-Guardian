"""Acquire the officially linked public D-Fire mirror v1, never train automatically.

No credentials/login. New directory only, bounded size/time, partials retained,
upstream snapshots + local checksums. The mirror's splits are development inputs.
"""
import argparse
import hashlib
import json
import shutil
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
import requests
from safe_dataset_zip import extract_new_dataset
from data_integrity import sha256

REF = "sayedgamal99/smoke-fire-detection-yolo"
COMMIT = "4bf9c31b18fadcd44d5f0b6d66f82bc56fa5e328"
EXPECTED_BYTES = 3049605157
EXPECTED_ETAG = '"9745236834d65441af996cb6da6a4fde"'


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-download-seconds", type=int, default=1800)
    a = p.parse_args(); out = a.out.resolve()
    if out.exists(): raise SystemExit("Existing acquisition directory; inspect instead of restarting")
    if not 60 <= a.max_download_seconds <= 3600: p.error("Invalid duration bound")
    if shutil.disk_usage(out.anchor).free < 12_000_000_000:
        raise SystemExit("Need at least 12 GB free on acquisition drive")
    out.mkdir(parents=True)
    marker = out / "acquisition.json"
    record = {"role": "unreviewed source acquisition, no training or independent-test claim",
              "started_utc": datetime.now(timezone.utc).isoformat(), "dataset_ref": REF, "mirror_version": 1,
              "official_commit": COMMIT, "script_sha256": sha256(Path(__file__)),
              "safe_zip_script_sha256": sha256(Path(__file__).with_name("safe_dataset_zip.py")),
              "expected_archive_bytes": EXPECTED_BYTES, "max_download_seconds": a.max_download_seconds,
              "status": "started"}
    def save(): marker.write_text(json.dumps(record, indent=2), encoding="utf-8")
    save()
    try:
        for name in ("README.md", "LICENSE"):
            url = f"https://raw.githubusercontent.com/gaia-solutions-on-demand/DFireDataset/{COMMIT}/{name}"
            response = requests.get(url, timeout=(10, 30)); response.raise_for_status()
            if len(response.content) > 1_000_000: raise ValueError("Upstream snapshot unexpectedly large")
            target = out / ("upstream_" + name); target.write_bytes(response.content)
            record["upstream_" + name + "_sha256"] = sha256(target)
        metadata_url = f"https://www.kaggle.com/api/v1/datasets/view/{REF}"
        response = requests.get(metadata_url, timeout=(10, 30)); response.raise_for_status()
        metadata = response.json()
        if metadata.get("id") != 6556263 or metadata.get("ref") != REF or metadata.get("currentVersionNumber") != 1 or metadata.get("isPrivate") is not False:
            raise ValueError("Mirror identity/version/access changed")
        if metadata.get("licenseName") != "CC0: Public Domain" or metadata.get("totalBytes") != 3118334483:
            raise ValueError("Mirror license/size declaration changed")
        (out / "mirror_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        record["mirror_metadata_sha256"] = sha256(out / "mirror_metadata.json")
        record["license_scope"] = "Publisher declares collection CC0 and disclaims owning collected image copyrights; no per-image rights verification"
        record["download_url"] = f"https://www.kaggle.com/api/v1/datasets/download/{REF}?datasetVersionNumber=1"
        record.update(status="downloading", bytes_received=0); save()
        part = out / "dataset.zip.part"; total = 0; last_report = time.monotonic(); started = last_report
        digest = hashlib.sha256(); md5 = hashlib.md5()
        with requests.get(record["download_url"], stream=True, timeout=(10, 45)) as response:
            response.raise_for_status()
            record["download_final_host"] = urlsplit(response.url).netloc
            if response.headers.get("content-type", "").split(";")[0] != "application/zip" or int(response.headers.get("content-length", "0")) != EXPECTED_BYTES or response.headers.get("etag") != EXPECTED_ETAG:
                raise ValueError("Archive response identity/size/type changed")
            record["etag"] = response.headers["etag"]; save()
            with part.open("xb") as stream:
                for block in response.iter_content(chunk_size=1 << 20):
                    if not block: continue
                    total += len(block)
                    if total > EXPECTED_BYTES or time.monotonic() - started > a.max_download_seconds:
                        raise ValueError("Download size/time bound exceeded")
                    stream.write(block); digest.update(block); md5.update(block)
                    if time.monotonic() - last_report >= 10:
                        record.update(bytes_received=total, elapsed_download_seconds=time.monotonic()-started); save()
                        print(f"Downloaded {total}/{EXPECTED_BYTES} bytes", flush=True); last_report=time.monotonic()
        if total != EXPECTED_BYTES or md5.hexdigest() != EXPECTED_ETAG.strip('"'):
            raise ValueError("Archive received length/ETag MD5 check differs")
        complete = out / "dataset.zip"; part.rename(complete)
        record.update(status="extracting", bytes_received=total, archive_sha256=digest.hexdigest(), archive_md5=md5.hexdigest(),
                      elapsed_download_seconds=time.monotonic()-started); save()
        with zipfile.ZipFile(complete) as archive:
            record["extraction"] = extract_new_dataset(archive, out / "raw")
        record.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                      integrity="Expected received length, object ETag MD5, archive SHA-256 and all extracted ZIP CRCs verified; local SHA is not a publisher-issued SHA")
        save(); print(json.dumps({k: record[k] for k in ("status", "archive_sha256", "bytes_received", "extraction")}), flush=True)
    except Exception as error:
        record.update(status="failed", error_type=type(error).__name__, failed_utc=datetime.now(timezone.utc).isoformat())
        # Keep diagnostics/partials, but never log signed redirect URLs or credentials.
        save(); raise SystemExit(f"Acquisition failed ({type(error).__name__}); inspect acquisition.json and retained partials") from None


if __name__ == "__main__": main()
