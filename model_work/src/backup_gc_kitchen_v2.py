"""Copy the GC kitchen raw export to C: and verify each copied file by SHA-256.

The source is read-only. The destination must not already exist. No cleanup is
performed on failure so a partial copy remains visible for inspection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main(source: Path, destination: Path) -> None:
    if not source.is_dir() or destination.exists():
        raise SystemExit("Source missing or destination already exists")
    workspace = Path(r"C:\Users\ASUS\Documents\ChatGPT\SRTP").resolve()
    if not destination.resolve().is_relative_to(workspace):
        raise SystemExit("Destination must stay inside the SRTP workspace")
    files = sorted(path for path in source.rglob("*") if path.is_file())
    if not files:
        raise SystemExit("Source contains no files")
    destination.mkdir(parents=True)
    manifest = []
    for original in files:
        relative = original.relative_to(source)
        copied = destination / relative
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, copied)
        source_hash = sha256(original)
        target_hash = sha256(copied)
        if source_hash != target_hash:
            raise RuntimeError(f"Backup mismatch: {relative}")
        manifest.append({"path": relative.as_posix(), "bytes": original.stat().st_size, "sha256": source_hash})
    metadata = {
        "source": str(source.resolve()),
        "backup": str(destination.resolve()),
        "file_count": len(manifest),
        "total_bytes": sum(item["bytes"] for item in manifest),
        "verification": "every copied file SHA-256 matches its source",
        "files": manifest,
    }
    (destination / "MANIFEST_SHA256.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: metadata[key] for key in ("source", "backup", "file_count", "total_bytes", "verification")}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    main(args.source, args.destination)
