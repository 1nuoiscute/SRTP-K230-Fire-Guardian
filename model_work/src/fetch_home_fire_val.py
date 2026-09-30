"""Fetch and verify the upstream Home-fire v1 validation archive only.

Uses a size cap and atomic rename. Refuses to overwrite an existing file.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import requests


URL = "https://github.com/PengBo0/Home-fire-dataset/releases/download/v1.0.0/val.zip"
EXPECTED_SIZE = 352_482_306  # GitHub release asset metadata, v1.0.0


def main(output: Path) -> None:
    if output.exists() or output.with_suffix(output.suffix + ".part").exists():
        raise SystemExit(f"Refusing existing output or partial file: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    digest = hashlib.sha256()
    size = 0
    try:
        with requests.get(URL, stream=True, timeout=(20, 60)) as response:
            response.raise_for_status()
            declared_size = int(response.headers.get("Content-Length", "0"))
            if declared_size and declared_size != EXPECTED_SIZE:
                raise ValueError(f"Unexpected server size: {declared_size}")
            with temporary.open("xb") as stream:
                for block in response.iter_content(1024 * 1024):
                    if not block:
                        continue
                    size += len(block)
                    if size > EXPECTED_SIZE:
                        raise ValueError("Download exceeded expected size")
                    digest.update(block)
                    stream.write(block)
        if size != EXPECTED_SIZE:
            raise ValueError(f"Incomplete download: {size} of {EXPECTED_SIZE}")
        temporary.rename(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"{output}\nbytes={size}\nsha256={digest.hexdigest()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args().out)
