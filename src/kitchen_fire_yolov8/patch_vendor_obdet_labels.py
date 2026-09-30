#!/usr/bin/env python3
"""Patch only four fixed COCO label slots in a copy of vendor ob_det.elf."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


PATCHES = {
    b"motorcycle\0": b"fire\0",
    b"traffic light\0": b"smoke\0",
    b"fire hydrant\0": b"stove\0",
    b"parking meter\0": b"pan\0",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    original = args.input.read_bytes()
    patched = bytearray(original)
    print(f"source_sha256={sha256(original)}")

    for old, new in PATCHES.items():
        count = original.count(old)
        if count != 1:
            raise SystemExit(f"Refusing to patch {old!r}: expected once, found {count}")
        offset = original.index(old)
        # The executable renders these as fixed-width strings.  NUL padding
        # appears as question marks in OpenCV text, so pad with spaces.
        replacement = new[:-1].ljust(len(old) - 1, b" ") + b"\0"
        patched[offset : offset + len(old)] = replacement
        print(f"0x{offset:X}: {old[:-1].decode()} -> {new[:-1].decode()}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(patched)
    print(f"output_sha256={sha256(patched)}")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
