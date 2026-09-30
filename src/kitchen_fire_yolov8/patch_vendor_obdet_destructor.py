#!/usr/bin/env python3
"""Create a diagnostic copy of the kit's ob_det.elf without one bad free call.

The original binary is preserved.  This patch is specific to the backed-up
December 2024 kit executable and must not be applied to another ELF.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ORIGINAL_SHA256 = "3bdf55ae5b6723be5887167d1ee6319686cfaf7b787d65160abc275597e24334"
CALL_OFFSET = 0x11B8A
EXPECTED_CALL = bytes.fromhex("97708000e780602e")
TWO_RISCV_NOPS = bytes.fromhex("1300000013000000")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    original = args.source.read_bytes()
    actual_hash = hashlib.sha256(original).hexdigest()
    if actual_hash != ORIGINAL_SHA256:
        raise SystemExit(f"Unexpected source SHA-256: {actual_hash}")
    if original[CALL_OFFSET:CALL_OFFSET + len(EXPECTED_CALL)] != EXPECTED_CALL:
        raise SystemExit("Destructor call bytes differ; refusing to patch")

    patched = bytearray(original)
    patched[CALL_OFFSET:CALL_OFFSET + len(EXPECTED_CALL)] = TWO_RISCV_NOPS
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}")
    args.output.write_bytes(patched)
    print(f"patched {args.output}, sha256={hashlib.sha256(patched).hexdigest()}")


if __name__ == "__main__":
    main()
