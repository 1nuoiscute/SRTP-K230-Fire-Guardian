"""Replace only the stock fastboot ELF bytes in a verified K230 partition backup.

The ROMFS file table and every other file remain byte-for-byte unchanged after
decompression. The replacement ELF is padded to the stock file's exact length,
so no ROMFS offsets or file-size fields need to move.
"""

import argparse
import gzip
import hashlib
import pathlib
import struct

from package_rtt_uimage import HEADER, PARTITION_BYTES, build_uimage, inspect_original


BACKUP_SHA256 = "83f8286ae132a917c72c15cff004f695d35de1a9194024cf93f012b1532f9cf6"
STOCK_SHA256 = "025e795058fe7678dfc18cdf6bfb0780d403060d42065d9dbe4ab2525e15deec"
KEY_SHA256 = "d50aa814db7df2ea4cd0cc19b70b82be2716f9ff43997d081effd618b2cf04e7"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def wrap_partition(uimage: bytes) -> bytes:
    # SDK firmware_gen.py -n adds a four-byte version to its input, then writes
    # K230 magic, little-endian length/type, SHA-256 and a reserved zero area.
    payload = bytes(4) + uimage
    header = b"K230" + struct.pack("<II", len(payload), 0)
    header += hashlib.sha256(payload).digest() + bytes(516 - 32)
    firmware = header + payload
    if len(firmware) > PARTITION_BYTES:
        raise ValueError("firmware exceeds the 20 MiB partition")
    return firmware + bytes(PARTITION_BYTES - len(firmware))


def unpack_uimage(uimage: bytes) -> bytes:
    gz = uimage[HEADER.size + 8 :]
    if gz[:3] != b"\x1f\x8b\x09":
        raise ValueError("unexpected K230 gzip signature")
    return gzip.decompress(gz[:2] + b"\x08" + gz[3:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backup", type=pathlib.Path, required=True)
    parser.add_argument("--stock-elf", type=pathlib.Path, required=True)
    parser.add_argument("--key-elf", type=pathlib.Path, required=True)
    parser.add_argument("--key-sha256", default=KEY_SHA256,
                        help="expected SHA-256 of the replacement K1 ELF")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    backup = args.backup.read_bytes()
    stock = args.stock_elf.read_bytes()
    key = args.key_elf.read_bytes()
    for name, data, expected in (
        ("original partition", backup, BACKUP_SHA256),
        ("stock fastboot ELF", stock, STOCK_SHA256),
        ("K1 ELF", key, args.key_sha256.lower()),
    ):
        if sha256(data) != expected:
            raise ValueError(f"{name} SHA-256 mismatch")
    if len(key) > len(stock):
        raise ValueError("replacement does not fit stock ROMFS file slot")

    original_fields, original_uimage = inspect_original(backup)
    if wrap_partition(original_uimage) != backup:
        raise ValueError("repack does not reproduce the original partition exactly")

    original_raw = unpack_uimage(original_uimage)
    offset = original_raw.find(stock)
    if offset < 0 or original_raw.find(stock, offset + 1) >= 0:
        raise ValueError("stock ELF is missing or appears more than once")
    replacement = key + bytes(len(stock) - len(key))
    patched_raw = original_raw[:offset] + replacement + original_raw[offset + len(stock) :]
    if len(patched_raw) != len(original_raw):
        raise ValueError("ROMFS size changed")

    new_gzip = gzip.compress(patched_raw, compresslevel=9, mtime=0)
    new_uimage = build_uimage(original_fields, new_gzip)
    candidate = wrap_partition(new_uimage)
    _, checked_uimage = inspect_original(candidate)
    if unpack_uimage(checked_uimage) != patched_raw:
        raise ValueError("candidate round-trip failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(candidate)

    print(f"stock ELF offset in decompressed ROMFS: {offset}")
    print(f"changed region: {len(stock)} bytes; all other bytes unchanged")
    print(f"candidate size: {len(candidate)} bytes")
    print(f"candidate SHA-256: {sha256(candidate)}")


if __name__ == "__main__":
    main()
