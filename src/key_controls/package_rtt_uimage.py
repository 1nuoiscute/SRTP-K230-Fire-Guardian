"""Build a K230 RT-Smart legacy multi-image from a verified partition backup.

The SDK's mkimage host tool is unavailable in this workspace. This script
recreates its 64-byte legacy header and checks that doing so exactly reproduces
the header already on the board. K230's outer header is added separately with
the unmodified SDK firmware_gen.py.
"""

import argparse
import gzip
import pathlib
import struct
import zlib


HEADER = struct.Struct(">7I4B32s")
MAGIC = 0x27051956
UIMAGE_OFFSET = 532
PARTITION_BYTES = 20 * 1024 * 1024


def crc(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


def inspect_original(partition: bytes) -> tuple[tuple, bytes]:
    if len(partition) != PARTITION_BYTES or partition[:4] != b"K230":
        raise ValueError("not the expected 20 MiB K230 RT-Smart partition")
    header = partition[UIMAGE_OFFSET : UIMAGE_OFFSET + HEADER.size]
    fields = HEADER.unpack(header)
    magic, header_crc, timestamp, size, load, entry, data_crc, os_id, arch, image_type, compression, name = fields
    if magic != MAGIC or (os_id, arch, image_type, compression) != (27, 26, 4, 1):
        raise ValueError("unexpected U-Boot multi-image header")
    if crc(header[:4] + bytes(4) + header[8:]) != header_crc:
        raise ValueError("original U-Boot header CRC failed")
    data = partition[UIMAGE_OFFSET + HEADER.size : UIMAGE_OFFSET + HEADER.size + size]
    if crc(data) != data_crc:
        raise ValueError("original U-Boot payload CRC failed")
    item_size, terminator = struct.unpack(">II", data[:8])
    if terminator != 0 or item_size != len(data) - 8 or data[8:11] != b"\x1f\x8b\x09":
        raise ValueError("unexpected original multi-image layout")
    gzip.decompress(data[8:10] + b"\x08" + data[11:])
    return fields, header + data


def build_uimage(original: tuple, gz: bytes) -> bytes:
    if gz[:2] != b"\x1f\x8b" or gz[2] not in (8, 9):
        raise ValueError("not a K230-compatible gzip payload")
    gz = gz[:2] + b"\x09" + gz[3:]
    gzip.decompress(gz[:2] + b"\x08" + gz[3:])
    data = struct.pack(">II", len(gz), 0) + gz
    magic, _, timestamp, _, load, entry, _, os_id, arch, image_type, compression, name = original
    header = HEADER.pack(magic, 0, timestamp, len(data), load, entry,
                         crc(data), os_id, arch, image_type, compression, name)
    header = HEADER.pack(magic, crc(header), timestamp, len(data), load, entry,
                         crc(data), os_id, arch, image_type, compression, name)
    return header + data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backup", type=pathlib.Path, required=True)
    parser.add_argument("--gzip", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    original, original_uimage = inspect_original(args.backup.read_bytes())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "original.uimg").write_bytes(original_uimage)
    new_uimage = build_uimage(original, args.gzip.read_bytes())
    (args.output_dir / "k1_autostart.uimg").write_bytes(new_uimage)
    print(f"original U-Boot image verified: {len(original_uimage)} bytes")
    print(f"new U-Boot image built: {len(new_uimage)} bytes")


if __name__ == "__main__":
    main()
