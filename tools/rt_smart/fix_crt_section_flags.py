"""Set SHF_ALLOC|SHF_EXECINSTR on crt1.o's `.start` section.

The SDK toolchain's crt1.o carries `.start` with sh_flags = 0 (neither
SHF_ALLOC nor SHF_EXECINSTR), which GNU ld tolerates but ld.lld treats as a
non-allocated section: it then assigns no address, and the PC-relative
relocations against __global_pointer$ end up ~2 GB out of range.

zig objcopy refuses to run in this sandbox (it tries to re-exec itself), so the
section header is patched directly.
"""

from __future__ import annotations

import shutil
import struct
from pathlib import Path

TC_LIB = Path(r"E:\great\dsh\k230_research\toolchain"
              r"\riscv64-linux-musleabi_for_x86_64-pc-linux-gnu"
              r"\riscv64-unknown-linux-musl\lib")
SRC = TC_LIB / "crt1.o"
DST = TC_LIB / "crt1_fixed.o"

SHF_ALLOC = 0x2
SHF_EXECINSTR = 0x4


def main() -> int:
    if not SRC.exists():
        print(f"missing {SRC}")
        return 1
    shutil.copyfile(SRC, DST)
    data = bytearray(DST.read_bytes())

    (e_shoff,) = struct.unpack_from("<Q", data, 0x28)
    (e_shentsize,) = struct.unpack_from("<H", data, 0x3A)
    (e_shnum,) = struct.unpack_from("<H", data, 0x3C)
    (e_shstrndx,) = struct.unpack_from("<H", data, 0x3E)

    strtab_off, strtab_size = struct.unpack_from("<QQ", data, e_shoff + e_shstrndx * e_shentsize + 0x18)
    strtab = bytes(data[strtab_off:strtab_off + strtab_size])

    def name_at(n: int) -> str:
        end = strtab.find(b"\x00", n)
        return strtab[n:end].decode("utf-8", "replace")

    patched = False
    for i in range(e_shnum):
        base = e_shoff + i * e_shentsize
        (sh_name,) = struct.unpack_from("<I", data, base)
        section = name_at(sh_name)
        if section == ".start":
            flags_off = base + 0x08
            (old,) = struct.unpack_from("<Q", data, flags_off)
            new = old | SHF_ALLOC | SHF_EXECINSTR
            struct.pack_into("<Q", data, flags_off, new)
            print(f"  .start: flags 0x{old:x} -> 0x{new:x}")
            patched = True

    if not patched:
        print("  .start not found")
        return 1

    DST.write_bytes(bytes(data))
    print(f"wrote {DST}  ({DST.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
