"""Two-stage build of an RT-Smart ELF: Zig compiles, ld.lld links.

Zig's `cc` driver insists on injecting its own musl CRT and compiler_rt, both of
which are built with the medlow code model and therefore cannot be placed at
RT-Smart's 0x200000000 base (a `lui` immediate cannot encode an address that
high). Zig also ships a standalone `ld.lld`, so the build is split:

  stage 1: zig cc -c        compile each .c with -mcmodel=medany (works fine)
  stage 2: zig ld.lld       link against the SDK toolchain's medany CRT,
                            libc.a, libgcc.a and the MPP static libraries

Nothing from Zig's runtime ends up in the output.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

WS = Path(r"E:\great\dsh")
ZIG = WS / "k230_research" / "toolchain" / "zig-windows-x86_64-0.13.0" / "zig.exe"
TC = WS / "k230_research" / "toolchain" / "riscv64-linux-musleabi_for_x86_64-pc-linux-gnu"
SYSROOT = TC / "riscv64-unknown-linux-musl"
SYS_LIB = SYSROOT / "lib"
LIBGCC = TC / "lib" / "gcc" / "riscv64-unknown-linux-musl" / "12.0.1"

SDK = WS / "k230_research" / "sdk_slice" / "mpp"
MPP_LIB = SDK / "userapps" / "lib"
LDS = SDK / "userapps" / "sample" / "linker_scripts" / "riscv64" / "link_rt.lds"
UI_DIR = WS / "fire-guardian" / "src" / "board_ui_overlay"
YOLO_DIR = WS / "fire-guardian" / "src" / "kitchen_fire_yolov8"
OBJ_DIR = WS / "k230_research" / "sensor_gas" / "obj"

EXCLUDE_LIBS = {"vg_lite", "vg_lite_util"}

CFLAGS = [
    "-target", "riscv64-linux-musl",
    "-mcpu=generic_rv64+a+c+d+f+m",
    "-mcmodel=medany",
    "-fno-pie",
    "-std=c11", "-O2",
]

INCLUDES = [
    "-isystem", str(SYSROOT / "include"),
    "-I", str(SDK / "include"),
    "-I", str(SDK / "include" / "comm"),
    "-I", str(SDK / "include" / "ioctl"),
    "-I", str(SDK / "userapps" / "api"),
    "-I", str(YOLO_DIR),
    "-I", str(UI_DIR),
]


def env() -> dict:
    e = dict(os.environ)
    e["ZIG_GLOBAL_CACHE_DIR"] = str(WS / "k230_research" / "toolchain" / "zig-cache-global")
    e["ZIG_LOCAL_CACHE_DIR"] = str(WS / "k230_research" / "toolchain" / "zig-cache-local")
    return e


def compile_one(src: Path, defines: list[str]) -> Path | None:
    OBJ_DIR.mkdir(parents=True, exist_ok=True)
    obj = OBJ_DIR / (src.stem + ".o")
    cmd = [str(ZIG), "cc", *CFLAGS, *[f"-D{d}" for d in defines], *INCLUDES,
           "-c", str(src), "-o", str(obj)]
    print(f"  cc {src.name}")
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env(), timeout=600)
    if r.returncode != 0:
        print(f"    FAILED\n{r.stderr[:2500]}")
        return None
    if r.stderr.strip():
        # surface real diagnostics but keep the log readable
        for line in r.stderr.splitlines():
            if "error" in line.lower():
                print(f"    {line[:160]}")
    return obj


def link(objects: list[Path], out: Path) -> int:
    libs = [f"-l{p.stem[3:]}" for p in sorted(MPP_LIB.glob("lib*.a"))
            if p.stem[3:] not in EXCLUDE_LIBS]
    cmd = [
        str(ZIG), "ld.lld",
        "-o", str(out),
        "-static", "--no-dynamic-linker", "-T", str(LDS),
        "-L", str(MPP_LIB),
        "-L", str(SYS_LIB),
        "-L", str(LIBGCC),
        str(SYS_LIB / "crt1_fixed.o"),
        str(SYS_LIB / "crti.o"),
        *[str(o) for o in objects],
        "--start-group", *libs, "-lc", "-lgcc", "--end-group",
        "-lpthread",
        str(SYS_LIB / "crtn.o"),
    ]
    print("=== link ===")
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env(), timeout=900)
    if r.stdout.strip():
        print(r.stdout[:2000])
    if r.stderr.strip():
        print(r.stderr[:5000])
    if r.returncode == 0 and out.exists():
        print(f"  OK {out.name}  {out.stat().st_size:,} bytes")
        return 0
    print(f"  FAILED exit={r.returncode}")
    return 1


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: build_rt_two_stage.py <out.elf> <src.c> [src.c ...] [-DDEFINE ...]")
        return 2
    out = Path(sys.argv[1])
    sources, defines = [], []
    for a in sys.argv[2:]:
        if a.startswith("-D"):
            defines.append(a[2:])
        else:
            p = Path(a)
            sources.append(p if p.is_absolute() else UI_DIR / a)

    print(f"=== compile {len(sources)} source(s) ===")
    objects = []
    for s in sources:
        o = compile_one(s, defines)
        if o is None:
            return 1
        objects.append(o)
    return link(objects, out)


if __name__ == "__main__":
    sys.exit(main())





