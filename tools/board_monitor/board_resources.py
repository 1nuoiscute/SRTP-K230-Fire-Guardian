"""Linux small-core memory, mounted filesystems and detected MMC devices."""

from dataclasses import dataclass
import re


RESOURCE_COMMAND = r"""
grep -E '^Mem(Total|Available):' /proc/meminfo
for target in / /tmp /sharefs; do
    [ -d "$target" ] || continue
    printf 'fs %s %s\n' "$target" "$(df -kP "$target" 2>/dev/null | tail -n 1)"
done
for node in /sys/block/mmcblk[0-9]*; do
    [ -r "$node/device/type" ] && [ -r "$node/size" ] || continue
    printf 'disk %s %s %s\n' "${node##*/}" "$(head -c 16 "$node/device/type")" "$(head -c 32 "$node/size")"
done | head -n 8
"""


@dataclass
class Resources:
    total_kib: int | None
    available_kib: int | None
    filesystems: list
    disks: list


def parse_resources(text):
    memory, mounts, disks = {}, [], []
    for line in text.splitlines():
        match = re.fullmatch(r"(MemTotal|MemAvailable):\s+(\d+)\s+kB", line)
        if match:
            memory[match[1]] = int(match[2])
            continue
        fields = line.split()
        if len(fields) == 8 and fields[0] == "fs" and all(x.isdecimal() for x in fields[3:6]):
            _, path, source, capacity, used, available, usage, _ = fields
            mounts.append((path, source, int(capacity), int(used), int(available), usage))
        elif (len(fields) == 4 and fields[0] == "disk" and fields[2] in ("SD", "MMC")
              and fields[-1].isdecimal()):
            _, name, kind, sectors = fields
            disks.append((name, "SD 卡" if kind == "SD" else "eMMC", int(sectors) // 2))
    total, available = memory.get("MemTotal"), memory.get("MemAvailable")
    if total is not None and available is not None and available > total:
        available = None
    return Resources(total, available, mounts, disks)


def size_text(kib):
    if kib is None:
        return "—"
    if kib >= 1024 * 1024:
        return f"{kib / (1024 * 1024):.2f} GiB"
    return f"{kib / 1024:.1f} MiB"
