"""Read K230 chip temperature through the existing vendor kernel driver."""

from dataclasses import dataclass
import re

from board_data import MonitorError, run_adb_shell


# This vendor driver's temp attribute contains the TS status/data register.
COMMAND = r"""
export LC_ALL=C
printf 'boot %s\n' "$(head -c 64 /proc/sys/kernel/random/boot_id 2>/dev/null)"
for node in /sys/class/thermal/thermal_zone*; do
    [ "$(head -c 64 "$node/type" 2>/dev/null)" = "canaan_thermal_zone" ] || continue
    value=$(head -c 32 "$node/temp" 2>/dev/null)
    printf 'ts %s %s\n' "$value" "${node##*/}"
    break
done
printf 'SRTP_HW_DONE\n'
"""


@dataclass
class HardwareSnapshot:
    boot: str = ""
    temperature: float | None = None
    thermal_source: str = ""


def canaan_temperature(raw):
    if not 4096 <= raw < 8192:
        return None
    code = raw & 0xfff
    value = (((1.01472e-10 * code - 1.10063e-6) * code + 4.36150e-3)
             * code - 7.10128) * code + 3565.87
    return value if -60 <= value <= 160 else None


def parse_hardware(text):
    result = HardwareSnapshot()
    for line in text.splitlines():
        if line.startswith("boot "):
            candidate = line[5:]
            if re.fullmatch(r"[0-9a-f-]{36}", candidate):
                result.boot = candidate
        elif match := re.fullmatch(r"ts ([0-9]{1,9}) (thermal_zone[0-9]+)", line):
            raw = int(match[1])
            result.temperature = canaan_temperature(raw)
            result.thermal_source = f"{match[2]}/temp · 原始码 {raw}"
    return result


def read_chip_temperature(adb, serial):
    result = run_adb_shell(adb, serial, COMMAND).strip()
    if not result.endswith("\nSRTP_HW_DONE"):
        raise MonitorError("硬件快照未完整返回")
    return result.rsplit("\n", 1)[0]
