"""Read bounded diagnostic snapshots from the K230 Linux core over ADB."""

import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass

from board_resources import RESOURCE_COMMAND


DEFAULT_SERIAL = "RTT_K230_ADB"
BOARD_ADB = Path("D:/platform-tools-latest-windows/platform-tools/adb.exe")
STATUS_PATH = "/sharefs/srtp_clean/ui_overlay/status_ram/status.txt"
SECTIONS = {
    "board": (
        "head -c 64 /proc/sys/kernel/random/boot_id; "
        "head -c 64 /proc/uptime; "
        "head -c 128 /tmp/srtp_recording.status; "
        "head -c 128 /tmp/srtp_demo_last640_20261124/READY"
    ),
    "status": f"head -c 256 {STATUS_PATH}",
    "guard": "head -c 256 /tmp/fire_guard.state",
    "processes": (
        "ps w | grep -E '[f]ire_guard.sh run|[s]ensor_key|[s]tatus_bridge' "
        "| head -c 1536"
    ),
    "sensor_log": "tail -c 2048 /tmp/sensor_mvp.log",
    "guard_log": "tail -c 2048 /tmp/fire_guard.log",
    "launcher_log": "tail -c 1024 /tmp/fire_guard.out",
    "resources": RESOURCE_COMMAND,
}
MARKER = "===SRTP_BOARD_MONITOR_"
COMMAND = "; ".join(
    f"printf '\\n{MARKER}{key}===\\n'; ({command}) 2>/dev/null"
    for key, command in SECTIONS.items()
) + f"; printf '\\n{MARKER}end===\\n'"


class MonitorError(Exception):
    """ADB or board output could not provide a complete snapshot."""


@dataclass(frozen=True)
class Status:
    version: str
    sequence: int
    temperature: float | None
    humidity: float | None
    pressure: float | None
    wifi: bool
    mq2: int | None
    flame: int | None


def parse_status(raw):
    fields = raw.split()
    if not fields:
        raise ValueError("状态文件为空或不可读")
    count = {"v1": 8, "v2": 10}.get(fields[0])
    if len(fields) != count:
        raise ValueError("状态文件格式异常")

    version, seq, sht, temp, humidity, bmp, pressure, wifi = fields[:8]
    sequence = int(seq)
    if sequence < 0 or any(flag not in ("0", "1") for flag in (sht, bmp, wifi)):
        raise ValueError("状态序号或有效标志异常")

    temp, humidity, pressure = map(float, (temp, humidity, pressure))
    if not all(map(math.isfinite, (temp, humidity, pressure))):
        raise ValueError("状态数据含非有限数值")
    mq2, flame = map(int, fields[8:]) if version == "v2" else (-1, -1)
    return Status(
        version, sequence, temp if sht == "1" else None,
        humidity if sht == "1" else None, pressure if bmp == "1" else None,
        wifi == "1", mq2 if mq2 != -1 else None, flame if flame != -1 else None,
    )


def parse_snapshot(raw):
    parts = re.split(rf"(?m)^{MARKER}(\w+)===\r*$", raw)
    if parts[1::2] != [*SECTIONS, "end"]:
        raise MonitorError("ADB 返回的诊断快照不完整")
    return {key: value.strip() for key, value in zip(parts[1::2], parts[2::2])
            if key != "end"}


def find_adb():
    discovered = shutil.which("adb")
    if discovered:
        return discovered
    beside_app = Path(sys.executable).resolve().parent
    for path in (beside_app / "platform-tools" / "adb.exe", beside_app / "adb.exe", BOARD_ADB):
        if path.is_file():
            return str(path)
    return "adb"


def run_adb_shell(executable, serial, command):
    try:
        result = subprocess.run(
            [executable, "-s", serial, "shell", command],
            capture_output=True, encoding="utf-8", errors="replace", timeout=4,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except FileNotFoundError as exc:
        raise MonitorError("未找到 ADB。安装 Android Platform-Tools 后，点「选择 ADB」。") from exc
    except subprocess.TimeoutExpired as exc:
        raise MonitorError("读取超时，请检查 USB 连接和板端响应") from exc
    except OSError as exc:
        raise MonitorError(f"无法启动 adb（{exc}）") from exc
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise MonitorError(detail or f"adb 返回 {result.returncode}")
    return result.stdout


def adb_snapshot(executable, serial):
    return parse_snapshot(run_adb_shell(executable, serial, COMMAND))


class Freshness:
    """Require a changing sequence even on first connection or after reboot."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.boot = None
        self.sequence = None
        self.advanced_at = None
        self.first_seen = None

    def observe(self, boot, sequence, now):
        if self.boot != boot or self.sequence is None or sequence < self.sequence:
            self.advanced_at = None
            self.first_seen = now
        elif sequence > self.sequence:
            self.advanced_at = now
        self.boot = boot
        self.sequence = sequence
        if self.advanced_at is None:
            return "stale" if now - self.first_seen >= 6 else "waiting"
        return "fresh" if now - self.advanced_at < 6 else "stale"


def board_info(snapshot):
    lines = snapshot["board"].splitlines()
    boot = lines[0] if lines and re.fullmatch(r"[a-f0-9-]{36}", lines[0]) else ""
    release = next((line for line in lines[1:] if line.startswith(
        ("hardware-preview-", "single-owner-"))), "")
    return boot, release


def guard_info(snapshot):
    if not any(re.search(r"/fire_guard\.sh\s+run\s*$", line)
               for line in snapshot["processes"].splitlines()):
        return "守护进程未检出", "查看联动日志"
    fields = snapshot["guard"].split()
    if len(fields) != 5 or fields[0] not in ("open", "closed"):
        return "决策记录不可读", "查看联动日志"
    label = "开阀" if fields[0] == "open" else "关阀"
    return label, fields[-1]


def demo_snapshot():
    now = time.monotonic()
    mq2 = round(370 + 65 * math.sin(now / 8))
    temp = 25 + math.sin(now / 15)
    return {
        "board": "00000000-0000-0000-0000-000000000000\n"
                 "120.00 60.00\nhardware-preview-demo",
        "status": f"v2 {int(now * 2)} 1 {temp:.1f} 54.6 1 1007.8 0 {mq2} 719",
        "guard": f"closed {mq2} 719 {temp:.1f} still-elevated",
        "processes": "  42 root sh /sharefs/srtp_clean/key_controls/fire_guard.sh run",
        "sensor_log": f"SHT31: {temp:.1f} C, 54.6 %RH (CRC OK)\nMQ2: {mq2} mV",
        "guard_log": "此页为模拟数据。\n37.42 closed (gas-485) mq2=485 flame=719 temp=25.1",
        "launcher_log": "",
        "resources": (
            "MemTotal: 105876 kB\nMemAvailable: 37424 kB\n"
            "fs / /dev/root 122835 116593 2311 98% /\n"
            "fs /tmp tmpfs 52936 21000 31936 40% /tmp\n"
            "fs /sharefs tmpfs 52936 21000 31936 40% /tmp\n"
            "disk mmcblk0 MMC 30535680\ndisk mmcblk1 SD 30703616"
        ),
    }
