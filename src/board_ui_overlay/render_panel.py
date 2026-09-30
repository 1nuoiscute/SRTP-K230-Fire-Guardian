"""Render a bounded, PC-tethered K230 OSD panel from read-only ADB status.

The board's clock is invalid, so freshness is checked by log growth using PC
time. A generated panel is a development bridge, not a standalone board UI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


WIDTH, HEIGHT = 536, 348
ROOT = Path(__file__).resolve().parent
DEFAULT_ADB = Path(r"D:\platform-tools-latest-windows\platform-tools\adb.exe")
DEFAULT_EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
SENSOR_RE = re.compile(r"SHT31:\s*([\d.]+) C,\s*([\d.]+) %RH \(CRC OK\)")
PRESSURE_RE = re.compile(r"BMP280:\s*([\d.]+) C,\s*([\d.]+) hPa")


def adb_shell(adb: Path, command: str) -> str:
    result = subprocess.run([str(adb), "shell", command], check=True, capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=12)
    return result.stdout.replace("\r", "")


def read_status(adb: Path) -> dict:
    output = adb_shell(adb, "wc -c /tmp/sensor_mvp.log; sleep 1; wc -c /tmp/sensor_mvp.log; tail -n 8 /tmp/sensor_mvp.log")
    sizes = [int(m.group(1)) for m in re.finditer(r"(?m)^(\d+) /tmp/sensor_mvp\.log$", output)]
    fresh = len(sizes) >= 2 and sizes[-1] > sizes[-2]
    sht = SENSOR_RE.findall(output)
    bmp = PRESSURE_RE.findall(output)
    net = adb_shell(adb, "cat /sys/class/net/wlan0/operstate; ip -o -4 addr show wlan0")
    net_lines = net.splitlines()
    wifi_has_addr = bool(net_lines and net_lines[0].strip() == "up" and "inet " in net)
    return {
        "captured_at_pc": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "sensor_log_grew_in_1s": fresh,
        "temperature_c": round(float(sht[-1][0]), 1) if fresh and sht else None,
        "humidity_pct": round(float(sht[-1][1]), 1) if fresh and sht else None,
        "pressure_hpa": round(float(bmp[-1][1]), 1) if fresh and bmp else None,
        "wifi_mainboard": "已获取地址" if wifi_has_addr else "未连接",
        "wifi_source": "Linux wlan0; RW007 not measured",
        "fire_judgment": "暂未判定",
    }


def render(status: dict, edge: Path, out_dir: Path) -> tuple[Path, Path]:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    template = (ROOT / "panel_template.html").read_text(encoding="utf-8")
    replacements = {
        "WIFI": status["wifi_mainboard"],
        "WIFI_COLOR": "#587d61" if status["wifi_mainboard"] == "已获取地址" else "#a87946",
        "TEMP": "--" if status["temperature_c"] is None else f'{status["temperature_c"]:.1f}',
        "HUMIDITY": "--" if status["humidity_pct"] is None else f'{status["humidity_pct"]:.1f}',
        "PRESSURE": "--" if status["pressure_hpa"] is None else f'{status["pressure_hpa"]:.1f}',
    }
    for key, value in replacements.items():
        template = template.replace("{{" + key + "}}", value)
    if "{{" in template:
        raise ValueError("Unfilled panel template token")
    html_path = out_dir / "panel_render.html"
    png_path = out_dir / "panel.png"
    raw_path = out_dir / "panel.bgra"
    html_path.write_text(template, encoding="utf-8")
    profile = Path(tempfile.gettempdir()) / "srtp-board-ui-edge"
    subprocess.run([str(edge), "--headless", "--disable-gpu", "--no-first-run",
                    f"--user-data-dir={profile}", f"--window-size={WIDTH},{HEIGHT}",
                    f"--screenshot={png_path}", html_path.as_uri()],
                   check=True, capture_output=True, timeout=20)
    with Image.open(png_path) as image:
        if image.width < WIDTH or image.height < HEIGHT:
            raise ValueError(f"Edge screenshot too small: {image.size}")
        panel = image.convert("RGBA").crop((0, 0, WIDTH, HEIGHT))
        raw_path.write_bytes(panel.tobytes("raw", "BGRA"))
    if raw_path.stat().st_size != WIDTH * HEIGHT * 4:
        raise AssertionError("Wrong OSD byte count")
    status["panel_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    (out_dir / "status_snapshot.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    return png_path, raw_path


def push(adb: Path, raw_path: Path) -> None:
    board_dir = "/sharefs/srtp_clean/ui_overlay"
    subprocess.run([str(adb), "shell", f"mkdir -p {board_dir}"], check=True, timeout=10)
    subprocess.run([str(adb), "push", str(raw_path), f"{board_dir}/panel.next.bgra"],
                   check=True, timeout=20)
    subprocess.run([str(adb), "shell", f"mv {board_dir}/panel.next.bgra {board_dir}/panel.bgra"],
                   check=True, timeout=10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adb", type=Path, default=DEFAULT_ADB)
    parser.add_argument("--edge", type=Path, default=DEFAULT_EDGE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    status = read_status(args.adb)
    png_path, raw_path = render(status, args.edge, args.out)
    if args.push:
        push(args.adb, raw_path)
    print(json.dumps({"status": status, "png": str(png_path), "bgra": str(raw_path),
                      "pushed": args.push}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
