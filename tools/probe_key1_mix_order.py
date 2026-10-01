"""Bounded K230 OSD mix-order probe; default is read-only.

Only applies to the frozen KEY2 firmware. The board shell owns its restore
trap/timer, so the restore does not depend on a host-side sleep loop. It never
stops the vision consumer, changes firmware, or changes GPIO.
"""
from __future__ import annotations
import argparse
import datetime
import json
from pathlib import Path
import re
import subprocess

P1_SHA = "5cd8667b3035199fcc63d7e557c8257247bc0080f2b9cda7e3a916a3616ef1d2"
ORDER_LOW = 0x908403CC
ORDER_HIGH = 0x90840950
ENABLE = 0x90840118
UPDATE = 0x90840004
DEFAULT_LOW = 0x3210
DEFAULT_HIGH = 0xBA987654
TRIAL_HIGH = 0xBA984657  # source4 OSD0 <-> source7 OSD3, all other sources kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adb", type=Path, required=True)
    parser.add_argument("--serial", default="RTT_K230_ADB")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--seconds", type=int, default=30)
    args = parser.parse_args()
    if not 5 <= args.seconds <= 60:
        parser.error("seconds must be 5..60")
    if args.output.exists():
        raise FileExistsError(args.output)
    log = []

    def shell(script: str, timeout: int = 20) -> str:
        result = subprocess.run([str(args.adb), "-s", args.serial, "shell", script],
                                capture_output=True, text=True, timeout=timeout)
        log.append({"script": script, "returncode": result.returncode,
                    "stdout": result.stdout, "stderr": result.stderr})
        if result.returncode:
            raise RuntimeError(result.stderr or result.stdout)
        return result.stdout

    def read32(address: int) -> int:
        output = shell(f"devmem 0x{address:08x} 32")
        match = re.fullmatch(r"\s*0x([0-9A-Fa-f]{8})\s*", output)
        if not match:
            raise ValueError(f"Unexpected register response: {output!r}")
        return int(match.group(1), 16)

    receipt = {"started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "mode": "apply" if args.apply else "read-only", "device": args.serial}
    try:
        actual_sha = shell("sha256sum /dev/mmcblk0p1").split()[0]
        receipt["p1_sha256"] = actual_sha
        if actual_sha != P1_SHA:
            raise ValueError("Different boot baseline; refusing this profile")
        values = {f"0x{x:08x}": read32(x) for x in (ORDER_LOW, ORDER_HIGH, ENABLE)}
        receipt["before"] = values
        if args.apply:
            if values[f"0x{ORDER_LOW:08x}"] != DEFAULT_LOW or values[f"0x{ORDER_HIGH:08x}"] != DEFAULT_HIGH:
                raise ValueError("Unexpected mix order or display clock off; no write")
            if values[f"0x{ENABLE:08x}"] & 0x90 != 0x90:
                raise ValueError("Both OSD0 panel and OSD3 must be visible; no write")
            # Recheck inside the board shell immediately before writing. The
            # trap is registered first. If a mode switch changes the order or
            # gates the clock, cleanup leaves the driver-owned value intact.
            script = f"""set -e
cleanup() {{
 trap - EXIT HUP INT TERM
 current=$(devmem 0x{ORDER_HIGH:08x} 32)
 if [ "$current" = "0x{TRIAL_HIGH:08X}" ]; then
  devmem 0x{ORDER_HIGH:08x} 32 0x{DEFAULT_HIGH:08x}
  devmem 0x{UPDATE:08x} 32 0x11
  printf 'RESTORED='
  devmem 0x{ORDER_HIGH:08x} 32
 else
  printf 'RESTORE_SKIPPED_DRIVER_CHANGED=%s\n' "$current"
 fi
}}
test "$(devmem 0x{ORDER_LOW:08x} 32)" = "0x{DEFAULT_LOW:08X}"
test "$(devmem 0x{ORDER_HIGH:08x} 32)" = "0x{DEFAULT_HIGH:08X}"
enabled=$(devmem 0x{ENABLE:08x} 32)
test $((enabled & 144)) -eq 144
trap cleanup EXIT HUP INT TERM
devmem 0x{ORDER_HIGH:08x} 32 0x{TRIAL_HIGH:08x}
devmem 0x{UPDATE:08x} 32 0x11
printf 'APPLIED='
devmem 0x{ORDER_HIGH:08x} 32
sleep {args.seconds}
"""
            output = shell(script, timeout=args.seconds + 20)
            receipt["trial_output"] = output
            receipt["after"] = {f"0x{x:08x}": read32(x) for x in (ORDER_LOW, ORDER_HIGH, ENABLE)}
            receipt["result"] = "bounded trial completed; screen confirmation separate"
        else:
            receipt["result"] = "read-only snapshot, no register write"
    except Exception as error:
        receipt["error"] = str(error)
        raise
    finally:
        receipt["commands"] = log
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in receipt.items() if k != "commands"}, indent=2))


if __name__ == "__main__":
    main()
