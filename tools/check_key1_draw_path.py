"""Read-only static display-path scan of the frozen K230 visual ELF.

This is a bounded static scan, not a runtime trace. Computed targets and
dynamically created pointers are not resolved.
The SDK objdump is required; executable sections can also contain data.
The proprietary binary stays local; only identities and numeric findings emit.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
BASELINE_SHA = "ec2caafd85e56851c44c838b4c9a2a15477cae8b24ac915f1c571f1a0f9a2ac1"
PROFILES = {
    "hardware_draw": (0xc01857b7, 0xf2c78593, "397106fc22f880002334a4fc", 0xc0184f2c),
    "osd_insert": (0xc0e857b7, 0xf4078593, "6d7106e622e2000a", 0xc0e84f40),
}


def text_section(blob: bytes) -> tuple[int, int, int]:
    if blob[:6] != b"\x7fELF\x02\x01":
        raise ValueError("Expected ELF64 little-endian")
    header = struct.unpack_from("<16sHHIQQQIHHHHHH", blob)
    if header[2] != 243 or header[11] != 64 or not 0 < header[13] < header[12]:
        raise ValueError("Expected RISC-V ELF64 section table")
    if header[6] + header[11] * header[12] > len(blob):
        raise ValueError("Section table out of bounds")
    sections = [struct.unpack_from("<IIQQQQIIQQ", blob, header[6] + i * 64)
                for i in range(header[12])]
    names = sections[header[13]]
    if names[4] + names[5] > len(blob):
        raise ValueError("Section names out of bounds")
    strings = blob[names[4]:names[4] + names[5]]
    found = []
    for section in sections:
        if section[0] >= len(strings):
            raise ValueError("Invalid section name")
        name = strings[section[0]:].split(b"\0", 1)[0]
        if name == b".text":
            if section[1] != 1 or not section[2] & 4 or section[4] + section[5] > len(blob):
                raise ValueError("Invalid executable text bounds")
            found.append((section[3], section[4], section[5]))
    if len(found) != 1:
        raise ValueError("Expected exactly one .text section")
    return found[0]


def scan_lines(lines, targets: dict[int, str]) -> tuple[int, dict[str, list]]:
    references = {name: [] for name in targets.values()}
    previous, count = [], 0
    for line in lines:
        match = re.match(r"\s*([0-9a-f]+):\s+[0-9a-f]+\s+([^\r\n]+)", line)
        if not match:
            continue
        address, assembly = int(match[1], 16), match[2].strip()
        mnemonic = assembly.split()[0]
        count += 1
        if mnemonic in {"j", "jal", "jalr", "beq", "bne", "blt", "bge", "bltu", "bgeu", "beqz", "bnez"}:
            for value in re.findall(r"0x([0-9a-f]+)\b", assembly):
                target = int(value, 16)
                if target in targets:
                    item = {"from": hex(address), "mnemonic": mnemonic}
                    if len(previous) == 2 and previous[-1][1].startswith("auipc"):
                        if re.fullmatch(r"li\s+a0,6", previous[-2][1]):
                            item["preceding_a0_literal"] = 6
                    references[targets[target]].append(item)
                    break
        previous = (previous + [(address, assembly)])[-2:]
    return count, references


def inspect(blob: bytes, baseline: Path, objdump: Path) -> dict:
    digest = hashlib.sha256(blob).hexdigest()
    if digest != BASELINE_SHA:
        raise ValueError("Different baseline ELF; refusing board-specific signatures")
    base, offset, size = text_section(blob)
    code = blob[offset:offset + size]
    wrappers = {}
    for name, (upper, lower, prologue, ioctl) in PROFILES.items():
        pattern = struct.pack("<II", upper, lower)
        hit = code.find(pattern)
        if hit < 0 or code.find(pattern, hit + 1) >= 0:
            raise ValueError(f"{name}: expected unique ioctl signature")
        entry = code.rfind(bytes.fromhex(prologue), max(0, hit-240), hit)
        if entry < 0:
            raise ValueError(f"{name}: expected nearby SDK wrapper prologue")
        wrappers[name] = {"virtual_address": hex(base + entry),
                          "ioctl_pattern_file_offset": hex(offset + hit), "ioctl": hex(ioctl)}
    targets = {int(item["virtual_address"], 16): name for name, item in wrappers.items()}
    # Stream the full disassembly: do not retain or publish proprietary code.
    # Raw 2/4-byte walking is insufficient because this .text contains strings.
    with subprocess.Popen([str(objdump), "-d", str(baseline)], stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace") as process:
        count, references = scan_lines(process.stdout, targets)
        error = process.stderr.read()
        if process.wait() != 0:
            raise RuntimeError(error or "objdump failed")
    for name, item in wrappers.items():
        item["direct_references"] = references[name]
        needle = struct.pack("<Q", int(item["virtual_address"], 16))
        positions, cursor = [], 0
        while (hit := blob.find(needle, cursor)) >= 0:
            positions.append(hex(hit))
            cursor = hit + 1
        item["literal_pointer_file_offsets"] = positions
    return {"baseline_sha256": digest, "text_virtual_address": hex(base),
            "text_file_offset": hex(offset), "text_bytes": size,
            "objdump_instruction_lines": count, "wrappers": wrappers,
            "boundary": "Static objdump resolved jump/branch targets and literal-pointer scan; "
                        "not runtime proof; indirect/computed calls and embedded text data remain limits"}


def self_test() -> None:
    # Two actual SDK-disassembled OSD call sites and one unrelated jump.
    lines = [
        "20000150e: 4519 li a0,6",
        "200001510: 00513097 auipc ra,0x513",
        "200001514: f50080e7 jalr -176(ra) # 0x200514460",
        "200002402: 4519 li a0,6",
        "200002404: 00512097 auipc ra,0x512",
        "200002408: 05c080e7 jalr 92(ra) # 0x200514460",
        "20000240c: 000b2503 lw a0,0(s6)",
        "200002414: 2b060613 addi a2,a2,688 # 0x2005142bc",
        "200002418: a009 j 0x20000241a",
    ]
    count, refs = scan_lines(lines, {0x200514460: "osd", 0x2005142bc: "hardware"})
    expected = [{"from": "0x200001514", "mnemonic": "jalr", "preceding_a0_literal": 6},
                {"from": "0x200002408", "mnemonic": "jalr", "preceding_a0_literal": 6}]
    if count != 9 or refs != {"osd": expected, "hardware": []}:
        raise AssertionError(refs)
    print("Actual-call fixtures passed; data-address annotation rejected as a call")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path,
                        default=ROOT / "artifacts/ui_baseline_2026-09-28/ob_det_fire_mvp_spaces.elf")
    parser.add_argument("--objdump", type=Path, help="SDK RISC-V objdump executable")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if args.objdump is None:
            parser.error("--objdump is required for full binary inspection")
        print(json.dumps(inspect(args.baseline.read_bytes(), args.baseline, args.objdump), indent=2))


if __name__ == "__main__":
    main()
