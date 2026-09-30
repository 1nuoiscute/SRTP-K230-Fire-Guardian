"""Read-only verification of this board's connector profile and candidate payload."""
import argparse
import hashlib
import json
import re
import struct
from pathlib import Path

BASELINE_SHA = 'ec2caafd85e56851c44c838b4c9a2a15477cae8b24ac915f1c571f1a0f9a2ac1'
ROOT = Path(__file__).resolve().parents[1]


def records(blob):
    if blob[:6] != b'\x7fELF\x02\x01':
        raise ValueError('Expected ELF64 little-endian')
    e = struct.unpack_from('<16sHHIQQQIHHHHHH', blob)
    if e[2] != 243 or e[11] != 64:
        raise ValueError('Expected RISC-V ELF64 section layout')
    sections = [struct.unpack_from('<IIQQQQIIQQ', blob, e[6] + i * e[11]) for i in range(e[12])]
    pointers = []
    for s in sections:
        if s[1] != 1:
            continue
        at, end = s[4], s[4] + s[5]
        if end > len(blob):
            raise ValueError('Invalid section bounds')
        while True:
            hit = blob.find(b'nt35516\0', at, end)
            if hit < 0:
                break
            pointers.append(s[3] + hit - s[4])
            at = hit + 1
    found = []
    for pointer in pointers:
        needle = struct.pack('<Q', pointer)
        for s in sections:
            if s[1] != 1:
                continue
            at, end = s[4], s[4] + s[5]
            while True:
                hit = blob.find(needle, at, end)
                if hit < 0:
                    break
                at = hit + 1
                if hit + 112 > end:
                    continue
                words = list(struct.unpack_from('<26I', blob, hit + 8))
                if words[15] == 540 and words[20] == 960 and words[24] == 5 and words[25] == 0:
                    found.append((hit, words))
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=ROOT/'artifacts/ui_baseline_2026-09-28/ob_det_fire_mvp_spaces.elf')
    parser.add_argument('--candidate', type=Path)
    args = parser.parse_args()
    blob = args.baseline.read_bytes()
    if hashlib.sha256(blob).hexdigest() != BASELINE_SHA:
        raise SystemExit('Different baseline ELF: profile is board-specific, refuse reuse')
    profile = json.loads((ROOT/'docs/key2_connector_profile_20260930.json').read_text(encoding='utf-8'))
    found = records(blob)
    if len(found) != 1 or found[0] != (profile['source_file_offset'], profile['u32_words']) or profile['source_sha256'] != BASELINE_SHA:
        raise SystemExit('Profile identity, offset or payload mismatch')
    header = (ROOT/'src/board_ui_overlay/board_connector_profile.h').read_text(encoding='utf-8')
    match = re.search(r'board_connector_words\[26\]\s*=\s*\{([^}]+)\}', header)
    if not match or [int(n) for n in re.findall(r'(\d+)U', match[1])] != found[0][1]:
        raise SystemExit('C header differs from verified original profile')
    if args.candidate:
        data = args.candidate.read_bytes()
        if data[:6] != b'\x7fELF\x02\x01' or struct.unpack_from('<H', data, 18)[0] != 243:
            raise SystemExit('Candidate is not RISC-V ELF64 LE')
        if data.count(struct.pack('<26I', *found[0][1])) != 1:
            raise SystemExit('Candidate lacks exactly one complete profile payload')
    print('Original ELF, profile JSON and C payload agree; candidate constants checked when supplied. Live display acceptance remains separate.')


if __name__ == '__main__':
    main()
