#!/bin/sh
set -eu
# New named candidate only: no push, boot patching or deployment.
SDK=${SRTP_SDK:-/home/k230/k230_sdk_v1.6}
MPP=${SRTP_MPP:-/home/k230/mpp_rtos_v06}
ROOT=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
OUT=${1:?Usage: build_key2_candidate.sh NEW_OUTPUT_DIRECTORY}
[ ! -e "$OUT/build_complete.json" ] || { echo "Existing completed build; choose a new output" >&2; exit 2; }
python3 "$ROOT/tools/check_key2_connector_profile.py"
mkdir -p "$OUT"
CC="$SDK/toolchain/riscv64-linux-musleabi_for_x86_64-pc-linux-gnu/bin/riscv64-unknown-linux-musl-gcc"
FLAGS="-O2 -Wall -Wextra -Werror -std=c11 -D_POSIX_C_SOURCE=200809L -mcmodel=medany -march=rv64imafdcv -mabi=lp64d"
RT="$SDK/src/big/rt-smart/userapps/sdk"
LINK="$SDK/src/big/mpp/userapps/sample/linker_scripts/riscv64/link.lds"
"$CC" $FLAGS -I"$ROOT/src/stock_fastboot_v06_baseline" -I"$MPP/include" -I"$MPP/include/comm" -I"$MPP/include/ioctl" -I"$MPP/userapps/api" -c "$ROOT/src/board_ui_overlay/data_page.c" -o "$OUT/data_page.o"
"$CC" $FLAGS -Wno-error=sign-compare -Wno-error=missing-field-initializers -Wno-error=return-type \
 -include "$ROOT/src/stock_fastboot_v06_baseline/k_autoconf_comm.h" \
 -I"$ROOT/src/stock_fastboot_v06_baseline" -I"$MPP/include" -I"$MPP/include/comm" -I"$MPP/include/ioctl" -I"$MPP/userapps/api" \
 -c "$MPP/userapps/src/connector/mpi_connector.c" -o "$OUT/mpi_connector.o"
"$CC" -o "$OUT/data_page_connector_candidate.elf" -mcmodel=medany -march=rv64imafdcv -mabi=lp64d \
 -T "$LINK" -L"$RT/rt-thread/lib" -Wl,--whole-archive -lrtthread -Wl,--no-whole-archive -n --static \
 "$OUT/data_page.o" "$OUT/mpi_connector.o" -L"$MPP/userapps/lib" \
 -L"$RT/lib/risc-v/rv64" -L"$RT/rt-thread/lib/risc-v/rv64" \
 -Wl,--start-group -lvo -lvb -lsys -lcommon -lrtthread -lm -Wl,--end-group
"$CC" $FLAGS -I"$ROOT/src/key_controls" -c "$ROOT/src/key_controls/key_input.c" -o "$OUT/key_input.o"
"$CC" $FLAGS '-DDATA_APP="/sharefs/srtp_clean/key2_20260930/data_page_connector_candidate.elf"' \
 -I"$ROOT/src/key_controls" -c "$ROOT/src/key_controls/vision_key.c" -o "$OUT/vision_key.o"
"$CC" -o "$OUT/vision_key_connector_candidate.elf" -mcmodel=medany -march=rv64imafdcv -mabi=lp64d \
 -T "$LINK" -L"$RT/rt-thread/lib" -Wl,--whole-archive -lrtthread -Wl,--no-whole-archive -n --static \
 "$OUT/vision_key.o" "$OUT/key_input.o" -L"$RT/lib/risc-v/rv64" -L"$RT/rt-thread/lib/risc-v/rv64" \
 -Wl,--start-group -lrtthread -Wl,--end-group
python3 "$ROOT/tools/check_key2_connector_profile.py" --candidate "$OUT/data_page_connector_candidate.elf"
sha256sum "$OUT/"*.elf
python3 - "$OUT" "$ROOT" "$MPP" "$CC" <<'PY'
import sys, json, hashlib, datetime, subprocess
from pathlib import Path
out, root, mpp = map(Path, sys.argv[1:4])
paths = [root/'src/board_ui_overlay/data_page.c', root/'src/board_ui_overlay/board_connector_profile.h', root/'src/key_controls/vision_key.c', root/'src/key_controls/key_input.c', mpp/'userapps/src/connector/mpi_connector.c', mpp/'include/comm/k_connector_comm.h', mpp/'userapps/lib/libvo.a', mpp/'userapps/lib/libvb.a', mpp/'userapps/lib/libsys.a', mpp/'userapps/lib/libcommon.a']
record = {'built_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'compiler': subprocess.check_output([sys.argv[4], '--version'], text=True).splitlines()[0], 'inputs': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths], 'outputs': [{'name': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size} for p in sorted(out.glob('*.elf'))], 'status': 'compiled only, no live board acceptance implied'}
with (out/'build_complete.json').open('x', encoding='utf-8') as f: json.dump(record, f, indent=2)
PY
