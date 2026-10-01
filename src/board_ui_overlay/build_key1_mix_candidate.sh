#!/bin/sh
set -eu
# Builds only. Existing output directories, boot patches and deployments excluded.
SDK=${SRTP_SDK:-/home/k230/k230_sdk_v1.6}
MPP=${SRTP_MPP:-/home/k230/mpp_rtos_v06}
ROOT=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
OUT=${1:?Usage: build_key1_mix_candidate.sh NEW_OUTPUT_DIRECTORY}
[ ! -e "$OUT" ] || { echo "Choose a new output directory" >&2; exit 2; }
python3 "$ROOT/tools/check_key2_connector_profile.py"
mkdir -p "$OUT"
CC="$SDK/toolchain/riscv64-linux-musleabi_for_x86_64-pc-linux-gnu/bin/riscv64-unknown-linux-musl-gcc"
LINUX_CC="$SDK/toolchain/Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.0/bin/riscv64-unknown-linux-gnu-gcc"
RT="$SDK/src/big/rt-smart/userapps/sdk"
LINK="$SDK/src/big/mpp/userapps/sample/linker_scripts/riscv64/link.lds"
FLAGS="-O2 -Wall -Wextra -Werror -std=c11 -D_POSIX_C_SOURCE=200809L -mcmodel=medany -march=rv64imafdcv -mabi=lp64d"
INCLUDES="-I$ROOT/src/stock_fastboot_v06_baseline -I$MPP/include -I$MPP/include/comm -I$MPP/include/ioctl -I$MPP/userapps/api"
"$CC" $FLAGS $INCLUDES -Wno-missing-field-initializers -Dmain=key1_panel_main -c "$ROOT/src/board_ui_overlay/overlay_live.c" -o "$OUT/overlay_live.o"
"$CC" $FLAGS $INCLUDES -c "$ROOT/src/board_ui_overlay/overlay_mix_candidate.c" -o "$OUT/overlay_mix_candidate.o"
"$CC" -o "$OUT/key1_mix_bounded_candidate.elf" -mcmodel=medany -march=rv64imafdcv -mabi=lp64d \
 -T "$LINK" -L"$RT/rt-thread/lib" -Wl,--whole-archive -lrtthread -Wl,--no-whole-archive -n --static \
 -Wl,--wrap=kd_mpi_vo_osd_enable -Wl,--wrap=kd_mpi_vo_osd_disable \
 "$OUT/overlay_live.o" "$OUT/overlay_mix_candidate.o" -L"$MPP/userapps/lib" \
 -L"$RT/lib/risc-v/rv64" -L"$RT/rt-thread/lib/risc-v/rv64" \
 -Wl,--start-group -lvo -lvb -lsys -lcommon -lrtthread -lm -Wl,--end-group
"$LINUX_CC" -static -O2 -Wall -Wextra -Werror -std=c11 -march=rv64gc -mabi=lp64d \
 -o "$OUT/key1_mix_guard_fake_test" "$ROOT/src/board_ui_overlay/test_key1_mix_guard.c"
python3 - "$ROOT" "$OUT" "$SDK" "$MPP" "$CC" <<'PY'
import datetime,hashlib,json,subprocess,sys
from pathlib import Path
root,out,sdk,mpp=map(Path,sys.argv[1:5]); cc=Path(sys.argv[5])
nm=cc.with_name(cc.name.replace('gcc','nm')); objdump=cc.with_name(cc.name.replace('gcc','objdump'))
symbols=subprocess.check_output([str(nm),str(out/'key1_mix_bounded_candidate.elf')],text=True)
for name in ('key1_panel_main','__wrap_kd_mpi_vo_osd_enable','__wrap_kd_mpi_vo_osd_disable'):
 if not any(line.split()[-1:]==[name] for line in symbols.splitlines()): raise SystemExit('Missing candidate symbol '+name)
assembly=subprocess.check_output([str(objdump),'--disassemble=key1_panel_main','-d',str(out/'key1_mix_bounded_candidate.elf')],text=True)
for name in ('__wrap_kd_mpi_vo_osd_enable','__wrap_kd_mpi_vo_osd_disable'):
 if '<'+name+'>' not in assembly: raise SystemExit('Panel calls not wrapped: '+name)
inputs=[root/'src/board_ui_overlay'/name for name in ('overlay_live.c','overlay_mix_candidate.c','key1_mix_guard.h','test_key1_mix_guard.c','build_key1_mix_candidate.sh')]
inputs += [mpp/'userapps/lib'/name for name in ('libvo.a','libvb.a','libsys.a','libcommon.a')]
inputs += [sdk/'src/big/rt-smart/userapps/sdk/rt-thread/lib/risc-v/rv64/librtthread.a',sdk/'src/big/mpp/userapps/sample/linker_scripts/riscv64/link.lds']
record={'built_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'compiler':subprocess.check_output([str(cc),'--version'],text=True).splitlines()[0],
 'inputs':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs],
 'outputs':[{'name':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in (out/'key1_mix_bounded_candidate.elf',out/'key1_mix_guard_fake_test')],
 'link_audit':'Original panel function references both OSD wrappers',
 'status':'compiled only; fake test and physical screen acceptance are separate'}
with (out/'build_complete.json').open('x',encoding='utf-8') as f: json.dump(record,f,indent=2)
print(json.dumps(record['outputs'],indent=2))
PY
