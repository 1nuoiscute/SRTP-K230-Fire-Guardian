#!/bin/bash
set -euo pipefail
ROOT=/mnt/c/Users/ASUS/Documents/ChatGPT/SRTP
P=$ROOT/artifacts/board_diagnosis_20261119/capture_probe
O=${1:-$ROOT/artifacts/recording_demo_20261124}
S=$ROOT/src/recording_demo
CACHE=$ROOT/artifacts/board_deploy_yolov8n_last640_20261124/keys_restore
mkdir -p "$O"
T=/root/srtp-capture-probe-20261119/toolchain/riscv64-linux-musleabi_for_x86_64-pc-linux-gnu/bin/riscv64-unknown-linux-musl
cd "$P"
flags=(-O2 -Wall -Wextra -Werror -Wno-misleading-indentation -std=c11 -D_POSIX_C_SOURCE=200809L -mcmodel=medany -march=rv64imafdcv -mabi=lp64d -Igenerated -Ideps/mpp/include -Ideps/mpp/include/comm -Ideps/mpp/include/ioctl -include generated/k_sensor_comm.h -Ideps/mpp/userapps/api -I"$ROOT/src/board_ui_overlay" -I"$ROOT/src/key_controls")
link=(-static -n -T deps/mpp/userapps/sample/linker_scripts/riscv64/link.lds)


N=$CACHE/nncase_rtos_2_9_0/nncase_k230_v2.9.0_runtime_rtos
for n in panel_renderer data_renderer single_owner_connector status_cache; do "$T-gcc" "${flags[@]}" -c "$S/$n.c" -o "$O/$n.o"; done
"$T-g++" -O2 -std=c++17 -Wall -Wextra -Werror -Wno-unused-parameter -Wno-multichar -Wno-misleading-indentation -mcmodel=medany -march=rv64imafdcv -mabi=lp64d -Igenerated -Ideps/mpp/include -Ideps/mpp/include/comm -Ideps/mpp/include/ioctl -include generated/k_sensor_comm.h -Ideps/mpp/userapps/api -I"$N/include" -I"$CACHE/nncase_rtos_2_9_0/deps" -static -n -T deps/mpp/userapps/sample/linker_scripts/riscv64/link.lds -Wl,--wrap=kd_mpi_sensor_set_focus_power -o "$O/single_owner_vision.elf" "$S/single_owner_vision.cc" "$O/panel_renderer.o" "$O/data_renderer.o" "$O/single_owner_connector.o" "$O/status_cache.o" -Wl,--start-group "$N/lib/"*.a deps/mpp/userapps/lib/lib*.a obj/mpi_sensor.o obj/mpi_sensor_type_to_mirror.o "$ROOT/artifacts/key2_20260930_connector_candidate/mpi_connector.o" deps/runtime/librtthread.a -Wl,--end-group -lpthread -latomic -lm
"$T-strip" "$O/single_owner_vision.elf"
sha256sum "$O/single_owner_vision.elf"

"$T-gcc" "${flags[@]}" "${link[@]}" -o "$O/single_owner_controller.elf" "$S/single_owner_controller.c" "$ROOT/src/key_controls/key_input.c" deps/runtime/librtthread.a -lpthread -lm
"$T-gcc" "${flags[@]}" "${link[@]}" -o "$O/low200.elf" "$S/kpu_session.c" deps/runtime/librtthread.a -lpthread -lm
sha256sum "$O/"*.elf
