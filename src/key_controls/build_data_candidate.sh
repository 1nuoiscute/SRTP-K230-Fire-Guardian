#!/bin/sh
set -eu
SDK=/home/k230/k230_sdk_v1.6
SOURCE=/mnt/c/Users/ASUS/Documents/ChatGPT/SRTP/src/key_controls
OUTPUT=/mnt/c/Users/ASUS/Documents/ChatGPT/SRTP/artifacts/sensor_validation/key_controls/vision_key_data_candidate.elf
CC="$SDK/toolchain/riscv64-linux-musleabi_for_x86_64-pc-linux-gnu/bin/riscv64-unknown-linux-musl-gcc"
"$CC" -O2 -Wall -Wextra -Werror -std=c11 -D_POSIX_C_SOURCE=200809L \
  -mcmodel=medany -march=rv64imafdcv -mabi=lp64d \
  -c -o /tmp/srtp_data_key_input.o "$SOURCE/key_input.c"
"$CC" -O2 -Wall -Wextra -Werror -std=c11 -D_POSIX_C_SOURCE=200809L \
  -mcmodel=medany -march=rv64imafdcv -mabi=lp64d \
  -c -o /tmp/srtp_data_vision_key.o "$SOURCE/vision_key.c"
"$CC" -o "$OUTPUT" \
  -mcmodel=medany -march=rv64imafdcv -mabi=lp64d \
  -T "$SDK/src/big/mpp/userapps/sample/linker_scripts/riscv64/link.lds" \
  -L"$SDK/src/big/rt-smart/userapps/sdk/rt-thread/lib" \
  -Wl,--whole-archive -lrtthread -Wl,--no-whole-archive \
  -n --static /tmp/srtp_data_vision_key.o /tmp/srtp_data_key_input.o \
  -L"$SDK/src/big/rt-smart/userapps/sdk/lib/risc-v/rv64" \
  -L"$SDK/src/big/rt-smart/userapps/sdk/rt-thread/lib/risc-v/rv64" \
  -Wl,--start-group -lrtthread -Wl,--end-group
