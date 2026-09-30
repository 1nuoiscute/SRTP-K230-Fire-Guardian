#!/bin/sh
set -eu

# Current vision_key.c includes the unverified full-screen data-page experiment.
if [ "${1:-}" != "--data-candidate" ]; then
  echo "Explicit candidate only: sh build.sh --data-candidate. No verified v4b source snapshot is available." >&2
  exit 2
fi

SDK=/home/k230/k230_sdk_v1.6
SOURCE=/mnt/c/Users/ASUS/Documents/ChatGPT/SRTP/src/key_controls
SENSOR_SOURCE=/mnt/c/Users/ASUS/Documents/ChatGPT/SRTP/src/sensor_mvp
OUTPUT=/mnt/c/Users/ASUS/Documents/ChatGPT/SRTP/artifacts/sensor_validation/key_controls
LINUX_CC="$SDK/toolchain/Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.0/bin/riscv64-unknown-linux-gnu-gcc"
SMART_CC="$SDK/toolchain/riscv64-linux-musleabi_for_x86_64-pc-linux-gnu/bin/riscv64-unknown-linux-musl-gcc"

mkdir -p "$OUTPUT"
"$LINUX_CC" -static -O2 -Wall -Wextra -std=c11 -D_POSIX_C_SOURCE=200809L \
  -o "$OUTPUT/sensor_key" "$SOURCE/sensor_key.c" "$SOURCE/key_input.c"
"$LINUX_CC" -O2 -Wall -Wextra -std=c11 -D_POSIX_C_SOURCE=200809L \
  -o "$OUTPUT/sensor_key_dynamic" "$SOURCE/sensor_key.c" "$SOURCE/key_input.c"
"$LINUX_CC" -O2 -Wall -Wextra -std=c11 -D_POSIX_C_SOURCE=200809L \
  -o "$OUTPUT/sensor_mvp_oled_dynamic" \
  "$SENSOR_SOURCE/main.c" "$SENSOR_SOURCE/gpio_i2c.c" \
  "$SENSOR_SOURCE/sht31.c" "$SENSOR_SOURCE/bmp280.c" \
  "$SENSOR_SOURCE/ssd1306.c"

"$SMART_CC" -O2 -Wall -Wextra -std=c11 -D_POSIX_C_SOURCE=200809L \
  -mcmodel=medany -march=rv64imafdcv -mabi=lp64d \
  -c -o /tmp/srtp_key_input.o "$SOURCE/key_input.c"
"$SMART_CC" -O2 -Wall -Wextra -std=c11 -D_POSIX_C_SOURCE=200809L \
  -mcmodel=medany -march=rv64imafdcv -mabi=lp64d \
  -c -o /tmp/srtp_vision_key.o "$SOURCE/vision_key.c"
"$SMART_CC" -o "$OUTPUT/vision_key_data_candidate.elf" \
  -mcmodel=medany -march=rv64imafdcv -mabi=lp64d \
  -T "$SDK/src/big/mpp/userapps/sample/linker_scripts/riscv64/link.lds" \
  -L"$SDK/src/big/rt-smart/userapps/sdk/rt-thread/lib" \
  -Wl,--whole-archive -lrtthread -Wl,--no-whole-archive \
  -n --static /tmp/srtp_vision_key.o /tmp/srtp_key_input.o \
  -L"$SDK/src/big/rt-smart/userapps/sdk/lib/risc-v/rv64" \
  -L"$SDK/src/big/rt-smart/userapps/sdk/rt-thread/lib/risc-v/rv64" \
  -Wl,--start-group -lrtthread -Wl,--end-group
