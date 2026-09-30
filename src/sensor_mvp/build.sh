#!/bin/sh
set -eu

SOURCE_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUTPUT=${1:-"$SOURCE_DIR/../../artifacts/sensor_validation/sensor_mvp_oled"}
CC=${CC:-/home/k230/k230_sdk_v1.6/toolchain/Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.0/bin/riscv64-unknown-linux-gnu-gcc}

"$CC" -static -O2 -Wall -Wextra -std=c11 -D_POSIX_C_SOURCE=200809L \
  -o "$OUTPUT" \
  "$SOURCE_DIR/main.c" "$SOURCE_DIR/gpio_i2c.c" \
  "$SOURCE_DIR/sht31.c" "$SOURCE_DIR/bmp280.c" "$SOURCE_DIR/ssd1306.c"
