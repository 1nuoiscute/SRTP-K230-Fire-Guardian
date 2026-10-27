#!/bin/sh
# Sweep a pad range, pulsing each pad low so the relay clicks on the right one.
#
# Delegates the bank/bit arithmetic to relay_gpio.sh, which knows that
#   pads 0..31  -> GPIO0_DIR (0x9140B000), bit = pad
#   pads 32..63 -> GPIO1_DIR (0x9140C004), bit = pad - 32
#
# Usage: sh relay_scan.sh <start> <end> [on_s] [off_s]
#
# Pads belonging to other functions are skipped:
#   14 K2, 19+28 matrix keys, 43 K1, 44/45 bit-banged I2C.

RELAY=/tmp/relay.sh
SKIP="14 19 28 43 44 45"

START="${1:-0}"
END="${2:-31}"
ON="${3:-0.45}"
OFF="${4:-0.35}"

echo "=== scanning pads $START..$END (on=$ON off=$OFF) ==="
p=$START
while [ "$p" -le "$END" ]; do
    skip=0
    for s in $SKIP; do [ "$p" = "$s" ] && skip=1; done
    if [ "$skip" = "0" ]; then
        echo "PAD $p"
        sh $RELAY on "$p"  >/dev/null 2>&1
        sleep "$ON"
        sh $RELAY off "$p" >/dev/null 2>&1
        sleep "$OFF"
        sh $RELAY on "$p"  >/dev/null 2>&1
        sleep "$ON"
        sh $RELAY off "$p" >/dev/null 2>&1
        sleep "$OFF"
    fi
    p=$((p + 1))
done
echo "SCAN_DONE"
