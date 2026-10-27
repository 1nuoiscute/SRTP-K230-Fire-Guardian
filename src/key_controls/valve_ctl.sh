#!/bin/sh
# Control the valve relay wired to daughterboard D2 = GPIO63.
#
# Relay wiring on this board (established by exhaustive pad scan):
#   D2 -> pad 63 -> GPIO1_DIR bit 31 (0x9140C004, 0x80000000)
#   jumper LOW -> driving the pin low energises the relay
#   released  -> module pull-up takes it high -> relay off
#
# Because the valve is normally-closed, "relay off" = "valve closed", which is
# the fail-safe state: losing power or losing the GPIO drive closes the valve.
#
# Usage: sh valve_ctl.sh {open|close|status|test}

PAD=63
RELAY=/sharefs/srtp_clean/key_controls/relay_gpio.sh

case "$1" in
    open)
        sh $RELAY on $PAD >/dev/null 2>&1
        echo "valve OPEN  (relay energised, pad $PAD driven low)"
        ;;
    close)
        sh $RELAY off $PAD >/dev/null 2>&1
        echo "valve CLOSED (relay released, pad $PAD pulled high by module)"
        ;;
    status)
        printf 'pad %s level: ' "$PAD"
        sh $RELAY read $PAD
        printf 'GPIO1_DIR   : '
        devmem 0x9140C004 32
        ;;
    test)
        # three open/close cycles so the valve mechanism can be observed
        i=1
        while [ "$i" -le 3 ]; do
            echo "cycle $i: OPEN"
            sh $RELAY on $PAD >/dev/null 2>&1
            sleep 3
            echo "cycle $i: CLOSED"
            sh $RELAY off $PAD >/dev/null 2>&1
            sleep 3
            i=$((i + 1))
        done
        echo "test done"
        ;;
    *)
        echo "usage: $0 {open|close|status|test}" >&2
        exit 2
        ;;
esac

