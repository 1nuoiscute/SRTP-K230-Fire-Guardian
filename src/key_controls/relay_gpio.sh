#!/bin/sh
# Drive a GPIO pin as a push-pull output for relay control.
#
# Register facts (reverse-engineered and verified during the I2C work):
#   IOMUX pad N      = 0x91105000 + 4*N     low 9 bits 0x1CF = GPIO function
#                                            bit31 = input level (read)
#   GPIO1 dir reg    = 0x9140C004           bit=1 -> drive low, bit=0 -> release
#   GPIO0 dir reg    = 0x9140B000           (from the QEMU K230 model)
#
# IMPORTANT: bits 0..31 of the direction register cover GPIO0..GPIO31 in one
# register, and GPIO32..63 in the *other* bank.  For an output pin we must be
# able to drive it HIGH, which this register cannot do directly - it can only
# pull low or release.  A released pin with an external pull-up reads high,
# but that is not a reliable push-pull output for a relay module.
#
# The relay module is opto-isolated and its IN pin is pulled up on the module,
# so the workable arrangement is:
#   * drive the pin LOW  -> relay ON   (module sees a valid low)
#   * release the pin    -> relay OFF  (module's own pull-up takes it high)
#
# That is exactly the "low-level trigger" wiring the hardware plan already
# calls for, and it matches the documented jumper setting (LOW).
#
# Usage:
#   sh relay_gpio.sh probe <pad>       configure pad as output, drive low 2 s, release
#   sh relay_gpio.sh on <pad>          drive low (relay energised)
#   sh relay_gpio.sh off <pad>         release (relay de-energised)
#   sh relay_gpio.sh read <pad>        print the pad level
#   sh relay_gpio.sh scan              sweep pads 34..63, 2 s each, printing the pad
#
# The sweep is meant to be used with a multimeter on the daughterboard D2/D3/D4
# pins so the physical pin <-> pad mapping can be established empirically.

IOMUX_BASE=0x91105000
GPIO0_DIR=0x9140B000
GPIO1_DIR=0x9140C004

iomux_of() { echo $((IOMUX_BASE + 4 * $1)); }

# Which bank does this pad belong to, and which bit inside it?
bank_of() {
    if [ "$1" -lt 32 ]; then echo 0; else echo 1; fi
}
bit_of() {
    if [ "$1" -lt 32 ]; then echo "$1"; else echo $(($1 - 32)); fi
}
dir_reg() {
    if [ "$1" -lt 32 ]; then echo $GPIO0_DIR; else echo $GPIO1_DIR; fi
}

set_gpio_func() {   # pad
    devmem $(iomux_of "$1") 32 0x1CF >/dev/null
}
# Read-modify-write so the direction register keeps whatever the other users
# (notably gpio_i2c.c on the sensor program, which owns pads 44/45) have set.
# Writing the whole register here used to clear the I2C bits and break the bus.
drive_low() {       # pad
    b=$(bit_of "$1"); r=$(dir_reg "$1")
    cur=$(( $(devmem $r 32) ))
    devmem $r 32 $(( (cur | (1 << b)) & 0xFFFFFFFF )) >/dev/null
}
release() {         # pad
    b=$(bit_of "$1"); r=$(dir_reg "$1")
    cur=$(( $(devmem $r 32) ))
    devmem $r 32 $(( cur & ~(1 << b) & 0xFFFFFFFF )) >/dev/null
}
read_pad() {        # pad -> prints 0/1 from bit31 of the iomux register
    v=$(devmem $(iomux_of "$1"))
    n=$(printf '%s' "$v" | cut -c3)
    case "$n" in 8|9|A|B|C|D|E|F|a|b|c|d|e|f) echo 1 ;; *) echo 0 ;; esac
}

CMD="${1:-}"
PAD="${2:-}"

case "$CMD" in
    probe)
        [ -z "$PAD" ] && { echo "usage: probe <pad>"; exit 2; }
        set_gpio_func "$PAD"
        echo "pad $PAD: before=$(read_pad $PAD)"
        drive_low "$PAD"
        echo "pad $PAD: driven low  (relay should click ON if this is the signal pin)"
        sleep 2
        release "$PAD"
        echo "pad $PAD: released    (relay should click OFF)"
        ;;
    on)
        set_gpio_func "$PAD"; drive_low "$PAD"
        echo "pad $PAD: ON (low)"
        ;;
    off)
        set_gpio_func "$PAD"; release "$PAD"
        echo "pad $PAD: OFF (released)"
        ;;
    read)
        set_gpio_func "$PAD"
        echo "pad $PAD = $(read_pad $PAD)"
        ;;
    scan)
        echo "sweeping pads 34..63, 2 s low / 1 s high each"
        echo "watch the daughterboard D2/D3/D4 pins with a multimeter"
        for p in $(seq 34 63); do
            set_gpio_func "$p"
            printf 'pad %-3s low  -> ' "$p"
            drive_low "$p"
            sleep 2
            printf 'high\n'
            release "$p"
            sleep 1
        done
        echo "sweep done"
        ;;
    group)
        # Drive every pad in [start..end] LOW simultaneously and leave them low.
        # Used for a binary search: ask the user "is D2 low now?" then narrow.
        START="${2:-34}"
        END="${3:-63}"
        echo "driving pads $START..$END low"
        for p in $(seq "$START" "$END"); do
            set_gpio_func "$p"
            drive_low "$p"
        done
        echo "done - pads $START..$END are LOW now (relay/module pin should read low)"
        ;;
    release_all)
        echo "releasing pads 34..63"
        for p in $(seq 34 63); do
            set_gpio_func "$p"
            release "$p"
        done
        echo "all released (pins float high via pull-ups)"
        ;;
    *)
        echo "usage: sh relay_gpio.sh {probe|on|off|read} <pad> | scan | group <start> <end> | release_all"
        exit 2
        ;;
esac
