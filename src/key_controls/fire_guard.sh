#!/bin/sh
# Fire-guard valve controller.
#
# Holds the gas valve open only while the air is demonstrably safe, and closes it
# the moment any monitored channel indicates trouble.
#
# Hardware facts established on the bench (2026-10-26/27):
#   relay IN  -> daughterboard D2 -> pad 63 -> GPIO1_DIR bit 31 (0x9140C004)
#   jumper    -> LOW, so driving the pin low energises the relay
#   valve     -> normally-closed, spring return, ~12 s from closed to open
#   power off -> valve closes by itself, which makes the whole chain fail-safe.
#
# ---------------------------------------------------------------------------
# Inputs and why each one is used
#
#   MQ-2 gas (mV)      primary: a leak is odourless and invisible, and it is the
#                      only channel that sees it before there is any flame.
#   flame sensor (mV)  direct fire evidence, but see the caveat below.
#   SHT31 temperature  two separate checks, because a fire shows up in both:
#                        - absolute  : ambient above TEMP_ALARM_C is a fire
#                        - rate      : a fast rise is the classic fire signature,
#                                      and catches a fire before it is hot.
#   SHT31 humidity     NOT used for the decision. Fire both dries the air and
#                      adds water vapour from combustion, so the direction is
#                      ambiguous; it stays a display value only.
#   BMP280 pressure    NOT used. Indoors it barely moves, and a fan, a door or
#                      the HVAC shifts it far more than a fire does.
#
# ---------------------------------------------------------------------------
# Hysteresis
#
#   A first live gas test showed the valve chattering (closed, reopened, closed
#   again within three seconds) because a single threshold was used on a reading
#   that lands right on it while decaying. Open and close now use different
#   thresholds with a dead band in between, exactly like a thermostat.
#
#   Gas:   open when <= MQ2_CLEAR_MV (490), close when >= MQ2_ALARM_MV (650)
#   Flame: open when <= FLAME_CLEAR_MV (1000), close when >= FLAME_ALARM_MV (1200)
#   Temp:  cleared only when below TEMP_CLEAR_C, trips at TEMP_ALARM_C
#
#   With no fresh sample the valve stays closed (POLICY_NO_DATA=close): a dead
#   sensor must never be read as "the air is clean".
#
# Usage: sh fire_guard.sh {run|once [open|closed]|status|baseline [sec]|open|close}

VALVE=${VALVE:-/sharefs/srtp_clean/key_controls/valve_ctl.sh}
STATUS=${STATUS:-/sharefs/srtp_clean/ui_overlay/status_ram/status.txt}
STATE=${STATE:-/tmp/fire_guard.state}
LOG=${LOG:-/tmp/fire_guard.log}
TEMP_HIST=${TEMP_HIST:-/tmp/fire_guard.temp}

# Threshold precedence: environment > config file > built-in defaults.
# The environment values have to be captured BEFORE sourcing the config, because
# sourcing it would overwrite them and a one-off `MQ2_ALARM_MV=... sh fire_guard`
# would silently do nothing.
CONF=${CONF:-/etc/srtp_fireguard.conf}

_env_mq2_alarm=${MQ2_ALARM_MV:-}
_env_mq2_clear=${MQ2_CLEAR_MV:-}
_env_flame_alarm=${FLAME_ALARM_MV:-}
_env_flame_clear=${FLAME_CLEAR_MV:-}
_env_policy=${POLICY_NO_DATA:-}
_env_temp_alarm=${TEMP_ALARM_C:-}
_env_temp_clear=${TEMP_CLEAR_C:-}
_env_temp_rise=${TEMP_RISE_C:-}
_env_temp_rise_clear=${TEMP_RISE_CLEAR_C:-}

MQ2_ALARM_MV=650
MQ2_CLEAR_MV=490
FLAME_ALARM_MV=1200
FLAME_CLEAR_MV=1000
TEMP_ALARM_C=60
TEMP_CLEAR_C=45
TEMP_RISE_C=8
TEMP_RISE_WINDOW_S=30
TEMP_RISE_CLEAR_C=2
POLL_SECONDS=${POLL_SECONDS:-1}
POLICY_NO_DATA=close

[ -r "$CONF" ] && . "$CONF"

if [ -n "$_env_mq2_alarm" ];   then MQ2_ALARM_MV=$_env_mq2_alarm; fi
if [ -n "$_env_mq2_clear" ];   then MQ2_CLEAR_MV=$_env_mq2_clear; fi
if [ -n "$_env_flame_alarm" ]; then FLAME_ALARM_MV=$_env_flame_alarm; fi
if [ -n "$_env_flame_clear" ]; then FLAME_CLEAR_MV=$_env_flame_clear; fi
if [ -n "$_env_policy" ];      then POLICY_NO_DATA=$_env_policy; fi
if [ -n "$_env_temp_alarm" ];  then TEMP_ALARM_C=$_env_temp_alarm; fi
if [ -n "$_env_temp_clear" ];  then TEMP_CLEAR_C=$_env_temp_clear; fi
if [ -n "$_env_temp_rise" ];   then TEMP_RISE_C=$_env_temp_rise; fi
if [ -n "$_env_temp_rise_clear" ]; then TEMP_RISE_CLEAR_C=$_env_temp_rise_clear; fi

now() { cut -d' ' -f1 /proc/uptime; }

log() { printf '%s %s\n' "$(now)" "$*" >>"$LOG"; }

valve_open() { sh "$VALVE" open >/dev/null 2>&1; }
valve_close() { sh "$VALVE" close >/dev/null 2>&1; }

# Echo "mq2 flame temp humi press" for the current status file.
# Each missing or malformed field becomes -1 so the caller can tell "no data"
# apart from a real reading.
sample() {
    ver=""; seq=""; sht=""; temp=""; humi=""; bmp=""; press=""; wifi=""; mq2=""; flame=""
    if [ -r "$STATUS" ]; then
        read -r ver seq sht temp humi bmp press wifi mq2 flame <"$STATUS" 2>/dev/null
    fi
    case "$mq2" in ''|*[!0-9-]*) mq2=-1 ;; esac
    case "$flame" in ''|*[!0-9-]*) flame=-1 ;; esac
    case "$sht" in 1) ;; *) temp=-1; humi=-1 ;; esac
    case "$temp" in ''|-*|*[!0-9.]*) temp=-1 ;; esac
    case "$humi" in ''|*[!0-9.]*) humi=-1 ;; esac
    case "$press" in ''|*[!0-9.]*) press=-1 ;; esac
    echo "$mq2 $flame $temp $humi $press"
}

# Append the current temperature to the rolling window, then report the rise in
# degrees over the window as an integer count of tenths (so shell can compare).
#
# The guard polls twice a second but the bridge only refreshes the status file so
# often, so duplicate timestamps are dropped to keep the window honest.
record_and_rise() {
    t=$1
    [ "$t" = "-1" ] && { echo ""; return; }

    ts=$(now)
    echo "$ts $t" >>"$TEMP_HIST" 2>/dev/null
    # keep ~2 minutes of history at 1 Hz
    tail -n 150 "$TEMP_HIST" >"$TEMP_HIST.new" 2>/dev/null && mv "$TEMP_HIST.new" "$TEMP_HIST"

    awk -v win="$TEMP_RISE_WINDOW_S" -v tnow="$ts" '
        { t[NR] = $1; v[NR] = $2 }
        END {
            if (NR < 2) { print ""; exit }
            cur = v[NR]
            base = ""
            for (i = NR; i >= 1; i--) {
                if (tnow - t[i] >= win) { base = v[i]; break }
            }
            if (base == "") { print ""; exit }
            printf "%d", (cur - base) * 10
        }
    ' "$TEMP_HIST" 2>/dev/null
}

# Compare "25.6" against an integer Celsius threshold. Echo 1 when >=.
temp_at_least() {
    awk -v v="$1" -v lim="$2" 'BEGIN { exit !(v >= lim) }' 2>/dev/null && echo 1 || echo 0
}

# decide <current-state: open|closed>
# Echo "desired-state mq2 flame temp reason".
decide() {
    cur=$1
    set -- $(sample)
    mq2=$1
    flame=$2
    temp=$3
    humi=$4
    press=$5

    no_gas=$([ "$mq2" -lt 0 ] && echo 1 || echo 0)
    no_flame=$([ "$flame" -lt 0 ] && echo 1 || echo 0)
    no_temp=$([ "$temp" = "-1" ] && echo 1 || echo 0)

    rise=$(record_and_rise "$temp")

    if [ "$no_gas" = "1" ] && [ "$no_flame" = "1" ] && [ "$no_temp" = "1" ]; then
        if [ "$POLICY_NO_DATA" = "close" ]; then
            echo "closed -1 -1 -1 no-data"
        else
            echo "open -1 -1 -1 no-data(policy=open)"
        fi
        return
    fi

    hot=0
    if [ "$no_temp" = "0" ]; then
        if [ "$(temp_at_least "$temp" "$TEMP_ALARM_C")" = "1" ]; then hot=1; fi
        if [ -n "$rise" ] && [ "$rise" -ge $((TEMP_RISE_C * 10)) ]; then hot=1; fi
    fi

    if [ "$cur" = "open" ]; then
        if [ "$mq2" -ge "$MQ2_ALARM_MV" ]; then
            echo "closed $mq2 $flame $temp gas-${mq2}"
        elif [ "$flame" -ge "$FLAME_ALARM_MV" ]; then
            echo "closed $mq2 $flame $temp flame-${flame}"
        elif [ "$hot" = "1" ]; then
            echo "closed $mq2 $flame $temp heat-${temp}C/rise${rise}"
        else
            echo "open $mq2 $flame $temp clear-hold"
        fi
    else
        cool=1
        if [ "$mq2" -ge 0 ] && [ "$mq2" -gt "$MQ2_CLEAR_MV" ]; then cool=0; fi
        if [ "$flame" -ge 0 ] && [ "$flame" -gt "$FLAME_CLEAR_MV" ]; then cool=0; fi
        if [ "$no_temp" = "0" ] && [ "$(temp_at_least "$temp" "$TEMP_CLEAR_C")" = "1" ]; then cool=0; fi
        # A rise still above the clear level means the heat source is still
        # present, so the temperature has not really returned to normal. Without
        # this the valve flapped open/closed while the room was cooling: the
        # absolute temperature had dropped below the clear limit but the air was
        # still 3 C hotter than 30 s earlier.
        if [ -n "$rise" ] && [ "$rise" -gt $((TEMP_RISE_CLEAR_C * 10)) ]; then cool=0; fi
        if [ "$cool" = "1" ]; then
            echo "open $mq2 $flame $temp clear"
        else
            echo "closed $mq2 $flame $temp still-elevated"
        fi
    fi
}

apply_state() {
    want=$1
    mq2=$2
    flame=$3
    temp=$4
    reason=$5
    prev=$6
    if [ "$want" != "$prev" ]; then
        if [ "$want" = "open" ]; then valve_open; else valve_close; fi
        log "$want ($reason) mq2=$mq2 flame=$flame temp=$temp"
    fi
    echo "$want $mq2 $flame $temp $reason" >"$STATE"
}

case "${1:-run}" in
    run)
        log "guard started: gas open<=${MQ2_CLEAR_MV} close>=${MQ2_ALARM_MV}; flame open<=${FLAME_CLEAR_MV} close>=${FLAME_ALARM_MV}; temp close>=${TEMP_ALARM_C}C or rise>=${TEMP_RISE_C}C/${TEMP_RISE_WINDOW_S}s; no-data=$POLICY_NO_DATA"
        last="closed"
        set -- $(decide "$last")
        apply_state "$1" "$2" "$3" "$4" "$5" ""
        last=$1
        while :; do
            set -- $(decide "$last")
            apply_state "$1" "$2" "$3" "$4" "$5" "$last"
            last=$1
            sleep "$POLL_SECONDS"
        done
        ;;
    once)
        cur="${2:-closed}"
        set -- $(decide "$cur")
        if [ "$1" = "open" ]; then valve_open; else valve_close; fi
        echo "$1 ($5) mq2=$2 flame=$3 temp=$4"
        ;;
    status)
        printf 'gas      : open <= %s, close >= %s mV (hysteresis)\n' "$MQ2_CLEAR_MV" "$MQ2_ALARM_MV"
        printf 'flame    : open <= %s, close >= %s mV\n' "$FLAME_CLEAR_MV" "$FLAME_ALARM_MV"
        printf 'temp     : close >= %s C or rise >= %s C / %s s; clear < %s C\n' \
               "$TEMP_ALARM_C" "$TEMP_RISE_C" "$TEMP_RISE_WINDOW_S" "$TEMP_CLEAR_C"
        printf 'no-data  : %s\n' "$POLICY_NO_DATA"
        printf 'status   : '; cat "$STATUS" 2>/dev/null || echo "(missing)"
        printf 'guard    : '; cat "$STATE" 2>/dev/null || echo "(not running)"
        printf 'log tail :\n'; tail -6 "$LOG" 2>/dev/null || true
        ;;
    baseline)
        n=${2:-20}
        printf 'sampling %ss of clean air (gas channel)...\n' "$n"
        i=0; min=""; max=""; sum=0; cnt=0
        while [ "$i" -lt "$n" ]; do
            set -- $(sample)
            v=$1
            if [ "$v" -ge 0 ]; then
                if [ -z "$min" ] || [ "$v" -lt "$min" ]; then min=$v; fi
                if [ -z "$max" ] || [ "$v" -gt "$max" ]; then max=$v; fi
                sum=$((sum + v)); cnt=$((cnt + 1))
            fi
            sleep 1
            i=$((i + 1))
        done
        if [ "$cnt" -eq 0 ]; then echo "no samples - is the sensor program running?"; exit 1; fi
        mean=$((sum / cnt))
        printf 'samples     : %s\n' "$cnt"
        printf 'min/mean/max: %s / %s / %s mV\n' "$min" "$mean" "$max"
        printf 'suggested   : MQ2_CLEAR_MV=%s  MQ2_ALARM_MV=%s\n' "$((mean + 60))" "$((mean + 180))"
        printf 'current     : MQ2_CLEAR_MV=%s  MQ2_ALARM_MV=%s  (from %s)\n' \
               "$MQ2_CLEAR_MV" "$MQ2_ALARM_MV" "$CONF"
        ;;
    open)
        valve_open; echo "valve OPEN (guard will override on the next poll)"
        ;;
    close)
        valve_close; echo "valve CLOSED (guard will reopen once readings are clearly clear)"
        ;;
    *)
        echo "usage: $0 {run|once [open|closed]|status|baseline [sec]|open|close}" >&2
        exit 2
        ;;
esac
