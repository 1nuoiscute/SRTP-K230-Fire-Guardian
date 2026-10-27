#!/bin/sh
# Logic test for fire_guard.sh: gas + flame + temperature, with hysteresis.
#
# The valve is replaced by a stub that only prints what it was asked to do, so
# the test runs without powering the relay.
#
# Thresholds are passed explicitly so the deployed /etc config cannot skew the
# result:
#   MQ2_CLEAR_MV=490    MQ2_ALARM_MV=650
#   FLAME_CLEAR_MV=1000 FLAME_ALARM_MV=1200
#   TEMP_CLEAR_C=45     TEMP_ALARM_C=60
#   TEMP_RISE_C=8       TEMP_RISE_CLEAR_C=2      window 30 s
#
# Rules:
#   from CLOSED: open only when gas <= 490, flame <= 1000, temp < 45 and the
#                30 s rise has dropped below 2 C
#   from OPEN:   close when gas >= 650, flame >= 1200, temp >= 60 or rise >= 8 C
#   dead bands:  keep the current state

G=/sharefs/srtp_clean/key_controls/fire_guard.sh
HIST=/tmp/tg_hist

cat > /tmp/stub_valve.sh <<'STUB'
#!/bin/sh
echo "        [stub valve] $1"
STUB
chmod 755 /tmp/stub_valve.sh

# mk <name> <mq2> <flame> <temp> <humi> <pressure>
mk() {
    printf 'v2 1 1 %s %s 1 %s 0 %s %s\n' "$4" "$5" "$6" "$2" "$3" >"/tmp/t_$1.txt"
}

mk clean     100  800  25.0 50.0 950.0
mk gas       900  800  25.0 50.0 950.0
mk fire      100 1500  25.0 50.0 950.0
mk dead      550  800  25.0 50.0 950.0
mk justover  700  800  25.0 50.0 950.0
mk justunder 480  800  25.0 50.0 950.0
mk hot       100  800  65.0 30.0 950.0
mk warm      100  800  50.0 40.0 950.0
mk cool      100  800  40.0 50.0 950.0
mk nodata    -1   -1   -1   -1   -1
mk rise29    100  800  29.0 50.0 950.0

pass=0
fail=0

run() {
    STATUS=$1 VALVE=/tmp/stub_valve.sh CONF=/dev/null TEMP_HIST=$HIST \
    MQ2_CLEAR_MV=490 MQ2_ALARM_MV=650 \
    FLAME_CLEAR_MV=1000 FLAME_ALARM_MV=1200 \
    TEMP_CLEAR_C=45 TEMP_ALARM_C=60 \
    TEMP_RISE_C=8 TEMP_RISE_CLEAR_C=2 TEMP_RISE_WINDOW_S=30 \
    sh "$G" once "$2" 2>&1
}

verdict() {
    label=$1
    file=$2
    cur=$3
    expect=$4
    out=$(run "$file" "$cur")
    want=$(echo "$out" | cut -d' ' -f1)
    if [ "$want" = "$expect" ]; then
        printf '  %-34s ok    %s\n' "$label" "$out"
        pass=$((pass + 1))
    else
        printf '  %-34s FAIL  %s (wanted %s)\n' "$label" "$out" "$expect"
        fail=$((fail + 1))
    fi
}

check() {
    rm -f "$HIST"
    verdict "$@"
}

# feed_hist <temp_30s_ago>
# Only the old entry is written: the guard appends the current temperature from
# the status file itself, and if the test pre-loaded a conflicting "now" value
# the computed rise would be zero.
feed_hist() {
    t=$(cut -d' ' -f1 /proc/uptime)
    awk -v a="$t" -v v="$1" 'BEGIN { printf "%.2f %s\n", a - 35, v }' >"$HIST"
}

echo "=== from CLOSED: reopen only when everything is clearly quiet ==="
check "clean 100/800/25, closed"       /tmp/t_clean.txt     closed open
check "gas 900, closed"                /tmp/t_gas.txt       closed closed
check "fire 1500, closed"              /tmp/t_fire.txt      closed closed
check "dead 550, closed"               /tmp/t_dead.txt      closed closed
check "warm 50C, closed"               /tmp/t_warm.txt      closed closed
check "cool 40C, closed"               /tmp/t_cool.txt      closed open
check "nodata, closed"                 /tmp/t_nodata.txt    closed closed

echo ""
echo "=== from OPEN: close on any unambiguous signal ==="
check "clean, open"                    /tmp/t_clean.txt     open   open
check "gas 900, open"                  /tmp/t_gas.txt       open   closed
check "fire 1500, open"                /tmp/t_fire.txt      open   closed
check "dead 550, open"                 /tmp/t_dead.txt      open   open
check "justover 700, open"             /tmp/t_justover.txt  open   closed
check "hot 65C, open"                  /tmp/t_hot.txt       open   closed
check "warm 50C, open"                 /tmp/t_warm.txt      open   open
check "nodata, open"                   /tmp/t_nodata.txt    open   closed

echo ""
echo "=== dead band boundaries ==="
check "justunder 480, closed"          /tmp/t_justunder.txt closed open
check "dead 550, open (no chatter)"    /tmp/t_dead.txt      open   open

echo ""
echo "=== rate of rise (history-seeded) ==="
# A room that was 25 C half a minute ago but reads 29 C now is still heating up:
# 4 C/30 s is under the 8 C trip, but above the 2 C clear level, so the valve
# must stay shut rather than reopen and flap.
rm -f "$HIST"; feed_hist 25.0
verdict "rise 4C/30s, closed -> stay shut"  /tmp/t_rise29.txt closed closed

# Same history but the trip level lowered to 2 C: must close from open.
rm -f "$HIST"; feed_hist 25.0
out=$(STATUS=/tmp/t_rise29.txt VALVE=/tmp/stub_valve.sh CONF=/dev/null TEMP_HIST=$HIST \
      MQ2_CLEAR_MV=490 MQ2_ALARM_MV=650 FLAME_CLEAR_MV=1000 FLAME_ALARM_MV=1200 \
      TEMP_CLEAR_C=45 TEMP_ALARM_C=60 \
      TEMP_RISE_C=3 TEMP_RISE_CLEAR_C=2 TEMP_RISE_WINDOW_S=30 \
      sh "$G" once open 2>&1)
case "$out" in
    closed*) printf '  %-34s ok    %s\n' "rise 4C/30s, open, trip 3C -> close" "$out"; pass=$((pass + 1)) ;;
    *)       printf '  %-34s FAIL  %s\n' "rise 4C/30s, open, trip 3C -> close" "$out"; fail=$((fail + 1)) ;;
esac

rm -f "$HIST"

echo ""
echo "pass=$pass fail=$fail"
if [ "$fail" -eq 0 ]; then echo "ALL_TESTS_PASSED"; else echo "TESTS_FAILED"; exit 1; fi
