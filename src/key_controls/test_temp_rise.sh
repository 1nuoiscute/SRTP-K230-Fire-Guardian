#!/bin/sh
# Verify the rate-of-rise maths used by fire_guard.sh's record_and_rise().
#
# It writes a synthetic temperature history with a known ramp and checks that the
# awk expression reports the expected rise over the window. This isolates the one
# piece of the guard that cannot be covered by a single status file.

TEST_HIST=/tmp/rise_test.txt
WINDOW=30

# rise_of <history file> -> prints the rise in tenths of a degree over WINDOW s
rise_of() {
    awk -v win="$WINDOW" -v tnow="$(cut -d' ' -f1 /proc/uptime)" '
        { t[NR] = $1; v[NR] = $2 }
        END {
            if (NR < 2) { print "ERR"; exit }
            cur = v[NR]
            base = ""
            for (i = NR; i >= 1; i--) {
                if (tnow - t[i] >= win) { base = v[i]; break }
            }
            if (base == "") { print "ERR"; exit }
            printf "%d", (cur - base) * 10
        }
    ' "$1" 2>/dev/null
}

now_s() { cut -d' ' -f1 /proc/uptime; }

pass=0
fail=0

check() {
    label=$1
    want=$2
    got=$3
    if [ "$got" = "$want" ]; then
        printf '  %-38s ok    rise=%s\n' "$label" "$got"
        pass=$((pass + 1))
    else
        printf '  %-38s FAIL  rise=%s (wanted %s)\n' "$label" "$got" "$want"
        fail=$((fail + 1))
    fi
}

echo "=== synthetic ramps (window = ${WINDOW}s) ==="

# 1) flat: 25.0 held for 40 s -> rise 0
T=$(now_s)
{
    awk -v t="$T" 'BEGIN { printf "%.2f 25.0\n", t - 40 }'
    awk -v t="$T" 'BEGIN { printf "%.2f 25.0\n", t - 20 }'
    echo "$T 25.0"
} >"$TEST_HIST"
check "flat 25C over 40s -> 0" "0" "$(rise_of $TEST_HIST)"

# 2) +10 C in 30 s -> 100 tenths
T=$(now_s)
{
    awk -v t="$T" 'BEGIN { printf "%.2f 25.0\n", t - 31 }'
    echo "$T 35.0"
} >"$TEST_HIST"
check "+10C in 31s -> 100" "100" "$(rise_of $TEST_HIST)"

# 3) +8 C exactly at the trip level -> 80 tenths
T=$(now_s)
{
    awk -v t="$T" 'BEGIN { printf "%.2f 25.0\n", t - 35 }'
    echo "$T 33.0"
} >"$TEST_HIST"
check "+8C in 35s -> 80 (trips at 80)" "80" "$(rise_of $TEST_HIST)"

# 4) fast rise inside the window only: baseline entry too recent -> ERR
T=$(now_s)
{
    awk -v t="$T" 'BEGIN { printf "%.2f 25.0\n", t - 5 }'
    echo "$T 35.0"
} >"$TEST_HIST"
check "baseline too recent -> ERR" "ERR" "$(rise_of $TEST_HIST)"

# 5) falling: -5 C -> negative
T=$(now_s)
{
    awk -v t="$T" 'BEGIN { printf "%.2f 30.0\n", t - 40 }'
    echo "$T 25.0"
} >"$TEST_HIST"
check "falling 30->25 -> -50" "-50" "$(rise_of $TEST_HIST)"

rm -f "$TEST_HIST"

echo ""
echo "pass=$pass fail=$fail"
if [ "$fail" -eq 0 ]; then echo "RISE_TESTS_PASSED"; else echo "RISE_TESTS_FAILED"; exit 1; fi
