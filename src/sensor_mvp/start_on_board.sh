#!/bin/sh
set -eu

DIRECTORY=/sharefs/srtp_clean/sensor_mvp
APPLICATION="$DIRECTORY/sensor_mvp_oled"
LOG=/tmp/sensor_mvp.log

if pidof sensor_mvp_oled >/dev/null 2>&1; then
    echo "sensor_mvp_oled is already running"
    exit 0
fi

cd "$DIRECTORY"
nohup setsid "$APPLICATION" >"$LOG" 2>&1 </dev/null &
echo "started sensor_mvp_oled; log: $LOG"
