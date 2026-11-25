#!/bin/sh
set -eu
n=0
while ! grep -q ' /sharefs .*vfat ' /proc/mounts || ! pidof sharefs >/dev/null; do
 n=$((n+1)); test "$n" -lt 180 || { echo SHAREFS_TIMEOUT; exit 1; }; sleep 1
done
exec sh /usr/bin/srtp_recording_prepare
