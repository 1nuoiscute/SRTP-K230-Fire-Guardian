#!/bin/sh
set -eu
p=/usr/share/srtp_recording_release_20261124
h=/usr/share/srtp_recording_hw_20261125
r=/tmp/srtp_demo_last640_20261124
root=/tmp/srtp_recording_sharefs_20261124
v=/sharefs/srtp_clean/demo_last640_20261124
# READY is published last; the RT bootstrap must never see a partial package.
(cd "$p" && sha256sum -c manifest.sha256)
mkdir -p "$r" "$root"
test ! -e "$r/READY" || { echo ALREADY_PREPARED_REFUSED; exit 1; }
cp -R "$p/demo/." "$r/"
cp -R "$p/linux_runtime/." "$root/"
(cd "$r" && sha256sum -c "$p/demo.sha256")
(cd "$root" && sha256sum -c "$p/linux.sha256")
mode=single-owner-last640-200-20261124
vision="$r/single_owner_vision.elf"
for item in nv12 teardown original; do
    case "$item" in
        nv12)
            package=/usr/share/srtp_recording_nv12_20261125
            label=hardware-preview-last640-200-nv12-20261125 ;;
        teardown)
            package=/usr/share/srtp_recording_hw_teardown_20261125
            label=hardware-preview-last640-200-teardown-20261125 ;;
        original)
            package="$h"
            label=hardware-preview-last640-200-20261125 ;;
    esac
    test -d "$package" || continue
    if gzip -dc "$package/single_owner_vision.elf.gz" > "$vision.new" &&
       (cd "$r" && sha256sum -c "$package/vision.sha256") &&
       mv "$vision.new" "$vision"; then
        mode="$label"
        break
    fi
    rm -f "$vision.new"
    echo "HARDWARE_PREVIEW_INTEGRITY_FAILED package=$item"
done
printf '0 -\n' > "$r/control.txt"
printf '0 -\n' > "$r/parent_control.txt"
chmod 755 "$r/"*.elf "$root/srtp_clean/key_controls/"* "$root/srtp_clean/ui_overlay/status_bridge"
# rcS has now mounted /sharefs and started its server; retire disk-backed services.
sleep 2
/etc/init.d/S90srtp_sensor_key stop || true
/etc/init.d/S91srtp_ui_bridge stop || true
/etc/init.d/S96srtp_fire_guard stop || true
sleep 1
mkdir -p "$root/srtp_clean/ui_overlay/status_ram" "$root/srtp_clean/demo_last640_20261124"
mount -o bind "$root" /sharefs
mount -o bind "$r" "$v"
start-stop-daemon -S -b -m -p /tmp/srtp_recording_sensor.pid -x /sharefs/srtp_clean/key_controls/sensor_key
start-stop-daemon -S -b -m -p /tmp/srtp_recording_bridge.pid -x /sharefs/srtp_clean/ui_overlay/status_bridge -- /tmp/sensor_mvp.log /sharefs/srtp_clean/ui_overlay/status_ram/status.txt
/etc/init.d/S96srtp_fire_guard start
sleep 2
kill -0 "$(cat /tmp/srtp_recording_sensor.pid)"
kill -0 "$(cat /tmp/srtp_recording_bridge.pid)"
test -s /sharefs/srtp_clean/ui_overlay/status_ram/status.txt
printf '%s\n' "$mode" > "$r/READY.new"
mv "$r/READY.new" "$r/READY"
printf 'SINGLE_OWNER_PERSISTENT_READY\n' > /tmp/srtp_recording.status
echo SINGLE_OWNER_PERSISTENT_READY
echo "SRTP_RECORDING_RELEASE=$mode"
