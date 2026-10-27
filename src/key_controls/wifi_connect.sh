#!/bin/sh
# Connect wlan0 to a WPA2-PSK network WITHOUT toggling the interface.
#
# Toggling `ifconfig wlan0 down/up` on this board makes the RTL8188FU driver
# re-enumerate the USB device, the interface disappears for a few seconds and
# wpa_supplicant loses its grip on it. So this version only rewrites the config,
# restarts wpa_supplicant (which reattaches to the existing netdev) and waits.
#
# Usage: sh wifi_connect2.sh <ssid> <password>

SSID="$1"
PSK="$2"

if [ -z "$SSID" ] || [ -z "$PSK" ]; then
    echo "usage: $0 <ssid> <password>" >&2
    exit 2
fi

CONF=/etc/wpa_supplicant.conf
BAK=/etc/wpa_supplicant.conf.orig

if [ -f "$CONF" ] && [ ! -f "$BAK" ]; then
    cp -f "$CONF" "$BAK" && echo "  已备份原配置到 $BAK"
fi

echo "=== 1. 写入配置 ==="
cat >"$CONF" <<CONF_EOF
ctrl_interface=/var/run/wpa_supplicant
ctrl_interface_group=0
ap_scan=1
update_config=1

network={
    ssid="$SSID"
    psk="$PSK"
    key_mgmt=WPA-PSK
    priority=1
}
CONF_EOF
echo "  OK"

echo ""
echo "=== 2. 确认接口存在 ==="
if ! ifconfig wlan0 >/dev/null 2>&1; then
    echo "  wlan0 不存在，等待驱动重新枚举..."
    i=0
    while [ "$i" -lt 20 ]; do
        sleep 1
        if ifconfig wlan0 >/dev/null 2>&1; then echo "  wlan0 已恢复（${i}s）"; break; fi
        i=$((i + 1))
    done
fi
ifconfig wlan0 2>&1 | head -2

echo ""
echo "=== 3. 重启 wpa_supplicant（不碰接口状态）==="
pidof wpa_supplicant >/dev/null 2>&1 && killall wpa_supplicant 2>/dev/null
sleep 2
rm -rf /var/run/wpa_supplicant
mkdir -p /var/run/wpa_supplicant
wpa_supplicant -B -i wlan0 -c "$CONF" 2>&1
sleep 4
if pidof wpa_supplicant >/dev/null 2>&1; then
    echo "  ✓ 运行中 (pid $(pidof wpa_supplicant))"
else
    echo "  ✗ 启动失败"
    exit 1
fi

echo ""
echo "=== 4. 等待关联（最多 30 秒）==="
i=0
while [ "$i" -lt 30 ]; do
    ST=$(wpa_cli -i wlan0 status 2>/dev/null | grep '^wpa_state=' | cut -d= -f2)
    [ $((i % 3)) -eq 0 ] && echo "  ${i}s: $ST"
    [ "$ST" = "COMPLETED" ] && { echo "  ✓ 关联成功（${i}s）"; break; }
    sleep 1
    i=$((i + 1))
done

echo ""
echo "=== 5. 关联详情 ==="
wpa_cli -i wlan0 status 2>/dev/null | head -10

echo ""
echo "=== 6. DHCP 取地址 ==="
udhcpc -i wlan0 -n -q -t 8 -T 3 2>&1 | tail -6

echo ""
echo "=== 7. 最终状态 ==="
ifconfig wlan0 2>&1 | head -5
echo "--- 路由 ---"
ip route 2>/dev/null | head -5

echo ""
echo "CONNECT_DONE"
