#!/bin/sh
# Probe the board's WiFi stack: driver binding, tools, rfkill and module list.

echo "=== 1. wlan0 详情 ==="
ip -d link show wlan0 2>/dev/null || echo "(ip -d 不支持)"

echo ""
echo "=== 2. WiFi 工具 ==="
for t in wpa_supplicant wpa_cli wpa_passphrase iwconfig iwlist iw ifconfig \
         udhcpc dhclient hostapd dnsmasq; do
    p=$(command -v $t 2>/dev/null)
    if [ -z "$p" ]; then printf '  %-16s (无)\n' "$t"; else printf '  %-16s %s\n' "$t" "$p"; fi
done

echo ""
echo "=== 3. 内核驱动日志（wifi 相关）==="
dmesg 2>/dev/null | grep -i -E 'wifi|wlan|rw007|aic|rtl8|bcmsdh|sdio|cfg80211' | tail -25

echo ""
echo "=== 4. 已加载模块 ==="
lsmod 2>/dev/null | head -20 || echo "(lsmod 无)"

echo ""
echo "=== 5. rfkill ==="
if [ -e /dev/rfkill ]; then echo "  /dev/rfkill 存在"; else echo "  (无 /dev/rfkill)"; fi

echo ""
echo "=== 6. 网络配置文件 ==="
for f in /etc/wpa_supplicant.conf /etc/wpa_supplicant/wpa_supplicant.conf \
         /etc/network/interfaces /etc/wpa_supplicant/wpa_supplicant-wlan0.conf; do
    if [ -e "$f" ]; then
        echo "  --- $f ---"
        head -25 "$f"
    fi
done

echo ""
echo "=== 7. 启动脚本里的 wifi 逻辑 ==="
for f in /etc/init.d/S*; do
    if [ -f "$f" ] && grep -qi -E 'wifi|wlan|wpa_supplicant' "$f" 2>/dev/null; then
        echo "  --- $f ---"
        grep -i -E 'wifi|wlan|wpa' "$f" | head -8
    fi
done

echo ""
echo "PROBE_DONE"
