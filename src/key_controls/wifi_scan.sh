#!/bin/sh
# Bring wlan0 up and list the visible access points.

echo "=== 启用 wlan0 ==="
ifconfig wlan0 up 2>&1
sleep 2
ifconfig wlan0 2>&1 | head -5

echo ""
echo "=== 扫描（约 5 秒）==="
iwlist wlan0 scan 2>/dev/null | grep -E 'ESSID|Encryption|Quality|Signal' | sed 's/^ *//'

echo ""
echo "=== 扫描结果摘要 ==="
iwlist wlan0 scan 2>/dev/null | grep 'ESSID:' | sed 's/.*ESSID:"\(.*\)"/\1/' | sort -u

echo ""
echo "SCAN_DONE"
