# SensorFusion 副板传感器与 OLED MVP

目标：K230 上以普通字号循环显示 SHT31 温度、SHT31 湿度、BMP280 温度、BMP280 气压，并预留一行 `WIFI --`。AHT20 芯片已掉落，不参与读取。火焰视觉程序与本程序相互独立。

## 硬件接法

- 底板 Arduino `SCL/SDA` 对应 K230 `GPIO44/GPIO45`。
- 副板最右下角两只黄色跳线帽：**上排 `2.2–SDA`，下排 `SCL–4.7`**。两排同时跳到 `4.7` 或同时跳到 `2.2` 会把 SCL、SDA 接到同一节点。
- SHT31 七位地址 `0x44`；BMP280 为 `0x76`，芯片 ID `0x58`；SSD1306 OLED 检测 `0x3C/0x3D`，实机为 `0x3C`。
- 所有移动副板、跳线帽的操作均在两根 USB 拔掉后进行。

## 模块

| 文件 | 职责 |
| --- | --- |
| `gpio_i2c.c/.h` | Linux 小核经 `/dev/mem` 用 GPIO44/45 模拟开漏 I²C，退出时恢复寄存器 |
| `sht31.c/.h` | 单次测量，温湿度转换与双 CRC 校验 |
| `bmp280.c/.h` | ID、校准参数、连续采样与温度/气压补偿 |
| `ssd1306.c/.h` | 128×64 OLED 初始化、普通字号绘制、仅发送变化页 |
| `main.c` | 采样、错误状态与五行显示；Wi-Fi 状态入口在此处 |

当前程序跑在 Linux 小核，使用 GPIO 模拟 I²C；RT-Smart 的硬件 I²C3 曾独立读到 BMP280 ID `0x58` 和约 27.5°C。该程序不修改固件或启动项，断电后需重新启动。

## 构建和运行

在 WSL `K230-Ubuntu` 中执行：

```sh
sh /mnt/c/Users/ASUS/Documents/ChatGPT/SRTP/src/sensor_mvp/build.sh \
  /mnt/c/Users/ASUS/Documents/ChatGPT/SRTP/artifacts/sensor_validation/sensor_mvp_oled
```

Windows 端使用 `D:\platform-tools-latest-windows\platform-tools\adb.exe` 推送编译产物到板端 Linux，例如：

```text
adb push C:\Users\ASUS\Documents\ChatGPT\SRTP\artifacts\sensor_validation\sensor_mvp_oled /sharefs/srtp_clean/sensor_mvp/sensor_mvp_oled
adb shell chmod 755 /sharefs/srtp_clean/sensor_mvp/sensor_mvp_oled
adb shell /sharefs/srtp_clean/sensor_mvp/sensor_mvp_oled
```

若要在退出 ADB 后继续运行，进入**交互式** `adb shell`，再执行：

```sh
sh /sharefs/srtp_clean/sensor_mvp/start_on_board.sh
pidof sensor_mvp_oled
tail -n 4 /tmp/sensor_mvp.log
```

2026-09-25 实机验证：退出交互 shell 后进程仍持续采样；一次性 `adb shell "sh ..."` 启动后台程序未能留下进程，不作为可靠启动方式。启动脚本将逐次读数日志写到 `/tmp/sensor_mvp.log`，避免持续写入 `/sharefs`。程序每轮打印 SHT31（含 CRC 状态）和 BMP280 数值。若读数失败，OLED 对应行显示 `ERR`。前台运行可用 `Ctrl+C` 退出，后台运行可用 `kill -TERM <pid>` 退出并释放 GPIO。启动前确保 `GPIO44/45` 为默认 GPIO 输入复用（IOMUX 低位 `0x1CF`），且同一总线上没有第二个控制进程。Wi-Fi 目前仅预留显示位置，尚未测试 RW007 连通与状态读取。

## 手册依据

- 本机底板原理图：`D:\111\01_学习手册\K230-底板.pdf`，Arduino SCL/SDA 对应 GPIO44/45。
- 本机副板原理图：`D:\111\01_学习手册\k230-副板.pdf`，P2 跳线网络、SSD1306 128×64。
- 用户提供 BMP280 手册：`D:\C83291_F39E84AB7DFC569A4C7554F8659640A1.pdf`；仅查通信和数据处理有关页面。
- [Bosch BMP280 数据手册](https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bmp280-ds001.pdf)：校准参数表、`0xF7–0xFC` 连续读取、附录 8.1 的温度和气压补偿公式。
