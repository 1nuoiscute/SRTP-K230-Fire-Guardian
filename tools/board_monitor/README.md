# K230 调试台

Windows 桌面上位机，用于记录 K230 的传感器、阀门守护和视觉诊断。使用 Python 3.10+ 自带的 Tkinter，串口监听复用 Windows PowerShell，无需安装 Python 第三方包。USB 联机需要 Windows ADB 平台工具和相应的板卡驱动。

## 运行

从 [GitHub Releases](https://github.com/1nuoiscute/SRTP-K230-Fire-Guardian/releases) 下载 Windows x64 单文件版，双击运行。连接板子需另备 [Android Platform-Tools](https://developer.android.com/tools/releases/platform-tools) 和板卡 USB 驱动。可把 `platform-tools` 目录放在 EXE 同一位置，或通过界面「选择 ADB」指定 `adb.exe`。

从源码运行时，在仓库根目录执行：

```powershell
python -B tools/board_monitor/monitor.py
```

- 主窗口每约 2 秒从 ADB 读取一次小核快照。默认设备序号为 `RTT_K230_ADB`。ADB 先从 PATH 查找，其次尝试现有项目的 D 盘工具路径。若实际路径或设备不同，可以传入 `--adb "实际路径"` 和 `--serial 设备序号`。
- 点「连接视觉串口」接收大核日志，也可在启动时加 `--listen` 自动连接。默认 COM6，115200、8N1，可在连接前改为实际端口。若端口被其他终端占用，释放端口后再连接。
- 趋势可切换温度、湿度、气压、MQ-2 和火焰模块读数。状态序号停止变化约 6 秒，会标记采样过期。MQ-2 和火焰模块以 mV 显示。守护记录显示脚本最近写出的决策与原因，物理阀位需要独立的位置反馈。
- 运行总览的速率根据相邻两条板端进度日志计算。预览提交指程序向显示层送出帧，LCD 实际画面需现场观察。「设备调试」默认显示实时输出，包含视觉串口原文、连接变化和操作回执。可暂停跟随、选择并复制日志。
- 发现异常时点「标记故障」，或在「现场现象」输入现象后按回车。时间轴、故障标记和 RT 串口日志使用电脑接收时钟。板端日志保留其原始时间与内容。
- 点顶栏「保存会话」导出从本次启动到当前的 JSONL。点「打开会话」，拖动时间轴或双击右侧标记即可回看。现场会话保存在内存，关闭调试台前请保存需要保留的记录。

无板时可用模拟数据试用，或直接打开已保存的会话：

```powershell
python -B tools/board_monitor/monitor.py --demo
python -B tools/board_monitor/monitor.py --replay '实际会话.jsonl'
```

## 设备调试

USB 与视觉串口连接后，可以启停摄像头和识别程序、显隐摄像头页上的传感器参数面板，以及切换板载 LCD 的摄像头页与全屏传感器数据页。按钮根据串口已知状态显示下一步动作，对应副板 K1、底板 KEY1 和 KEY2。显示操作需要视觉运行。

调试台只向已验证的 NV12 中期演示控制器发送命令。点击后在板端 RAM 写入对应的带序号命令，通过串口核对接收回执和动作结果。缺少回执时界面会显示未确认，检查当前状态再决定是否重新操作。此入口也保留原板载按键。

阀门栏显示守护程序最近发出的指令。上位机的控制按钮仅作用于视觉和板载显示。演示与离线回放不连接设备控制接口。


## 硬件监测

- Linux 小核内存从 `/proc/meminfo` 读取 `MemTotal` 与 `MemAvailable`。两者的差值用来估算占用，RT-Smart 独立内存不包含在内。
- 文件系统显示 `/`、`/tmp` 和 `/sharefs` 各自的实际挂载来源、容量和可用空间。同一个 tmpfs 通过绑定挂载出现在多个路径时，表格分别报告各路径，容量不能相加。SD 与 eMMC 的设备容量来自 Linux 块设备接口。
- 「读取芯片结温」使用板端 `canaan_thermal_zone` 读取一次片上温度。该驱动返回原始码，界面使用 [嘉楠小核驱动参考](https://www.kendryte.com/k230/en/v1.7/01_software/board/osdrv/K230_Little_Core_Linux_Driver_API_Reference.html) 的换算公式得到结温，并保留采样时间与原始码。环境温度来自 SHT31。
## 构建 Windows 单文件版

在 Windows x64 上使用附带 Tkinter 的 Python 3.10+ 和 PyInstaller 6.22.3。在仓库根目录运行 `powershell -File tools/board_monitor/build-windows.ps1`，产物位于 `artifacts/board_monitor`。Android Platform-Tools 独立提供。
