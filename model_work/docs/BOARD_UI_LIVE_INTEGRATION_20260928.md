# 大屏实时状态面板：2026-09-28 实机记录

## 当前交付与验收边界

- 原视觉 ELF 与 KModel 未修改；板端 SHA-256 分别为 `ec2caafd85e56851c44c838b4c9a2a15477cae8b24ac915f1c571f1a0f9a2ac1`、`ec475725794417e83b58c661769a915c7a0e7a71ed61a6e2d267ddc7e49096a7`。
- 副板 K1 现在启动原视觉程序，等待约 4 秒让 VO 就绪，再启动大屏 OSD0 状态面板；再次短按 K1 时先关闭面板，再按原路径向视觉程序发 `q` 释放摄像头。K2 仍独立控制 SHT31/BMP280/OLED，不改其行为。面板读数来自小核原有传感器日志，经 `status_bridge` 写到 1 MiB tmpfs，再由大核读取；Wi-Fi 显示依据小核 `wlan0` 是否获得 IPv4 地址。没有用 Wi-Fi 传输跨核数据。
- 2026-09-28 板端单缓冲面板 60 秒短测期间，串口打印多次变化的温湿度/气压数据，原视觉持续打印约 220 ms/轮；短测后大核 `list_process` 只有原视觉，没有面板残留。面板刚启用时仍有一次 `vb_is_blk_valid` 日志；未观察到旧双缓冲方案那种每次刷新时的 `vb_user_add`/`vb_user_sub` 告警。单缓冲面板的长期画面稳定性和现场目视动态变化仍待继续验收。
- 新 K1 控制器首个候选 `vision_key_ui_v4.elf` 自检卡在首次视觉启动后，**未刷入启动分区**。复位恢复旧 K1 后，改为由父进程等原视觉 VO 4 秒、面板子进程直接 `exec`，生成 `vision_key_ui_v4b.elf`。此版在 RT-Smart `selftest` 中连续两轮启动原视觉与面板；首轮日志有 `K1: UI stopped`、`K1: vision stopped cleanly`，第二轮启动也有传感器名与 UI 输出。自检结束后 `list_process` 无视觉和 UI 残留。此自检使用 K2 关闭状态，故面板数值显示 `--`，不代表传感器失效。
- v4b 候选 ELF SHA-256 `1d503288d14bdcdbd6e3feb3e0faabcced8cab210fbc66db7b03a023ee46168c`。用 `src/key_controls/patch_stock_romfs.py` 基于原始 p1 备份，仅定点替换 ROMFS 中原厂 `fastboot_app.elf` 的字节槽位，其他 ROMFS 字节不变。候选 20 MiB p1 镜像 SHA-256 `5fec5ccf0921e790357f523ba7b7a49dd177e101358ac86a8b55a85895264fa1`；板端写前 `/dev/mmcblk0p1` 是 v3 哈希 `fc010c3b6e9de506bff25e26ce021a24f2fa8bb7637713360660256d9a0ab1e8`，写后读回与 v4b 候选完全一致。原始、v2、v3 镜像仍在 `artifacts/board_backup_2026-09-25_autostart/`。
- 小核 `/etc/init.d/S91srtp_ui_bridge` 在首次推送后过快重启，文件变为 0 字节，桥接没有自启；重新推送并执行 `sync` 后，重启实测脚本 925 字节仍在、tmpfs 挂载成功、`status_bridge` 进程自启、状态序号递增，`/sharefs` FAT 保持 rw。桥接只写 tmpfs，不再每两秒写 SD 卡。此前 FAT 故障、完整备份与恢复见本日会话记录；备份 `artifacts/sd_sharefs_backup_2026-09-28_before_fsck/verification.json` 为 95/95 哈希一致。
- 刷入 v4b 后再次重启，板端 p1 读回哈希保持 `5fec5c...`，小核 K2 控制器和桥接自启；大核串口抓到 `RT-SMART Hello RISC-V` 与 `K1: waiting for vision files on /sharefs`。用户随后在屏幕上确认**面板参数会变化**；同时小核状态记录递增并有实际温湿度/气压，串口持续打印视觉推理耗时及 `UI: temp=...`。用户再短按副板 K1 两次，回报**画面和面板一起关闭、再次一起恢复，两次都正常**。因此新镜像的实体 K1 联动及实时参数已经现场验收；连续运行稳定性和真正分屏仍未验收。

## 屏幕下方三个图标键

工作区手册 `RT-Thread Smart AI 套件开发手册.docx` 的正面照片确有旧手机样式的菜单、主页、返回三个图标。当前小核 `/proc/bus/input/devices` 只有 `gpio_keys` 的 `event0`，大核 `list_device` 没有 `touch0` 或相应按键设备；已装固件未给面板程序提供可用事件。SDK 中有通用 FT5316 触摸驱动源文件，但不能据此认定本屏硬件已接线、驱动已启用或图标键可读。本版使用已验证的副板实体 K1 统一开关视觉和面板，没有加入无响应的屏幕按钮。

## 底板实体矩阵 KEY1：后续补做

用户指出底板另有 4×4 实体按键矩阵，并确认此前按的是其中 KEY1，不是副板 K1。底板原理图中矩阵 KEY1 接行 GPIO28、列 GPIO18。Linux `gpio_keys` 只有 GPIO33/GPIO27 两个设备树按键，用户按矩阵 KEY1 后中断计数仍为 0；不能把它误认为现成的 `/dev/input/event0` 事件。

先运行 `src/board_ui_overlay/matrix_key_probe.c`：只将 GPIO28 设为低电平输出、读 GPIO18，45 秒后把 GPIO28 恢复输入。首轮窗口记录列为高、无跳变，结束后板端 IOMUX 18/28 恢复 `0x8000018F`；当时尚不能区分用户是否在窗口内按下。其后用同一 GPIO 路径在 `overlay_live.c` 中加入 40 ms 轮询、三次采样消抖，矩阵 KEY1 只切换 OSD0 面板显示，摄像头与原识别进程持续运行。候选 ELF 先独立部署为 `ui_overlay_live_matrix_candidate.elf`，以 90 秒有界试验运行。用户在屏幕现场反馈“可以了，不错”，确认矩阵 KEY1 切换面板的效果；串口同时有持续变化的 `UI: temp=...` 和视觉推理日志。

已把候选 ELF 复制到 K1 控制器既有的 `ui_overlay_live_onebuf.elf` 路径，板端两文件 SHA-256 均为 `6dd398c1deea0d99721b44066f71df1cfe5801815d88740d9071ffaaaf9f9c46`；旧版面板 ELF 在电脑 `artifacts/board_ui_overlay/ui_overlay_live_onebuf_verified.elf`，SHA-256 `78325ec88e6f2c0d0f0a207112b4e0b15880f10efb7b753c614717d774422e12`。**K1 启动分区没有再刷写**，原视觉 ELF/KModel 仍不变。替换后重启读回 p1 哈希仍为 `5fec5ccf...`，面板 ELF 哈希保持新值，tmpfs 桥接与 K2 控制器自启。用户在此次重启后按副板 K1 打开画面，再按底板矩阵 KEY1 两次，反馈“重启后仍正常”：面板能隐藏、重新出现，摄像头保持运行。长时间连续运行仍未验证。

## 仍未解决

此版 OSD0 面板覆盖原视觉程序的全屏视频，**没有真正把摄像头画面缩进上方区域**，也没有让检测停止处理被面板遮住的下方区域。原厂视觉 ELF 只有现成二进制，没有同源可直接修改的画框/VO 分屏代码；现有 SDK 自编替代视觉程序曾出现摄像头与 MPP 兼容故障。因此此版遵守“不拉伸、不扭曲原画面”，但不能声称已解决“框可能出现在面板覆盖区域”的投诉。要实现真正分屏并裁剪框，需取得原视觉源码或先修复同版本 MPP 替代程序，再做坐标与显示区域联调。
