> **2026-09-30 最新：** KEY2 新 connector 页和矩阵切换已获用户实屏确认；新启动镜像已写入并实际重启，启动与读回一致；用户确认重启后两轮实体切换正常。历史 v4b 回退与旧失败候选继续保留。见 [修复与部署记录](../../model_work/docs/KEY2_CONNECTOR_CANDIDATE_20260930.md) 和 [版本身份](../../docs/key2_release_identity_20260930.json)。以下旧“未验收”段落保留为历史边界。

> 2026-09-30 版本保护：当前 vision_key.c 已含未验收数据页候选，不能重建为“已验收 v4b”。通用 build.sh 要求显式 --data-candidate，并只输出候选文件名；已验收二进制身份见 ../../docs/BOARD_BASELINE.md。

# 副板 K1/K2 程序开关

截至 2026-09-30：正式启动组合仍是历史验收 v4b，其 ELF/p1 已在本机重新校验并冻结。在阶段提交 9fa6172 时，vision_key.c 和 vision_key_data_candidate.c 内容相同；现在当前源码继续开发 connector 修复候选，旧命名候选源码保留；旧数据页实机曾乱码，当前候选亦不能当成 v4b 源码重新部署。副板 K1、K2 与底板矩阵 KEY1、KEY2 不同。见 [板端交接](../../docs/BOARD_HANDOFF_20260930.md)。

## 2026-09-25 至 28 历史过程

2026-09-28 已验收的 v4b K1 控制器会在原视觉进程启动约 4 秒后启动实时 OSD0 面板，关闭时先退面板，再走视觉 q 退出路径。两轮 selftest、p1 写入/读回和现场 K1 关闭/恢复已完成。此处描述历史二进制行为，不表示当前候选源码已通过相同验收。

本目录是 2026-09-25 的按键控制源码。完整开发过程见项目根目录 `2026-09-23_RT-Thread_Smart_AI套件实机验证记录.md` 第 20、21、23、24 节。

硬件：断电时在副板 J1 装 `K1–D9`、`K2–D10` 两只跳帽；分别对应 GPIO43、GPIO14，按下低电平。SDA/SCL 跳帽维持现有接法。两键按压引起的高低电平变化已实测。

`key_input.c/.h` 只读 IOMUX 电平，不改引脚配置；每 20 ms 采样，连续稳定约 60 ms 才触发。`vision_key.c` 在 RT-Smart 大核运行，K1 切换视觉进程：启动时 fork/exec 已有 ELF，关闭时向子进程 stdin 写 `q`，走原程序摄像头释放流程。RT-Smart 的 `waitpid` 忽略 `WNOHANG`，因此主轮询不能对运行中的视觉子进程调用它；只在发送 `q` 后等待回收。`sensor_key.c` 在 Linux 小核运行，K2 切换新版 SHT31/BMP280/OLED 程序，关闭时发 SIGTERM 让 OLED 清屏并释放 GPIO。控制器启动时两个功能默认关闭。

在 WSL `K230-Ubuntu` 构建当前**未验收数据页候选**（不部署、不覆盖 v4b；通用脚本也会重建传感器本地产物，回退快照先保留）：

```sh
sh /mnt/c/Users/ASUS/Documents/ChatGPT/SRTP/src/key_controls/build.sh --data-candidate
sh /mnt/c/Users/ASUS/Documents/ChatGPT/SRTP/src/sensor_mvp/build.sh \
  /mnt/c/Users/ASUS/Documents/ChatGPT/SRTP/artifacts/sensor_validation/sensor_mvp_oled
```

候选视觉产物名仅为 `vision_key_data_candidate.elf`；不能用它覆盖已验收 `vision_key.elf` 或启动分区。以下为历史部署路径。

产物在 `C:\Users\ASUS\Documents\ChatGPT\SRTP\artifacts\sensor_validation\key_controls\`。板端实际使用动态小核产物 `sensor_key_dynamic`、`sensor_mvp_oled_dynamic`（推送后分别命名为 `/sharefs/srtp_clean/key_controls/sensor_key`、`sensor_mvp_oled_keys`），以及大核 `/sharefs/srtp_clean/key_controls/vision_key_v3.elf`。K2 的小核启动脚本为 `/etc/init.d/S90srtp_sensor_key`。大核 p1 采用 `patch_stock_romfs.py` 仅在原厂 ROMFS 的 `fastboot_app.elf` 槽位放入 v3 控制器，原厂备份、v2 和 v3 镜像均在 `artifacts/board_backup_2026-09-25_autostart/`。

`probe` 和两轮 `selftest` 已通过，用户已验证实体 K2 开/关。v2 曾在**拔掉两根 USB 再上电**的方式下通过 K1 开/关/再开，但在 USB 始终插着、只拨右上角 ON/OFF 开关时出现共享区就绪前的过早启动，旧 PID 使 K1 失效。v3 在普通模式等到视觉 ELF 与 KModel 可访问后才初始化 K1，并在子进程已退出时清理旧状态。v3 已刷入 p1，SHA-256 `fc010c3b6e9de506bff25e26ce021a24f2fa8bb7637713360660256d9a0ab1e8`；首次日常 ON/OFF 启动后，COM6 证实先等待再运行，用户确认 K1 能打开视觉。用户再次自行拨 ON/OFF 后反馈“应该可以了”，今晚结束调试；关闭和重开细节未逐项回报。K2 控制代码未加入重启功能。旧版视觉 ELF、KModel、传感器程序均保留。

2026-09-26，用户再次自行通过右上角 ON/OFF 开机并测试，反馈“K1 2都能用”。此为两键在日常开机方式下的隔日现场确认；无新增串口日志。

最新独立 KEY2 候选由 `src/board_ui_overlay/build_key2_candidate.sh NEW_OUTPUT_DIRECTORY` 构建，DATA_APP 指向新隔离目录；构建不部署、不改 p1。当前进度见 [修复记录](../../model_work/docs/KEY2_CONNECTOR_CANDIDATE_20260930.md)。
