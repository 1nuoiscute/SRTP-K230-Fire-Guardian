# 板载 UI 独立叠加短测（2026-09-28）

目标：保留已能显示摄像头和检测框的厂商视觉 ELF，用 RT-Smart 独立 OSD0 在屏幕下部叠加 536×348 状态面板，测试成功后再考虑实时刷新及 K1 生命周期集成。

## 基线与文件

- 原视觉 ELF：`/sharefs/srtp_clean/vendor_ob_det/ob_det_fire_mvp_spaces.elf`，SHA-256 `ec2caafd85e56851c44c838b4c9a2a15477cae8b24ac915f1c571f1a0f9a2ac1`。
- 原模型：`/sharefs/srtp_clean/vendor_ob_det/best_vendor80_640.kmodel`，SHA-256 `ec475725794417e83b58c661769a915c7a0e7a71ed61a6e2d267ddc7e49096a7`。
- 两者及已有屏幕照片已存档于 `artifacts/ui_baseline_2026-09-28/`；本轮没有覆盖原文件、模型或刷写启动分区。
- 新面板 `artifacts/board_ui_overlay/panel.png` 是电脑渲染的静态快照，板端原始 BGRA 在 `/sharefs/srtp_clean/ui_overlay/panel.bgra`，746112 字节，SHA-256 `022033b7ffcde137d961212052ff485af66659e9998378aeceb86bb1d51f9331`。
- 修正版探针板端 `/sharefs/srtp_clean/ui_overlay/ui_osd_probe_vb.elf`，SHA-256 `04d46c443ca00db9ec25c1a2aacbc5d72dffe0bee4bed3e9858237ffa9edc626`。PC 与板端哈希相符。

## 数据口径

生成面板时，Linux `/tmp/sensor_mvp.log` 在 1 秒内持续增长，SHT31 为 25.7°C、64.9%RH，BMP280 气压 950.0 hPa（屏幕保留一位小数）。主板 `wlan0` 无 IPv4、显示“未连接”；RW007 未测。烟雾未接入、可燃气体/CO 预留，火情判断显示“暂未判定”。这些数值属于当时快照，不会在当前 5/30 秒 OSD 进程中自动更新，也不能用来做火灾报警判定。板端时钟不准确，时间以 PC 采集时间为准。

## 短测结果与故障

1. COM6 调试口恢复。原 K1 控制器占用前台时，输入 `q` 加回车使原视觉走 `vicap_stop_stream`、ISP 停流和 VB 释放，串口回到 `msh`。随后以原参数在 msh 后台启动原 ELF：`./ob_det_fire_mvp_spaces.elf best_vendor80_640.kmodel 0.5 0.6 None 0 &`。串口见 GC2093 初始化和约 220 ms/轮持续推理；`list_process` 有原视觉进程。
2. 首版探针用 `kd_mpi_sys_mmz_alloc` 物理地址直接插入 OSD0，未设置有效 VB pool。板端打印 `vb_is_blk_valid ... assertion failed`、`Exception 13: Load Page Fault`，仅终止探针；原视觉继续推理。此失败不可省略。
3. 改为 `kd_mpi_vb_create_pool` / `kd_mpi_vb_get_block` / `kd_mpi_sys_mmap`，并设置 `frame.pool_id` 后重新编译、单独上传。5 秒测试打印 `OSD0 panel visible for 5 seconds` 并返回 msh；30 秒测试也打印 `OSD0 panel visible for 30 seconds`，之后 `list_process` 不再有探针，原视觉仍有进程和持续推理日志。
4. 第二次测试串口出现 `vb_user_add ... try to increase counter for a free buffer`，说明仍需审查 OSD0 帧与 VB 释放时序。未得到现场屏幕目视反馈，因而不能确认实际面板位置、颜色、遮挡与检测框是否正确。
5. 短测后通过 RT-Smart `reboot` 恢复常规 K1 启动路径，并清除临时手动启动的视觉进程及 OSD 状态。ADB 重新连上；原视觉 ELF 和模型的板端哈希仍与上述基线完全相同。Linux `sensor_key` 控制器在运行；传感器采样子程序需要按原 K2 流程开启。串口重连时未抓到启动日志，所以本轮只确认 ADB、原文件与小核控制器，不扩写为 K1/K2 实体按键复验。

## 当前边界与下一步

首次测试达到“文件上板、短时叠加调用返回、原视觉进程存活”的级别，当时尚未获得屏幕目视结果；同日复测见下文。电脑脚本 `src/board_ui_overlay/render_panel.py` 每次只生成一次快照。仍需处理 `vb_user_add` 警告和缓冲区生命周期，之后设计跨核状态通道与实时刷新，并与 K1 启停集成。任何长期运行版都须验证画面、框、帧率、退出和回退。

## 同日现场复测

用户上次没有看屏幕，要求重新展示。我再次通过 COM6 退出 K1 控制器并在 msh 后台启动原视觉 ELF；GC2093 初始化和持续推理日志出现。随后运行 `ui_osd_probe_vb.elf panel.bgra 30`，串口打印 `OSD0 panel visible for 30 seconds`。用户在显示窗口内明确反馈“面板出现，画面正常”，因此**面板出现、摄像头画面正常**获得现场目视确认。用户选项中同时包含“蓝色框是否正常”，但他未单独说明测试时是否恰好有可检测目标；不能据此声称框坐标已逐项验收。30 秒后 `list_process` 只见原视觉进程、不见探针。随后发送 `reboot` 退出手工后台演示，恢复 K1 日常启动流程。面板内容仍是静态旧快照，不代表实时数据接入。

下一步改为先解决板端持续刷新与跨核状态有效期，再将 OSD 生命周期接入 K1 控制器；在此之前不把 30 秒短测称为正式部署。
