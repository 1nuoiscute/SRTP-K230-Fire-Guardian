> **2026-09-30 最新：** KEY2 新 connector 页和矩阵切换已获用户实屏确认；新启动镜像已写入并实际重启，启动与读回一致；用户确认重启后两轮实体切换正常。历史 v4b 回退与旧失败候选继续保留。见 [修复与部署记录](../../model_work/docs/KEY2_CONNECTOR_CANDIDATE_20260930.md) 和 [版本身份](../../docs/key2_release_identity_20260930.json)。以下旧“未验收”段落保留为历史边界。

# 独立 OSD 探针（2026-09-28 板端短测）

**最新状态（同日）：**本目录的 `overlay_live.c` 已改为单 VB 缓冲区、只在启动时向 OSD0 插入一次帧，之后在映射内存中更新读数；Linux `status_bridge.c` 通过挂载在 `/sharefs/srtp_clean/ui_overlay/status_ram` 的 tmpfs 将实时传感器状态送至大核。板端重启已验证 `/etc/init.d/S91srtp_ui_bridge` 自启，K1 联动面板版启动镜像已写入并核对哈希。用户已目视确认新镜像下大屏参数会变化，副板 K1 能让画面和面板一起关闭、再次一起恢复；长时间稳定性以及真正分屏仍未完成验收。完整实机边界见 `model_work/docs/BOARD_UI_LIVE_INTEGRATION_20260928.md`。矩阵 KEY2 全屏数据页为**尚未写入开机镜像的候选实现**，其失败探针、独立显示、历史缓存和待验收边界见 `model_work/docs/BOARD_DATA_PAGE_KEY2_20260928.md`。

同日晚些时候根据用户指出的底板 4×4 按键矩阵，`overlay_live.c` 又加入矩阵 KEY1（GPIO28 行 / GPIO18 列）对 OSD0 面板的显隐控制。90 秒有界板端试验以及部署后的重启复测均获用户现场确认；旧 ELF 已备份，新版文件哈希见上述实机记录。副板 K1 仍负责视觉和面板总开关，K2 仍负责传感器；真正分屏与框区域裁剪未实现。

## 早期静态探针历史

以下仅描述早期静态探针，当时的“尚待确认”不覆盖上方已完成的实时面板目视验收；KEY2 数据页失败仍未解决，见 [当前板端交接](../../docs/BOARD_HANDOFF_20260930.md)。

这不是替换视觉程序的版本。它只尝试在已由原视觉程序初始化的 VO 上启用 `OSD0`，把 536×348 的 BGRA 面板放在屏幕底部，1–30 秒后自动关闭。原视觉程序据现有源码使用 `OSD3` 画框；但厂商 ELF 没有同源源码，**OSD 层是否可同时运行仍须实机验证**。

不调用 `kd_mpi_vo_init`、`kd_mpi_vo_enable`、`kd_mpi_vb_set_config`、`kd_mpi_vb_init`，也不修改相机、模型、K1/K2 或原 ELF。通用 SDK 的 MPP 库与本板定制固件存在既往 ABI 不匹配史；此探针只能在串口可观测、原程序可回退时短测。输出目录采用新文件名，绝不覆盖旧版本。

输入 `panel.bgra` 必须恰好 `536×348×4 = 746112` 字节，按 OpenCV/小端 ARGB 对应的 BGRA 字节顺序排列。程序在传入的秒数后自动关闭 OSD0。`render_panel.py` 在电脑读取板上当时的传感器日志与主板 Wi-Fi 状态，生成静态快照；这个测试还不是板端自主刷新的 UI。

首次 MMZ 版本在 `kd_mpi_vo_chn_insert_frame` 被 `vb_is_blk_valid` 拒绝，发生 `Exception 13`，进程被板端终止，原视觉继续推理。修正版使用独立 VB pool/block 并填 `frame.pool_id`，以 `ui_osd_probe_vb.elf` 新文件名上板；5 秒和 30 秒测试均打印 `OSD0 panel visible` 并返回 msh，原视觉仍在运行。串口另出现一次 `vb_user_add ... free buffer` 警告，须进一步检查缓冲区生命周期。屏幕的实际位置、颜色和叠加效果尚待现场目视确认；不能把串口成功当作显示验收。完整记录见 `model_work/docs/BOARD_UI_ON_DEVICE_TRIAL_20260928.md`。

## 2026-09-30 KEY2 connector 修复候选

已发现旧数据页 connector 记录与已验收视觉 ELF 的字段布局/时序不同，按已验收记录构造新 payload，独立 120 秒试验已运行并回到 msh 后恢复 v4b；实屏及矩阵切换仍待确认。构建命令 `sh src/board_ui_overlay/build_key2_candidate.sh NEW_OUTPUT_DIRECTORY`，输出新目录并记录身份，不自动上板。见 [修复记录](../../model_work/docs/KEY2_CONNECTOR_CANDIDATE_20260930.md)。

## 2026-10-01 KEY1 限时图层候选

`overlay_mix_candidate.c` 与 `key1_mix_guard.h` 是独立、限时的图层候选；`overlay_live.c` 原样链接，通过 OSD0 enable/disable wrappers 调整并恢复顺序。构建脚本 `build_key1_mix_candidate.sh NEW_OUTPUT_DIRECTORY` 不部署、拒绝已有目录；候选要求显式1–300秒，不能直接代替正式开机面板。两次构建一致，真实显示候选未运行；屏幕试验反馈仍待答。详情见 [KEY1研究](../../model_work/docs/KEY1_BLUE_BOX_INVESTIGATION_20261001.md)。
