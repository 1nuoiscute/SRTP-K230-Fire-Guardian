# KEY2 数据页 connector 修复与部署｜2026-09-30

用户要求优先完成矩阵 KEY2，并已连接板子继续实机调试。本轮模型工作保持收尾。

## 新实机基线

ADB RTT_K230_ADB、COM5 Linux/COM6 RT-Smart 已确认。板端 p1、视觉 ELF、KModel、矩阵 KEY1 面板、20 分钟状态桥，以及 sensor_key/sensor_mvp_oled_keys 哈希与本地快照一致；S90/S91 服务存在，tmpfs 正常。采样仍关闭，状态有效位为 0，无 sensor_mvp.log，不能把数值 0 当作真实读数。

## 可复核的差异

失败数据页 ELF 的 connector_info_list 记录使用 v0.6 布局：有 buff_num，type 在偏移 108，屏幕宽 536，pclk 33000、pixclk_div 17，PHY n/m=3/64。此前只看结构 112 字节不足以证明 ABI 相同。

对已验收视觉 ELF（ec2caafd…）只读扫描 nt35516 字符串及数据段指针引用，发现文件偏移 0xac0a08 的 112 字节 connector 记录。它的 type 在偏移 104，末尾四字节为 padding；屏幕 hdisplay=540/vdisplay=960，pclk=39600、pixclk_div=14，PHY n/m=9/196，horizontal total620、vertical total1100。这与旧候选的字段和时序都不同。原显示记录使用的 536 缓冲画布与 540 屏幕时序是两个不同量，保持画布 536，不盲目改图像步长。

[提取 profile](../../docs/key2_connector_profile_20260930.json) 记录原 ELF SHA、偏移和 26 个 uint32。board_connector_profile.h 在当前进程重绑定 connector_name 指针，其余 104 字节原样构造；不使用旧库 kd_mpi_get_connector_info 的表。新 ELF 中完整 26-word payload 出现且仅出现一次。实际 ioctl/init 用途仍需实机显示验证，不能凭提取就宣称根因已彻底解决。

已核查：OSD 属性 stride=width×4/8、帧 stride=width 与本地 SDK OSD 样例一致；NT35516 connector 驱动的 init 路径自身启用 VO，不因缺显式 kd_mpi_vo_enable 就判错。没有修改相机、模型或正式 p1。

## 独立候选

构建脚本 src/board_ui_overlay/build_key2_candidate.sh，MPP=/home/k230/mpp_rtos_v06，工具链使用 SDK v1.6 RT-Smart RISC-V。项目源码 -Wall/-Wextra/-Werror；仅 SDK connector 自有警告降级，记录其 sign-compare 警告及 RT linker 的 RWX 提示。

- data_page_connector_candidate.elf：cbb468688ba9e610f5d2679a3306221942abc229b924e06694a80c5eb88e208a。
- vision_key_connector_candidate.elf：8d69a1c520b6ea6ad9f02e2a91060c3ac5caefae48806d4b938c6da89dce9f30。

本地输出 artifacts/key2_20260930_connector_candidate，源/输出哈希在 build_complete.json；板上隔离路径 /sharefs/srtp_clean/key2_20260930，推送后读回一致。已准备 SHA 1d503288… 的 v4b 恢复 ELF。未覆盖原数据页候选或正式面板。第一次提取检查误用 hdisplay 索引，保护检查在源码修改前拒绝；随后两次构建分别补齐基础 include 与板卡配置目录，成功前未启动候选。

新数据页默认 30 秒有界，可传 0 等待 q；控制器候选显式传 0。独立试验传 120 秒。初次串口发送后未立即看到 msh，保护检查拒绝启动；补查两串口、确认 list_process 无用户视觉进程，再执行新页。

## 第一次独立运行（历史）

独立数据页已实际运行，打印 verified-ELF profile 与 DATA 状态行，序号递增，history600；传感器仍关闭，因此不把 temp=0 当实测。启动时出现一次 vb_is_blk_valid/phys_addr=0 告警（原实时面板亦有类似历史告警），不能忽略；后续已抓日志中未见持续同类错误。120 秒后用空行确认返回 msh，已手动启动 v4b 恢复控制器，串口输出 K1 controls vision/initial state OFF，避免停在空终端。

当时等待实屏反馈，随后完成下述现场验收。

后续仍需重启、多轮切换与持续运行验收；异常时保留日志并恢复基线。

## 独立显示与矩阵 KEY2 现场验收

第二次独立页试验完整捕获 120 秒、60 条 DATA 更新，实际约 26.2–26.4°C、54–56% 湿度，20 分钟历史缓存为 600 点。到时返回 msh，恢复 v4b 控制器。用户明确确认“页面清晰，读数和曲线正常”。启动阶段仍有一次 VB pool 不存在告警，不能按画面清晰抹去日志风险。

矩阵试验启动 8d69a1c… 控制器，完整串口记录一次摄像→数据→摄像闭环：停止 UI/原视觉，启动数据页，9 条新读数，数据子进程 status=0，原视觉和面板恢复。用户反馈“非常好 key2做的非常完美”。这确认实际使用效果；日志只捕获一次完整闭环，不写成已证明两轮压力测试。第二个全新目录重构两个 ELF，SHA 与第一构建完全相同。

首次摄像启动（此前连续做独立数据页试验）出现 93 条 free-buffer 告警，回摄像后 1 条；两次摄像启动均有 vi wait-stop-timeout。没有数据页崩溃或退出错误，但资源生命周期和连续运行仍待新开机验证。原视觉 ELF 与模型不变，不把这些警告归为模型训练问题。

## 临时试验结束恢复造成 KEY2 失效

240 秒有界试验结束后，电脑侧脚本退出候选并自动恢复历史 v4b。用户随后发现 KEY2 不再响应；这是恢复到不含矩阵 KEY2 的旧控制器，不是已验收页面突然失效。恢复步骤没有及时向用户说明，现改为优先完成永久启动部署和 Git 冻结。KEY1 蓝框压住参数面板的问题保留；叠加顺序仅做只读核对，没有修改寄存器，修复按用户要求暂缓。

## 开机镜像准备

通过原始 p1（83f8286a…）和原厂 fastboot ELF（025e7950…）哈希保护，仅替换 ROMFS fastboot 槽位，其他解压字节保持一致；新镜像 20 MiB，SHA **5cd8667b3035199fcc63d7e557c8257247bc0080f2b9cda7e3a916a3616ef1d2**。保留当前 v4b p1（5fec5ccf…）及其恢复控制器；数据页路径仍是已验收隔离目录，不重建或重命名已测试 ELF。

源码、构建产物和现场边界见 [版本清单](../../docs/key2_release_identity_20260930.json)。当前为已完成本地打包，设备写入、重启和重启后按键结果将在下一段补记，不提前宣称已完成。
