# 板端基线与源码边界

**当前正式版本：**KEY2 控制器 8d69a1c5…、数据页 cbb46868…、启动 p1 5cd8667b…。写后及实际重启读回一致，用户确认重启后两轮切换正常。源码/产物/现场边界见 [KEY2 版本身份](key2_release_identity_20260930.json) 和 [修复部署记录](../model_work/docs/KEY2_CONNECTOR_CANDIDATE_20260930.md)。短时验收不等于长时可靠性。

以下表格保留 2026-09-28 的历史 v4b 回退身份，2026-09-30 已冻结快照并重新读回实机。原视觉、模型与 KEY1 面板继续沿用。完整路径见 [产物清单](board_handoff_artifacts_20260930.json)，推进入口见 [板端交接](BOARD_HANDOFF_20260930.md)。

| 产物 | SHA-256 | 身份 |
|---|---|---|
| 原视觉 ELF | ec2caafd85e56851c44c838b4c9a2a15477cae8b24ac915f1c571f1a0f9a2ac1 | 板端视觉 MVP |
| 640 KModel | ec475725794417e83b58c661769a915c7a0e7a71ed61a6e2d267ddc7e49096a7 | 早期四类映射模型 |
| v4b K1 控制器 | 1d503288d14bdcdbd6e3feb3e0faabcced8cab210fbc66db7b03a023ee46168c | 已验收启动控制器 |
| v4b p1 镜像 | 5fec5ccf0921e790357f523ba7b7a49dd177e101358ac86a8b55a85895264fa1 | 启动分区读回一致 |
| 矩阵 KEY1 面板 | 6dd398c1deea0d99721b44066f71df1cfe5801815d88740d9071ffaaaf9f9c46 | 已验收面板 |

**当前 vision_key.c 对应新 KEY2 控制器，仍不能声称它就是历史 v4b 的原始源码。** build.sh 现要求显式 --data-candidate，并仅输出候选文件名。历史脚本保留于本地归档，不运行自动烧录。

正式镜像、原始备份与候选二进制继续保留在本地 artifacts 目录。恢复前重新核对真实设备、分区和哈希；公开 Git 不包含这些二进制。原厂视觉 ELF 缺少同源源码，SDK 替代路径有历史 ABI/显示兼容问题，这是当前真正分屏的工程限制。
