# SRTP K230 Fire Guardian｜边缘视觉火情检测与多传感器融合

**2026-10-28 模型推进：**完成火焰 v21（12 轮）及烟雾扩充 v5、后段特征 v6（各 40 轮）的训练与固定对照。v21 紧框收益伴随误检代价，保留 v16 演示/v18 开发；新增 v6 last 独立烟雾网络开发候选，旧双头仍保留 v4。均为已曝光开发反馈，未完成独立测试或上板。[本轮记录](docs/VISION_RESUME_20261028.md)｜[交接](model_work/docs/VISION_HANDOFF_20261028.md)。

基于 **K230 + RT-Thread Smart / Linux** 的厨房火情检测研究原型，结合摄像头火焰检测、环境传感器、板端显示与阀门联动。项目公开源码、实验记录和已知问题。

**2026-10-27 更新：** 整合 [PR #2](https://github.com/1nuoiscute/SRTP-K230-Fire-Guardian/pull/2) 的 ADS1115 气体/火焰采集、OLED 与大屏显示、阀门联锁、WiFi 及开机启动脚本。电脑端模型记录与本次板端代码统一收录到 `main`，接手时从默认分支开始。

## 当前能力与验证状态

| 能力 | 当前实现 | 验证范围 |
|---|---|---|
| 摄像头火焰检测 | 早期四类映射 640 KModel + 历史视觉 ELF，板端摄像头画框 | 已有历史实屏证据；跨厨房可靠性仍未验收 |
| 环境采集 | SHT31 温湿度、BMP280 温度/气压、SSD1306 OLED | 已有实机记录；OLED 刷新故障不再直接终止采样循环 |
| 气体与火焰输入 | ADS1115 `0x49`，AIN2 接 MQ-2、AIN1 接火焰模块；输出 mV | 队友报告 MQ-2 实测响应；火焰模块尚无可用火焰响应 |
| 实时面板与数据页 | KEY1 面板、KEY2 数据页读取 v1/v2 状态；桥接每 0.5 s 刷新，历史保留约 20 分钟 | 旧 KEY2 有重启及切换确认；新增显示见队友 PR 记录，未在此次整合中复验 |
| 阀门联动 | 气体、火焰、温度阈值及温升判断；迟滞；D2/pad 63 继电器控制 | 队友报告约 12 s 阀门行程、断电自关和温度联动；未完成系统故障验收 |
| WiFi | 2.4 GHz 连接、扫描、诊断与开机启动 | 队友报告可 ping 公网；MQTT 与远程告警尚未实现闭环 |
| 电脑端模型 | v16 best 演示、v18 last 定位开发；v21 正式对照完成；烟雾新增 v6 last 独立网络开发候选 | 2026-10-28 恢复推进；均为开发反馈，新权重尚未上板 |

本次进展与数据来源见 [气体、阀门与联网整合记录](docs/GAS_VALVE_WIFI_INTEGRATION_20261027.md)。历史结果按记录日期理解，不代表当前设备已重新验收。

## 数据与控制链

```text
MQ-2 / 火焰模块 → ADS1115 ┐
SHT31 / BMP280 ──────────┴→ Linux 采样日志 → status_bridge（v2）
                                            ├→ RT-Smart KEY1 面板 / KEY2 数据页
                                            └→ fire_guard → 继电器 → 常闭阀门
摄像头 → KModel / 视觉 ELF → 检测框显示（尚未接入上述关阀判断）
```

阀门默认气体关断门限为 **650 mV**、恢复门限为 **490 mV**；温度达到 **60℃** 或约 **30 s 升高 8℃**也触发关断。读数满足恢复条件后会自动重新开阀。配置见 [srtp_fireguard.conf](src/key_controls/srtp_fireguard.conf)，实际判断见 [fire_guard.sh](src/key_controls/fire_guard.sh)。这些是当前原型参数，尚无标准气体浓度标定。

## 从这里开始

```sh
git clone https://github.com/1nuoiscute/SRTP-K230-Fire-Guardian.git
cd SRTP-K230-Fire-Guardian
git switch main
```

已有克隆可执行 `git fetch origin`，确认工作区改动后使用 `git switch main`、`git pull --ff-only origin main`。`codex/pc-model-20261002` 保留为开发分支。

| 内容 | 入口 |
|---|---|
| 最新板端整合、阈值与已知问题 | [2026-10-27 整合记录](docs/GAS_VALVE_WIFI_INTEGRATION_20261027.md) |
| 传感器接法及构建 | [sensor_mvp](src/sensor_mvp/README.md) |
| 按键、阀门、WiFi 与自启脚本 | [key_controls](src/key_controls/README.md) |
| 面板与数据页 | [board_ui_overlay](src/board_ui_overlay/README.md) |
| Windows 大核构建 | [Zig + LLD 构建说明](tools/rt_smart/README.md)；脚本仍需配置本机 SDK 路径 |
| 历史板端基线与完整哈希 | [板端基线](docs/BOARD_BASELINE.md)、[9 月交接](docs/BOARD_HANDOFF_20260930.md) |
| KEY2 修复与永久启动 | [实机记录](model_work/docs/KEY2_CONNECTOR_CANDIDATE_20260930.md) |
| 模型进展与可运行入口 | [项目状态](docs/STATUS.md)、[电脑端复现](docs/REPRODUCE_PC.md) |
| 当前电脑端候选 | [v16 演示](model_work/docs/SCRATCH_SEMANTIC_V16_20261003.md)、[v18 定位开发](model_work/docs/FROZEN_FLAME_HEAD_V18_20261003.md)、[v21 正式结果](model_work/docs/KITCHEN_PARTIAL_FEATURES_V21_RESULTS_20261028.md)、[烟雾 v5/v6](model_work/docs/SMOKE_EXTENSION_V5_V6_RESULTS_20261028.md) |
| 融合方案与项目路线 | [事件融合计划](docs/BOARD_EVENT_FUSION_PLAN_20261001.md)、[路线](docs/ROADMAP.md) |
| 数据、训练、失败实验 | [模型总报告](model_work/docs/MODEL_TEST_AND_TRAINING_REPORT_20260927.md)、[完整阶段状态](docs/STATUS.md) |
| 第三方资料与许可 | [参考资料](references/README.md)、[第三方说明](THIRD_PARTY_NOTICES.md) |

## 已知限制

- 阀门判断尚未使用摄像头检测框，正常灶火与异常火情的完整事件融合未验收。
- 火焰模块已接入 ADC，但当前配置注明其对火焰尚无可用响应；通道有电压不等于探火成功。
- 当前守护仅在气体、火焰、温度三路全部无效时执行无数据策略；单路失效和状态桥接停止更新尚无完整保护。
- MQ-2 大屏的 0～100% 为 250～500 mV 的相对刻度，OLED 报警门限为 500 mV，与阀门 650 mV 门限尚未统一；不是 ppm 或 %LEL。
- 电脑端候选仍有蓝焰漏检、锅体大框和非火对象误检；未通过独立跨厨房、完整事件或新权重上板验收。板端 ROI 选区尚未实现。
- KEY1 检测框覆盖面板的问题没有新的关闭证据；长时稳定性、执行器位置反馈、MQTT 闭环仍待补测。

## 源码与运行产物

`src/` 保存板端代码；`model_work/src/` 保存电脑端数据处理、训练、诊断与回放脚本；`model_work/docs/` 和 `docs/` 保存实验与交接记录；`tools/` 保存检查及构建工具。

**仓库不包含权重、KModel、ELF、固件备份和原视频。** 新克隆需按对应记录准备产物。历史视觉 ELF 缺少同源源码，`src/kitchen_fire_yolov8/` 不是已经复现摄像兼容性的正式视觉源码；编译成功不等于可直接替换已验收程序。接手设备先读回现有产物并核对哈希，历史身份见板端基线。

## 验证

```sh
python -B tools/check_public_repo.py
python -B tools/check_key1_draw_path.py --self-test
python -B -m unittest discover -s model_work/src -p 'test_*.py' -v
```

阀门逻辑另附 [19 例判据测试](src/key_controls/test_fire_guard.sh) 和 [5 例温升计算测试](src/key_controls/test_temp_rise.sh)。两者使用模拟输入，不能替代实机验收；测试脚本的路径适配方法见整合记录。

## Contributors

| 贡献者 | 工作 |
|---|---|
| [@1nuoiscute](https://github.com/1nuoiscute) | 项目维护、视觉模型实验、板端整合与研究记录 |
| [@rqw1425](https://github.com/rqw1425) | [PR #2](https://github.com/1nuoiscute/SRTP-K230-Fire-Guardian/pull/2)：ADS1115 气体/火焰采集、阀门联动、WiFi、自启与 Windows RT-Smart 构建工具 |

完整提交贡献见 [GitHub Contributors](https://github.com/1nuoiscute/SRTP-K230-Fire-Guardian/graphs/contributors)。
