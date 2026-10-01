# SRTP K230 Fire Guardian｜边缘视觉火情检测与多传感器融合

面向厨房场景的大学生 SRTP 研究原型。基于 **K230 + RT-Thread Smart / Linux**，开展火焰目标检测、环境采集、事件判断与模拟联动研究。

**当前阶段：电脑端模型改进暂时收尾，转向板端现有 MVP 的复测、显示与功能整合。** 正常灶火与异常火情需要分开判断；当前模型尚未通过跨厨房可靠性验收。

演示重点已按用户要求调整：采用无贴纸干扰的灶台，贴纸抑制作为可选改进，优先真实火焰定位和连续识别。见 [当前演示范围](docs/DEMO_SCOPE.md)。

## 已有成果

- 早期四类视觉模型已完成 KModel 转换与板端摄像头画框演示。
- SHT31 温湿度、BMP280 温度/气压、SSD1306 OLED 有实机验证。
- 副板 K1 控制视觉与实时面板，K2 控制传感器；底板矩阵 KEY1 控制面板显隐。
- 小核通过 tmpfs 状态文件向大核传递传感器读数；已有 20 分钟内存历史缓存。
- 电脑端具备数据来源审核、重复检查、训练、逐图与视频诊断脚本，以及可解释的融合状态机。

## 当前限制

- 2026-09-28 新厨房视频暴露锅下蓝焰漏检、蓝色贴纸误检；电脑端 v3 是开发基线，新权重尚未上板。
- KEY2 全屏数据页的旧乱码已在 9 月 30 日短时实屏试验通过修复；新开机版本已写入并通过启动/哈希复测，长时稳定性未验收。KEY1 蓝框覆盖参数区仍未解决。
- 烟雾/燃气输入、真实事件融合、自适应标定、执行反馈和 MQTT 闭环尚未验收。
- 旧图片集和已反复分析的视频均属于开发诊断；其帧数、mAP 或有框比例不能替代独立事件验收。

## 模型阶段收尾（2026-09-30）

按用户要求停止追加模型实验，保留 v3 开发基线及 v5–v11 失败结果。新片段的未完成审核/推理已暂存，不自动继续。见 [模型收尾](docs/PC_MODEL_STAGE_CLOSURE_20260930.md) 和 [板端交接](docs/BOARD_HANDOFF_20260930.md)。收尾提交仅核对本地产物；随后已连接板子，KEY2 页面和矩阵切换获用户确认，见 [KEY2 修复与部署](model_work/docs/KEY2_CONNECTOR_CANDIDATE_20260930.md)。

### 已完成实验

v5、v6 均已完成训练与对照。v6 减轻了旧图退化，但仍有贴纸误报，外部蓝焰定位未改善，继续保留 v3。v7 从 v3 开始、沿用 v6 参数，仅将两份审核过的既有蓝焰诊断原作加入训练，训练与按来源曝光分组的评测均已完成：last 在五张蓝焰诊断图上由 0/5 到 1/5，但旧图 mAP50 降至 0.906 且仍有误检，拒绝替换。详见 [v5 记录](model_work/docs/HARDCASE_V5_EXPERIMENT_20260930.md)、[v6 完整结果](model_work/docs/HARDCASE_V6_EXPERIMENT_20260930.md) 和 [v7 实验](model_work/docs/HARDCASE_V7_EXPERIMENT_20260930.md)。来源边界更正见 [原作对照](model_work/docs/COMMONS_SOURCE_SCOPE_CORRECTION_20260930.md)。

v8 已完成：冻结 11 个主干模块确实保住旧图诊断，但蓝焰仍为 0/5，难例适配不足，拒绝替换。随后 v9 增补了审核后的蓝焰与食物难负例，见 [v8 实验](model_work/docs/HARDCASE_V8_FREEZE_EXPERIMENT_20260930.md)。

v9 已完成：last 五图蓝焰定位提高到 3/5，但旧图 mAP50 降到 0.900 且仍有贴纸误检，拒绝替换。此次仅增补八张蓝焰照片与一个食物难负例，训练与完整对比身份已保存，见 [v9 实验](model_work/docs/HARDCASE_V9_SOURCE_DATA_EXPERIMENT_20260930.md)。


原训练集颜色/标签抽查完成：所抽厨房图中蓝焰并不少见，但场景重复且部分框包含较大锅体；通用来源全部 4 个空标签图中 3 个实际有可见火焰，已追溯为上游漏标。尚未修改 v9 或原标签。见 [抽查证据](model_work/docs/TRAINING_COLOR_QUALITY_REVIEW_20260930.md)。

v10 已完成：蓝焰仍为 3/5，食物误检重新出现，暂不采用整来源移除方案；按来源复查发现烟雾/反光误检增加，见 [来源消融](model_work/docs/HARDCASE_V10_SOURCE_ABLATION_20260930.md)。

v11 已完成：回到 v9 数据，隔离五张问题图、加权三张原训练烟雾无可见火焰图。last 蓝焰仍 3/5，烟雾来源误检为 10 框且有食物/关火反光误检，不采用。贴纸不是否决理由，见 [完整结果](model_work/docs/HARDCASE_V11_NEGATIVE_CURATION_20260930.md)。[新来源短视频探查](model_work/docs/PUBLIC_VIDEO_PROBE_20260930.md) 已完成煮饭片段：36个无可见火焰采样帧四模型均0误检；船上灶台暂存，未完成火焰判读，不把短片结果当成完整事件验收。

## 从这里开始

| 内容 | 入口 |
|---|---|
| 当前板端交接 | [上板顺序与版本边界](docs/BOARD_HANDOFF_20260930.md)、[模型阶段收尾](docs/PC_MODEL_STAGE_CLOSURE_20260930.md) |
| 报警、融合与联动方案 | [板端实施顺序与输入缺口](docs/BOARD_EVENT_FUSION_PLAN_20261001.md) |
| 最新状态及历史 | [项目状态](docs/STATUS.md)、[路线与实验门槛](docs/ROADMAP.md) |
| KEY1 蓝框问题 | [短时叠加试验与裁剪方案](model_work/docs/KEY1_BLUE_BOX_INVESTIGATION_20261001.md) |
| 板端实时面板 | [实机记录](model_work/docs/BOARD_UI_LIVE_INTEGRATION_20260928.md) |
| KEY2 数据页与部署 | [最新修复](model_work/docs/KEY2_CONNECTOR_CANDIDATE_20260930.md)、[历史失败](model_work/docs/BOARD_DATA_PAGE_KEY2_20260928.md) |
| 新厨房视频失败 | [冻结模型测试](model_work/docs/TEAMMATE_NEW_VIDEO_TEST_20260928.md) |
| 训练与数据沿革 | [模型总报告](model_work/docs/MODEL_TEST_AND_TRAINING_REPORT_20260927.md)、[视频适配](model_work/docs/USER_VIDEO_ADAPTATION_RUN_20260927.md) |
| 电脑端运行方法 | [复现指南](docs/REPRODUCE_PC.md) |
| 有用参考资料 | [参考入口](references/README.md)、[本地资料清单](references/LOCAL_MATERIALS.md) |
| 源码与版本边界 | [板端版本身份](docs/BOARD_BASELINE.md)、[第三方说明](THIRD_PARTY_NOTICES.md) |

## 目录

`src/` 保存板端代码与实验候选；`patches/` 保存历史 SDK 适配补丁；`model_work/src/` 保存电脑端数据、训练、诊断与回放脚本；`model_work/docs/` 保存实验记录；`docs/` 是公开导航与复现说明；`references/` 是参考资料索引与使用笔记。

本地大文件、原视频、私有数据、权重、固件备份和个人申报材料保留在原路径，不加入普通 Git。数据与固件的身份由各实验清单和 SHA-256 记录。部分旧脚本含历史机器路径，请按复现指南配置；新实验脚本提供路径参数。

## 验证

```sh
python -B -m unittest discover -s model_work/src -p 'test_*.py' -v
python -B tools/check_public_repo.py
```

该仓库公开研究过程与失败记录。第三方代码、数据及资料遵守各自许可，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
