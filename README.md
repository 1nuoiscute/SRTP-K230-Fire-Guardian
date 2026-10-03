# SRTP K230 Fire Guardian｜边缘视觉火情检测与多传感器融合

**最新电脑端进展v19（2026-10-03）：**[训练全场景与完整比较](model_work/docs/TRAIN_CONTEXT_V19_RESULTS_20261003.md)已完成12轮/3510次更新，七模型图片及9视频307同帧比较、26页实际复核、两份ONNX各80/80及last独立CPU视频27/27通过。last紧框10/15→11/15、补充FP5→3，但蓝焰分量漏检和挂件错误仍在、旧空标签来源预测18→25框；保留v16 best演示与v18 last开发候选，v19仅作取舍参照。没有独立厨房或板端验收。

**最新电脑端进展v18（2026-10-03）：**[固定特征检测头训练与完整比较](model_work/docs/FROZEN_FLAME_HEAD_V18_20261003.md)已完成12轮/3198次实际更新，23个特征模块及BN缓冲、保存权重实际核对保持。紧火焰匹配较v16从6/15提高到10/15、蓝焰保持4/5；五模型9视频307同帧、20页实际复核、两份ONNX各80/80及last独立CPU视频27/27一致性通过。**v18 last作为下一定位开发候选，当前演示保留v16 best**；挂件误检、锅底漏检/大框、重复框与旧AP代价已记录。152项本机测试通过，未连接开发板或独立验收。

**前一阶段v17（2026-10-03）：**[v17训练、图片/视频比较与独立CPU验证](model_work/docs/FLAME_EXTENSION_V17_20261003.md)已完成12轮/3198次实际更新。新增审核17图/44个物理火焰框、15张无可见火焰训练图提高权重；补充紧火焰匹配6/15→12/15，last保持4/5蓝焰。三模型9视频307同帧比较、28页实际复核、best/last ONNX各80/80及last独立CPU视频27/27一致性通过。**该阶段演示保留v16 best，v17 last作为定位开发候选**；新增彩色厨具误检、重叠框、一个采样蓝焰漏检和旧召回退化均已记录。没有独立厨房或板端验收，开发板未连接。

面向厨房场景的大学生 SRTP 研究原型。基于 **K230 + RT-Thread Smart / Linux**，开展火焰目标检测、环境采集、事件判断与模拟联动研究。

**前一阶段v16（2026-10-03）：**[v16训练、项目比较与CPU运行](model_work/docs/SCRATCH_SEMANTIC_V16_20261003.md) 已完成。best保持4/5蓝焰，补充物理火焰框匹配0→6处，锅底小蓝焰视频表现保持；动态ONNX80/80及独立CPU入口通过，作为下一电脑端开发候选。旧AP下降、无火误检及锅体大框仍记录，v14/v13回退保留；尚无独立厨房或板端验收。下文为此前阶段沿革。

**当前阶段（2026-10-02）：v14从零训练和v15语义适配均完成，优先推进v14锅底蓝焰电脑端候选，开发板未连接。** 已有fire v13保留回退：它相比v3蓝焰主定位0/5→1/5、队友已审核火焰TP 3→5，旧火焰召回及室内无火FP保持，旧mAP50 .97324→.97128，静态/动态ONNX各80/80及独立CPU入口通过。v3也保留；跨厨房可靠性和事件验收仍未完成。见 [v13模型卡](docs/PC_FIRE_V13_MODEL_CARD.md)。

此前 v12 来源采样训练的两份候选均未达到冻结门槛；本轮 v13 改用固定特征与教师约束后首次通过开发筛选。已对照新版计划书建立差距清单，烟雾/蒸汽语义和独立两类验收仍缺证据，原数据和旧权重保留。

已完成 [v14随机初始化100轮与v15语义适配12轮、项目视频比较](model_work/docs/PROJECT_MODEL_ROUTES_20261002.md)：v14蓝焰主定位4/5，五模型在307个共享视频画面上比较，27个固定画面实际复核支持其锅底小蓝焰收益；旧无火来源误检18/19，锅体范围问题仍未解决。按项目用途综合判断，优先推进v14 last，保留v13回退。动态ONNX 80/80图一致性与独立CPU图片/视频入口通过。

[可见火焰标签适配v15](model_work/docs/TRAIN_VISIBLE_FLAME_PILOT_20261002.md) 的last补充火焰匹配由1处升至5处，蓝焰1/5→3/5，但嵌套重复框增多，实际视频小蓝焰收益弱于v14；保留此路线继续修正物理框口径。旧AP变化作为代价记录，不再要求全部旧指标同时改善。

已完成 [fire v13与smoke v4实际组合](model_work/docs/JOINT_FIRE_V13_SMOKE_V4_20261002.md)：116图保留两来源输出、61图静态ONNX一致性及14图独立CPU入口通过。矩形策略下开发smoke为2 TP/0 FP/1 FN，但正方形入口多1个smoke FP；组合仅供两类开发，其fire来源保持v13，不能直接移植到已改变特征的v14，没有独立厨房或板端验收。

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

以下保留 9 月 30 日暂停时的决定；10 月 2 日用户重新授权模型工作，见 [最新电脑端推进](model_work/docs/PC_MODEL_PROGRESS_20261002.md)。当时保留 v3 开发基线及 v5–v11 失败结果，未完成素材审核暂存。见 [模型收尾](docs/PC_MODEL_STAGE_CLOSURE_20260930.md) 和 [板端交接](docs/BOARD_HANDOFF_20260930.md)。此后 KEY2 页面和矩阵切换获用户确认，见 [KEY2 修复与部署](model_work/docs/KEY2_CONNECTOR_CANDIDATE_20260930.md)。本轮没有连接或操作板子。

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
| 最新模型推进与可运行入口 | [v13 模型卡与 ONNX 身份](docs/PC_FIRE_V13_MODEL_CARD.md)、[v13 完整实验](model_work/docs/FIRE_TEACHER_V13_20261002.md)、[v3 回退模型卡](docs/PC_FIRE_V3_MODEL_CARD.md) |
| 持续模型训练与计划要求 | [v12 完整结果](model_work/docs/HARDCASE_V12_SOURCE_CAP_20261002.md)、[新版计划书证据矩阵](docs/PLAN_REQUIREMENT_MATRIX_20261002.md)、[烟雾标签审核](model_work/docs/SMOKE_SOURCE_SEMANTICS_20261002.md) |
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
