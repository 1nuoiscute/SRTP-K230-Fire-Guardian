# 厨房火焰模型近期训练与测试总报告（2026-09-27）

## 1. 当前结论

**四段实拍视频已用于后续针对性训练；当前保留视频适配 v3 为电脑端开发候选，尚未通过新视频独立验收。** 原补标 v2 在原 286 张静态测试集上 mAP50 比 v1 高；同参数无补标对照版又优于 v2，说明此前不能把提升归因于补标（第 7 节）。但用户新录的四段灶台视频揭示对照版在无火灶头上持续误报、在两段锅下蓝焰上整段漏检；v1 无火误报较少，却同样漏锅下蓝焰（第 8 节）。进一步训练后，v3 改正同一视频上的已知错误，跨场景照片仍暴露蓝焰定位不足；v4 补进七张不同场景照片后未改善保留蓝焰诊断图且原图片测试退步（第 9 节）。新权重均未编译 KModel、未在 K230 实测；板端仍是此前的 MVP 模型。

下文把**验证集指标、原测试集指标、逐图外部诊断**分开，分母和阈值分别写明。四类 v3 与单类 v1/v2/对照版的总体 mAP 不能直接横比；三个单类版本使用同一套 val/test，可以同口径比较。原 286 张 test 和外部照片已在多轮模型选择中反复查看，虽然从未用于梯度训练，**今后都只能作为开发对照/故障诊断，不能再充当一次性盲测的最终验收组**。

## 2. 经过：从 v3 到单类 v1、补标 v2

1. **检查 v3 的数据和低分。** DeepSeek 构建的 `D:\SRTP_Datasets\kitchen_vision_v3_20260926` 将部分厨房火焰映射为 `fire=0`，按拍摄段切分，并把已确认无火的 Kitchen Safety 图纳入。四类 v3 训练共 55 轮，按 `results.csv` 最佳第 35 轮，P=0.307、R=0.369、mAP50=0.330、mAP50-95=0.159。这是当时用户指出“数值很差”的那一版；该指标属于四类验证集，不等于板端识别成功率。构建说明在 `model_work/docs/BUILD_REPORT_kitchen_vision_v3.md`。
2. **由 Codex 训练单类 v1。** 从 v3 最佳权重继续训练 YOLO11s，把任务收缩为只检 `fire=0`。新数据 `D:\SRTP_Datasets\kitchen_fire_binary_codex_20260927` 有 2382 train、243 val、286 test；只把独立源图组的一张放入 train，保留厨房无火负例。v1 在第 32 轮早停，最佳第 17 轮。脚本、数据构建和第一次逐图结果见 `model_work/docs/FIRE_BINARY_CODEX_TRAIN_20260927.md`。
3. **复查用户早期手标的 64 张厨房图。** 目录 `C:\Users\ASUS\Documents\Codex\kitchen-fire-detection\datasets\kitchen-images` 的 64 张图曾 **64/64 被旧训练工程复制进 train**，而 v1 又从旧模型链条的权重起步，所以只能用于回归排错，不能当独立测试。逐图看发现 25 张原标签为空但画面实际有火，其中正常做饭小火/蓝焰 9 张。对这 25 张补了 36 个单类 fire 框，保存在 `model_work/data/manual_kitchen_labels_v2_20260927/labels/`，原图和原标签未改。框图已目视核对覆盖范围，但边界是初稿。旧 39 张原有标签未逐框审定，部分有图库水印或网页截图，未直接并入新训练。
4. **另找真实照片作为外部诊断。** 从逐个核对过作品页、作者和许可的 Wikimedia Commons 下载 13 张照片：10 张可见火焰（4 张蓝焰、其余含炒锅火）、3 张无可见火（含只有蒸汽的炒锅）。逐图写入 YOLO 框/空标签。与现有及前代数据和旧 64 张图共 9456 张做 SHA256 与 dHash 比对，13 张精确哈希重合为 0，最近 dHash 距离 12–19。在下述 v1/v2/无补标对照阶段，这 13 张**均留在训练之外**；后续 v4 训练选入其中 7 张，故原先的外部诊断身份只适用于 v4 之前的版本。哈希检查不能排除所有裁剪或同场景关系，且照片偏向摄影特写与餐厅，规模不足以验收总体性能。作品页、许可、清单、原图哈希和逐图判读见 `model_work/data/manual_kitchen_labels_v2_20260927/REAL_PHOTO_CANDIDATES.md`。
5. **训练补标 v2。** 以 v1 的 `best.pt` 为起点，在同一 train 上只增加上述 25 张补标图，得 2407 train；val 仍为 243，test 仍为 286，Commons 13 张仍隔离。YOLO11s、640、batch 4、AdamW，初始学习率由 v1 的 0.001 降至 0.0003，最多 40 轮、patience 10。第 21 轮早停，最佳第 11 轮。执行脚本 `model_work/src/train_fire_binary_manual25_v2.py`，组合数据哈希清单 `model_work/data/fire_binary_manual25_v2_20260927/build_manifest.json`，详细运行记录 `model_work/docs/FIRE_BINARY_MANUAL25_V2_RUN_20260927.md`。
6. **冻结权重后评测。** 对 v1、v2 在原 286 张 test、其中的 146 张厨房子集，以及同一批 13 张外部照片分别比较。该阶段的评测没有参与训练或选第 11 轮权重；后续数据角色变化见第 9 节。脚本和逐图 CSV 在第 5 节列明。

## 3. 训练配置与权重身份

| 项 | 单类 v1 | 补标 v2 |
|---|---|---|
| 起点 | 四类 v3 `best.pt` | 单类 v1 `best.pt` |
| train / val / test | 2382 / 243 / 286 | 2407 / 243 / 286 |
| 新增数据 | 无 | 25 张旧厨房图的补标，仅 train |
| 最大轮数 / 早停 | 60 / patience 15 | 40 / patience 10 |
| 实际轮数 / 最佳轮 | 32 / 17 | 21 / 11 |
| 初始学习率 | 0.001 | 0.0003 |
| 权重 SHA256 | `5f0fa05ebb92eb73cb39971159751e499df3f91d075c70ca818877c64ba05148` | `8cc003cb050065c9a47988959a236e27ae98d2d3d7e44468e4fcf01d69ec4d63` |

v1 权重：`C:\Users\ASUS\Documents\ChatGPT\SRTP\model_work\runs\fire-binary-codex-v1\weights\best.pt`。v2 权重：`C:\Users\ASUS\Documents\ChatGPT\SRTP\model_work\runs\fire-binary-codex-v2-manual25\weights\best.pt`。两次训练均在本机 RTX 5060 Laptop GPU 上运行；运行时 `args.yaml` 和逐轮 `results.csv` 留在各自 run 目录。原始数据、旧权重、板端文件均未覆盖。

## 4. 指标：同一口径与不同口径

### 4.1 最佳轮验证集（v1/v2 同一 243 张）

| 模型 | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| v1（epoch 17） | 0.632 | 0.792 | 0.570 | 0.189 |
| v2（epoch 11） | 0.541 | 0.729 | 0.515 | 0.174 |

v2 四项均低于 v1。第 21 轮收尾数字不是用于选择的最佳权重数字，不能混用。

### 4.2 完整原测试集（同一 286 张、77 个 fire 框，另有 215 张背景图）

| 模型 | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| v1 | 0.715 | 0.649 | 0.639 | 0.182 |
| v2 | 0.783 | 0.818 | 0.806 | 0.194 |

v2 在这套 test 上明显提高，但 mAP50-95 仅增加约 0.012。部分 Kitchen Safety 图来自同一厨房的不同日期/拍摄事件，因此此处更像**同来源测试**，不代表换厨房或板端机位后的准确率。v2 原始汇总：`model_work/out/fire_binary_manual25_v2_fulltest_20260927/metrics_summary.json`；v1 数字和运行说明见 `model_work/docs/FIRE_BINARY_CODEX_TRAIN_20260927.md`。

### 4.3 146 张厨房目标子集逐图（71 有火、75 确认无火）

“检出”指图上至少出现一个 fire 框；“定位”指至少一个预测框与真值框 IoU≥0.5。二者不能混称。

| conf | 模型 | 有火图检出 | 有火图定位 | 无火图误报 |
|---|---|---:|---:|---:|
| 0.25 | v1 | 70/71 | 54/71 | 0/75 |
| 0.25 | v2 | 71/71 | 60/71 | 0/75 |
| 0.50 | v1 | 62/71 | 46/71 | 0/75 |
| 0.50 | v2 | 61/71 | 53/71 | 0/75 |

这组数字展示同一批厨房图上的定位改善，也说明阈值会改变检出结论；没有覆盖不同厨房和大量反光/蒸汽难负例。

### 4.4 外部 13 张真实照片逐图（10 有火、3 无火）

| conf | 模型 | 有火图检出 | 有火图定位 IoU≥0.5 | 无火图误报 | 4 张蓝焰漏检 |
|---|---|---:|---:|---:|---:|
| 0.25 | v1 | 7/10 | 5/10 | 1/3 | 3/4 |
| 0.25 | v2 | 6/10 | 4/10 | 1/3 | 4/4 |
| 0.50 | v1 | 5/10 | 4/10 | 1/3 | 4/4 |
| 0.50 | v2 | 5/10 | 4/10 | 1/3 | 4/4 |

v2 的 `flaming_wok_kellyb` 虽有火框但定位明显退步；两版都把 `outdoor_wok` 的蒸汽图误报为火。这 13 张既小又是为了查弱点而挑选，**不可写成总体准确率**。它们直接暴露：单靠补旧图的 25 个标签未解决小蓝焰与蒸汽问题。

## 5. 可复核文件

| 用途 | 路径 |
|---|---|
| v3 训练曲线 | `C:\Users\ASUS\Documents\Codex\kitchen-fire-detection\runs\kitchen-vision-v3-640\results.csv` |
| v1 训练曲线 | `model_work/runs/fire-binary-codex-v1/results.csv` |
| v2 训练曲线 | `model_work/runs/fire-binary-codex-v2-manual25/results.csv` |
| 本地 64 张原图与补标清单 | `model_work/data/manual_kitchen_labels_v2_20260927/manifest.csv` |
| Commons 13 张来源、许可与哈希 | `model_work/data/manual_kitchen_labels_v2_20260927/commons_candidates/manifest.csv` |
| Commons 交集检查 | `model_work/data/manual_kitchen_labels_v2_20260927/commons_candidates/overlap_check.csv` |
| 64 张旧图历史回归 | `model_work/out/manual_kitchen_regression_20260927/per_image.csv` |
| v1 厨房 146 张逐图 | `model_work/out/fire_binary_codex_eval_20260927/per_image.csv` |
| v2 厨房 146 张逐图 | `model_work/out/fire_binary_manual25_v2_eval_20260927/per_image.csv` |
| v1/v2 外部 13 张逐图 | `model_work/data/manual_kitchen_labels_v2_20260927/commons_candidates/model_eval_v2_13_20260927/per_image.csv` |
| v2 完整 286 张测试 | `model_work/out/fire_binary_manual25_v2_fulltest_20260927/metrics_summary.json` |

## 6. 当时的决定和未完成边界（已由第 9 节更新）

- 本节原先依据静态图将不加补标的对照版列为下一轮候选；**后续实拍视频已修正这个决定**：它在当前机位无火画面持续误报，不能作为首选。三版权重均保留，v1 暂作下一轮训练基线。新的 `.pt` 尚未转换 KModel、未接板、未替换原视觉程序。
- 这两版都是**单类 fire 检测器**，不能直接宣称能判“正常做饭/爆炒/异常火灾”，也没有建立烟雾视觉分类结果。状态判断还需要时间序列、厨房区域和 SHT31/BMP280 等传感器数据；本报告没有测试融合算法。
- 下一次若继续优化，先按**不同厨房、不同机位/事件**补真实锅下蓝焰和蒸汽、反光无火照片并修准标签，再冻结不参与训练的新测试组。外部照片中的 CC 许可应随图片逐张保留；第 9 节已记录哪些原诊断图后来并入 v4 训练。

## 7. 补标 v2 的同参数对照（2026-09-27 追加）

补标 v2 与 v1 同时改变了数据和超参数，原来无法分辨 25 张补标图的净作用。现在从同一个 v1 `best.pt` 起步，用补标 v2 的全部训练参数及随机种子，在原 2382 张 train 上重训；唯一有意改变的是**不加入那 25 张图**。入口 `model_work/src/train_fire_binary_no_manual_control.py`，输入哈希和配置在 `model_work/runs/fire-binary-v2-no-manual-control/control_provenance.json`。对照版第 11 轮早停，最佳为第 1 轮，权重 SHA-256 `7be7c04717d11a0b86ad1aca1de8116e90ad508bb0a567c324dda83f0879c99f`。

| 同参数比较 | 无补标对照版 | 补标 v2 |
|---|---:|---:|
| train 张数 | 2382 | 2407 |
| 最佳轮验证集 P / R | 0.601 / 0.739 | 0.541 / 0.729 |
| 最佳轮验证集 mAP50 / mAP50-95 | 0.559 / 0.216 | 0.515 / 0.174 |
| 原 286 张 test mAP50 / mAP50-95 | 0.926 / 0.306 | 0.806 / 0.194 |
| 厨房 71 张有火图，conf 0.25 检出 / IoU≥0.5 定位 | 69 / 67 | 71 / 60 |
| 厨房 75 张无火图，conf 0.25 误报 | 0 | 0 |
| 外部 10 张有火照片，conf 0.25 检出 / 定位 | 8 / 6 | 6 / 4 |
| 其中 4 张蓝焰，conf 0.25 检出 / 定位 | 2 / 1 | 0 / 0 |
| 外部 3 张无火照片，conf 0.25 误报 | 1 | 1 |

对照版在这些口径上总体好于补标 v2，**不能说 25 张补标图改善了模型**。三版权重的 286 张测试指标又用同一个 `eval_fire_fulltest.py` 和同一运行环境复算，v1/v2 数字与此前记录一致。两次训练早停和最佳轮不同，且只做了一个随机种子；此实验比较的是“同一训练方案加入这 25 张图”的最终结果，并非每一步梯度的严格消融或统计显著性结论。原测试集同源偏差仍在，特别是 0.926 的 mAP50 不应推广成新厨房性能。

另有 5 张来源和许可已核对的蓝焰照片被刻意选作额外诊断，均未入训练、无逐框标签。conf 0.25 时 v1 与补标 v2 都是 0/5 出现 fire 框，对照版为 1/5；这个小样本只说明蓝焰弱点仍在。详情见 `model_work/docs/BLUE_FLAME_PROBE_20260927.md`。

对照版原始证据：`model_work/runs/fire-binary-v2-no-manual-control/results.csv`、`model_work/out/fire_binary_control_fulltest_20260927/metrics_summary.json`、`model_work/out/fire_binary_control_kitchen_20260927/summary.json` 和逐图 CSV、`model_work/out/fire_binary_control_external13_20260927/summary.json` 和逐图 CSV。三版同脚本复算的全量测试汇总在 `model_work/out/fire_binary_v1_fulltest_recheck_20260927/`、`model_work/out/fire_binary_v2_fulltest_recheck_20260927/` 与对照版目录。全量测试脚本 `model_work/src/eval_fire_fulltest.py`；厨房和外部诊断脚本仍用第 5 节列出的同一口径。

## 8. 实拍四段灶台视频：修正静态图结论（2026-09-27 追加）

用户新录的 `viedos/0.mp4` 至 `3.mp4` 分别是无火灶头、无遮挡蓝焰、两个锅下蓝焰视角，各约 31–33 秒。对照版在 5 fps 抽样、conf 0.25 下，**无火段 156/156 帧误报**，无遮挡蓝焰 156/157 帧有框，两个锅下段分别 0/160、0/166 帧有框。无火段预测分数中位数还高于无遮挡蓝焰段，统一调高阈值无法在这组视频中同时解决误报和漏检。

同帧 1 fps 对照：v1 无火 1/32 帧误报、无遮挡蓝焰 32/32 有框、两个锅下段 0/32 和 0/34；补标 v2 对应 5/32、32/32、0/32、1/34；无补标对照版 32/32、31/32、0/32、0/34。于是此前“对照版为最佳电脑端候选”的静态判断被推翻。所有帧来自少量连续片段，不能据帧数算总体准确率；四段又都用于开发诊断，不能作为最终盲测。完整来源、画框回放、逐帧 CSV 与判断见 `model_work/docs/USER_FOUR_VIDEO_PROBE_20260927.md`。

## 9. 实拍视频补标训练后的当前决策（2026-09-27 追加）

四段视频各抽六帧并核对火焰框/无火空标签，与旧图子集合成 v3 训练集；v3 在**已参与训练**的四段视频上，关火 0/32 帧有框，三段有火分别 32/32、32/32、34/34 帧与参考火焰框 IoU≥0.5。代表帧人工复查确认框落在蓝焰上。原 286 图 test mAP50=0.973、mAP50-95=0.279，但它已是反复用于模型选择的同源开发集；另五张未入训练的蓝焰照片只 1/5 有框，另外 13 张中的 4 张蓝焰为 2/4 有框、0/4 正确定位。

再把那 13 张中的四张蓝焰、三张无火/蒸汽图用于 v4 训练；v4 对上述未参与训练的五张蓝焰仍只 1/5 有框，原 286 图 mAP50 降到 0.766。因此**弃用 v4，保留 v3 作为新视频一次性检验的电脑端开发候选**。此举不代表模型跨厨房达标。权重 SHA-256、数据来源、重复加权方法、训练参数、逐帧定位对照及新视频验收口径见 `model_work/docs/USER_VIDEO_ADAPTATION_RUN_20260927.md` 和 `model_work/docs/FRESH_VIDEO_VISUAL_ACCEPTANCE_PROTOCOL_20260927.md`。
