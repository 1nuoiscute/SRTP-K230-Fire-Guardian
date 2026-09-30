# 火焰单类补标微调 v2（2026-09-27）

状态：训练和电脑端对照评测完成；**不替换板端模型**。

- 训练脚本：`C:\Users\ASUS\Documents\ChatGPT\SRTP\model_work\src\train_fire_binary_manual25_v2.py`
- 原数据：`D:\SRTP_Datasets\kitchen_fire_binary_codex_20260927`，2382 train / 243 val / 286 test，原文件未改。
- 新增数据：`C:\Users\ASUS\Documents\ChatGPT\SRTP\model_work\data\manual_kitchen_labels_v2_20260927` 中原空标签经目视补框的 25 张、36 个 fire 框，只加入 train。复核框仍属初稿；图像此前已被旧模型链条见过，故不能作为独立验收图。
- 组合数据及逐项哈希清单：`C:\Users\ASUS\Documents\ChatGPT\SRTP\model_work\data\fire_binary_manual25_v2_20260927\build_manifest.json`。训练 2407 张。保留原 val/test；Commons 13 张外部诊断图完全排除。
- 起点：`C:\Users\ASUS\Documents\ChatGPT\SRTP\model_work\runs\fire-binary-codex-v1\weights\best.pt`，SHA256 `5f0fa05ebb92eb73cb39971159751e499df3f91d075c70ca818877c64ba05148`。
- 输出：`C:\Users\ASUS\Documents\ChatGPT\SRTP\model_work\runs\fire-binary-codex-v2-manual25`。
- 配置：YOLO11s，640，batch 4，AdamW，初始学习率 0.0003，最多 40 轮，patience 10，seed 20260927；其余增强详见训练脚本和运行时 `args.yaml`。

训练完成后先用原 286 张 test 做与 v1 同口径比较，再用冻结的 13 张 Commons 外部诊断图逐图比较，特别报告 4 张蓝焰的漏检和 3 张无火图的误报。仅当独立厨房的定位与误报都改善，才考虑 KModel 编译和板端替换。本次训练本身不部署。

## 完成结果

训练于第 **21** 轮早停，最佳权重来自第 **11** 轮：`C:\Users\ASUS\Documents\ChatGPT\SRTP\model_work\runs\fire-binary-codex-v2-manual25\weights\best.pt`，SHA256 `8cc003cb050065c9a47988959a236e27ae98d2d3d7e44468e4fcf01d69ec4d63`。最佳轮验证集 P=0.541、R=0.729、mAP50=0.515、mAP50-95=0.174；上一版 v1 的最佳验证指标为 P=0.632、R=0.792、mAP50=0.570、mAP50-95=0.189，v2 验证集退步。

完整 286 张原测试集：v2 P=0.783、R=0.818、mAP50=0.806、mAP50-95=0.194；v1 对应 P=0.715、R=0.649、mAP50=0.639、mAP50-95=0.182。完整测试集有 77 个火框、215 张背景图；Kitchen Safety 的不同日期画面仍可能来自同一个厨房，不能将这项提高等同于独立厨房泛化提高。v2 测试输出：`model_work/out/fire_binary_manual25_v2_fulltest_20260927/metrics_summary.json`。

固定 146 张厨房子集（71 有火、75 无火）逐图比较：

| 阈值 | 模型 | 有火图出现火框 | 有火图 IoU≥0.5 | 无火误报 |
|---|---|---:|---:|---:|
| 0.25 | v1 | 70/71 | 54/71 | 0/75 |
| 0.25 | v2 | 71/71 | 60/71 | 0/75 |
| 0.50 | v1 | 62/71 | 46/71 | 0/75 |
| 0.50 | v2 | 61/71 | 53/71 | 0/75 |

v2 逐图结果：`model_work/out/fire_binary_manual25_v2_eval_20260927/per_image.csv`；v1 同口径报告：`model_work/out/fire_binary_codex_eval_20260927/per_image.csv`。

**真正关键的外部 13 张**（10 有火、3 无火；与既有数据无精确 SHA256 重合）却退步：0.25 阈值下 v1 有火检出 7/10、IoU≥0.5 5/10、无火误报 1/3；v2 分别为 6/10、4/10、1/3。v2 对 4 张蓝焰照片全部漏检，v1 也漏 3 张。两者都把一张只有蒸汽的炒锅图误报为火。外部逐图表：`model_work/data/manual_kitchen_labels_v2_20260927/commons_candidates/model_eval_v2_13_20260927/per_image.csv`。13 张样本太少且不代表真实厨房总体分布，只能指出具体失败，不是统计意义上的验收分数。

**决定**：保留 v1 为电脑端现有候选；v2 只保留为实验权重，不编译、不部署。下一次训练前要补不同厨房和机位下的锅遮挡蓝焰、蒸汽/反光无火难例，逐图检查标注与来源，按厨房/事件隔离训练与测试。不能再把现有同厨房切分的高测试 mAP 当成实际可靠性。
