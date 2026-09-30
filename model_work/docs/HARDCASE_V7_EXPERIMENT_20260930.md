# 蓝焰诊断源转入训练 v7｜2026-09-30

v6 完成后仍没有解决外部蓝焰定位，故不替换基线。本轮检验两份既有诊断原作首次有记录地加入训练的效果。

## 已固定的实验

起点仍为 v3，SHA-256：48f284f9094fe334e02c66647f95fed8bfb3396b09f4488753508f514ed38980。不从 v6 继续训练，避免混入额外适配阶段。

与 v6 相同：seed=20260930、imgsz=640、batch=4、AdamW、lr0=0.00002、warmup_bias_lr=0.00002、lrf=0.1、weight_decay=0.0005、计划最多 12 轮、patience=8，以及所有增强参数。唯一有意改变的训练因素是数据：在原 2654 条目上加入两份人工审核的既有诊断原作图像各 12 次，共 2678 条目/2428 个文件。实际早停轮数须按完成记录报告。

转入训练的照片版本、近似框、许可及来源见 [数据准备](BLUE_SOURCE_DATA_PREPARATION_20260930.md)。数据清单 SHA-256：0a40ce9345652e322c8af0cd2d8a5abb1831e94432065c4b546828d64ee47a3b。

训练前的新校验器核对审核清单/图、2428 个图像及标签、2678 个条目、权重，以及 YAML 的训练和验证分区。训练步长、增强或优化器未因校验器而改变。新校验和源代码身份写入 preflight。

```powershell
python -B model_work/src/train_hardcase_v5.py --data model_work/data/hardcase_v7_sources_20260930/data.yaml --start model_work/runs/fire-binary-user-video-v3/weights/best.pt --expected-sha256 48f284f9094fe334e02c66647f95fed8bfb3396b09f4488753508f514ed38980 --out model_work/runs/fire-binary-hardcase-v7-sources-20260930 --epochs 12 --lr0 0.00002 --warmup-bias-lr 0.00002
```

实际运行目录 model_work/runs/fire-binary-hardcase-v7-sources-20260930，日志和对照放在 model_work/out/hardcase_v7_sources_20260930。训练已在第 9 轮按 patience=8 正常早停，完整评测及 v6 同图参考对照均已完成。

## 预排对照

训练进程结束且 best/last 哈希验证通过后，顺序评测 v3、v7 best/last：原 44 派生样例、两份转入训练的照片、五张旧蓝焰、旧 286 图、Commons 13 图及九段开发视频。另在同一份 46 图上补测 v6 best/last，明确两张照片加入前后的差别；旧 v6 图/视频结果仍由其完整记录提供。

现有 best 仍由旧验证分区选择，并不自动代表厨房蓝焰最优。所有旧图/视频与两张新增来源均是开发材料；改善只可称开发回归或诊断改善。独立厨房/完整事件验收、每小时误报和板端量化/运行没有完成。

## 来源范围更正

两份训练增补均来自既有 Commons 诊断原作，不是全新来源。旧 13 图结果须分成 2 个训练来源与 11 个本轮未训练的开发诊断来源；剩余未训练蓝焰来源仅 2 个。见 [更正及原作对照](COMMONS_SOURCE_SCOPE_CORRECTION_20260930.md)。评测输出改为 comparison_sourceaware，训练仍为同一进程与数据哈希。

训练约 2503 秒，实际 9 轮。best SHA-256：b7c27a8b192b4ffdfd0b0d2ec8a8183b179354de9d23ed8ba32c1777498b8561；last SHA-256：7ce5b575d5c9d0d8d2581666305198ead3b4d3a1c3b1fc53d979dea45835e851。旧验证分区选择器在第 1 轮达到本次最佳，后续未改善；不从这个选择器直接推断蓝焰效果。


## 完整对照与决定

| 项目 | v3 | v7 best | v7 last |
|---|---:|---:|---:|
| 10 张队友派生正例 TP / FP / FN | 3 / 5 / 7 | 4 / 4 / 6 | 10 / 2 / 0 |
| 10 张无火裁剪误报图 | 3 | 2 | 2 |
| 原视频 24 帧 TP / FP / FN | 18 / 0 / 0 | 18 / 0 / 0 | 18 / 0 / 0 |
| 本轮转入训练的两份照片 TP / FP / FN | 0 / 1 / 2 | 0 / 0 / 2 | 1 / 0 / 1 |
| 五张旧蓝焰诊断主要框定位 | 0/5 | 0/5 | 1/5 |
| 旧 286 图 mAP50 | 0.97324 | 0.95109 | 0.90550 |
| Commons 训练来源蓝焰定位（2 原作） | 0/2 | 0/2 | 1/2 |
| Commons 本轮未训练来源蓝焰定位（2 原作） | 0/2 | 0/2 | 1/2 |

框匹配统一使用 conf=0.25、NMS IoU=0.6、匹配 IoU=0.5。Commons 两行在 conf=0.25 和 0.5 均为表中值；同一比较的来源分组一致，并非表示 v3 也训练过本轮加入的照片。五张旧蓝焰与 Commons 分组属于不同诊断清单，不能相加成一项准确率。

v7 last 在 blue_bogtar 的主炉位 IoU=0.83359，目视叠框确认定位在蓝焰环上，这是一次开发诊断改善。其余四张仍未定位。原作对照后的 Commons 剩余两份未训练蓝焰中也定位一份；它们均已被反复用于开发，不能称盲测泛化。

46 图补测 v6：v6 best/last 对两份本轮训练增补照片均 TP=0、FP=0、FN=2；v6 last 的队友正例为 10/1/0，v7 last 为 10/2/0。v7 目视结果中 teammate_82f51e3_666 错把锅内食物/蒸汽区域框作火，teammate_f0cb3eb_1608 仍框住蓝贴纸。这些与正确炉火框同时存在，不能用“有框帧”掩盖误检。

**决定：拒绝替换，保留 v3 开发基线和板端 MVP。** last 的蓝焰诊断出现局部改善，但旧图退化扩大且误检未消除；best 仍不能代表目标厨房蓝焰表现。下一轮先在完全相同的 v7 数据和 v3 起点上检验冻结主干的单因素改变，不同时加入第二批照片。它是有待检验的保持旧能力假设，不是预先批准的新部署模型。

完整本地结果：model_work/out/hardcase_v7_sources_20260930/comparison_sourceaware/comparison_complete.json；公开汇总：[v7 JSON](../../docs/hardcase_v7_comparison_results.json)。该轮训练没有重启。旧比较进程因启动时未加载新增的来源分组参数，在执行任何评测前经身份核实后终止，原因保存在旧 comparison/superseded_before_evaluation.json；替代评测目录与训练产物均另存，保留全过程。
