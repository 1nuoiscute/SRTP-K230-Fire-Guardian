# 冻结主干对照 v8｜2026-09-30

v7 last 出现旧蓝焰诊断 1/5 的局部改善，但旧 286 图 mAP50 降至 0.90550，食物/贴纸误检仍在，拒绝替换。本轮检验限制特征主干更新能否减少旧能力退化，同时保留难例适配能力；实验未完成前不预判效果。

## 唯一改变的训练因素

仍从 v3（SHA-256 48f284f9094fe334e02c66647f95fed8bfb3396b09f4488753508f514ed38980）开始，用完整相同的 v7 数据：2428 个文件、2678 个采样条目，manifest SHA-256 0a40ce9345652e322c8af0cd2d8a5abb1831e94432065c4b546828d64ee47a3b。第二批批准照片和新负裁剪不在此轮加入。

seed=20260930、640、batch=4、AdamW、lr0/warmup_bias_lr=0.00002、最多 12 轮、patience=8 及所有增强不变。实际 args.yaml 对照仅 freeze、name、save_dir 三项不同。早停可能改变实际轮数，必须在结束后如实报告。

从真实 YOLO11s 权重 YAML 读取 backbone 长度为 11，对应模块 0–10（包含 SPPF 和 C2PSA），不是套用其他 YOLO 架构的层数。设置 freeze=11；主干参数 5441984 个元素冻结，3986179 个元素可训练。当前 Ultralytics 8.4.70 的 trainer._model_train() 会将冻结模块中的 BatchNorm 设为 eval；框架文件身份与参数见 [实验身份](../../docs/hardcase_v8_freeze_identity.json)。

## 运行时验证

新增 backbone_freeze_audit.py 在启动时核对冻结范围及 v3 主干身份，每轮首批检查冻结 BatchNorm 未处于训练模式，轮末核对主干全部 240 个状态张量。参数、BN 均值/方差/计数发生任何变化即终止，不将失败当作有效冻结实验。

v3 主干张量摘要为 7180c9b7f55fdcad0e511e6328c1322de0e26c81c109802e4af2d28d4df0fa31；这是按张量名称、类型、形状、实际字节定义的运行时摘要，不等于完整 checkpoint 文件 SHA-256。EMA 计算与 FP16 保存可能带来舍入差异，因此不宣称导出的 checkpoint 主干逐字节等于 v3；运行时原模型的精确不变性单独留证。

8 项 PyTorch CPU 测试验证参数/BN 变化、BN 训练模式、错误冻结范围和无可训练参数等失败会被发现。在不安装 PyTorch 的 GitHub 基础检查中明确跳过这 8 项；其余 35 项依赖无关测试继续执行。真实训练也独立逐轮核验。

## 命令与预排对照

```powershell
python -B model_work/src/train_hardcase_v5.py --data model_work/data/hardcase_v7_sources_20260930/data.yaml --start model_work/runs/fire-binary-user-video-v3/weights/best.pt --expected-sha256 48f284f9094fe334e02c66647f95fed8bfb3396b09f4488753508f514ed38980 --out model_work/runs/fire-binary-hardcase-v8-freeze-20260930 --epochs 12 --lr0 0.00002 --warmup-bias-lr 0.00002 --freeze-backbone
```

训练记录位于 model_work/out/hardcase_v8_freeze_20260930/training.log；运行目录 model_work/runs/fire-binary-hardcase-v8-freeze-20260930。评测 runner 已绑定核实的真实进程，结束后比较 v3/v8 best/last 的 46 图、五图蓝焰、旧 286 图、按原作分组 Commons 13 图和九段开发视频；并在同一 46 图补测 v7 best/last。它不自动重训、不部署、不推送。

训练和完整评测已完成，拒绝替换。来源边界沿用 v7：Commons 两个蓝焰原作已进训练，剩余未训练诊断仍非盲测；独立厨房/完整事件验收、量化和板端运行继续未完成。


## 完整结果

最多 12 轮、patience=8，实际第 9 轮早停；results.csv 最后一行累计用时 1177.41 秒。启动、9 个轮末和结束共 11 次检查全部通过，240 个主干状态张量始终与 v3 的运行时摘要相同。此检查包括冻结 BN 的 running_mean、running_var 与 num_batches_tracked。

best SHA-256：c714a9f42439b502d458d97ca933da1cf5f7f9ee211958579a8c4afb36cd74c4；last SHA-256：6de21642ea1409b747d1142cc469ab756e2eba2a774d7548f2c8d2fa3a1ebd18。

| 项目 | v3 | v8 best | v8 last |
|---|---:|---:|---:|
| 队友 10 张正例 TP / FP / FN | 3 / 5 / 7 | 3 / 5 / 7 | 5 / 5 / 5 |
| 10 张无火裁剪误报图 | 3 | 2 | 3 |
| 旧视频 24 帧 TP / FP / FN | 18 / 0 / 0 | 18 / 0 / 0 | 18 / 0 / 0 |
| 两份转入训练的蓝焰照片 TP / FP / FN | 0 / 1 / 2 | 0 / 0 / 2 | 0 / 0 / 2 |
| 五张蓝焰主要框定位 | 0/5 | 0/5 | 0/5 |
| 旧 286 图 mAP50 | 0.97324 | 0.97368 | 0.95641 |

Commons 两个训练蓝焰来源及两个本轮未训练蓝焰来源，best/last 在 conf=0.25 和 0.5 均定位 0/2。目视 v8 last 的 teammate_f0cb3eb_1608 仍把蓝贴纸框作火，同时漏掉锅下火焰。46 图 v7 参考对照为 best 正例 4/4/6、last 10/2/0，两份加入训练的照片 last 为 1/0/1；因此完整冻结明显限制了本次难例适配。

**决定：拒绝替换。** best 的旧图诊断基本保持（极小差异不作统计提升解释），但目标蓝焰仍失败；last 仍有误检、漏检及退化。本次数据、起点、日程和实际 9 轮均与 v7 一致，仅冻结机制不同；开发结果支持“减少退化但妨碍难例适配”的本轮观察，不推广为其他数据或架构的必然规律。

下一轮回到 v7 的不冻结配置和 v3 起点，只将已批准的第二批 8 张蓝焰/11 框及 1 个食物无火裁剪加入数据，其他因素不变。此轮仍是开发实验，不补齐独立事件验收。

完整结果：model_work/out/hardcase_v8_freeze_20260930/comparison_sourceaware/comparison_complete.json；公开汇总：[v8 JSON](../../docs/hardcase_v8_comparison_results.json)。
