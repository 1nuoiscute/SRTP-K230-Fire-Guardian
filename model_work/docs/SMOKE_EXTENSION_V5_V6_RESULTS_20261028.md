# SMOKE_EXTENSION_V5_V6_RESULTS_20261028.md

## 实验定位与数据
- 定位：可见烟雾开发期实验；无独立测试、无 ONNX 导出、无上板（JSON 均为 false）。
- 数据：总计 52 图 = 原 36 + 新 16；拆分 train 43 / val 9，旧 val 9 图保持不变（original_val_unchanged=true）。新 16 中正例 4、无可见烟 12；held 8。
- 训练条件：v5 与 v6 同数据、同起点权重（start_sha256 e5c44663…）、同为 40 epochs、实际 optimizer steps 均为 432（batches 440），构成 v5/v6 的受控对照；环境 Python 3.13.13 / torch 2.12.0+cu132 / ultralytics 8.4.70 / RTX 5060 Laptop GPU。
- 机制声明：v5 为固定特征，v6 为后段适配（更新预算与 v5 相同，432 步）；但 v6 的 BN affine / 冻结机制相对 v5 有变化，v5-v6 差异不能只归于“更新了哪些参数”。
- v4 对比 v5 声明：两者不仅数据量变化（原 36 → 43train/9val），更新预算也不同，任何 v4→v5 差异不可单因素归因于数据量。

## 对照表（数据源：docs/smoke_extension_v5_v6_results_20261028.json）

| 模型 | 旧9图val TP/FP/FN | 新16训练反馈 TP/FP/FN | 未知19图有响应图数 |
|---|---|---|---|
| smoke_start（起点） | 0/0/3 | 0/0/4 | 0 |
| v4_best | 2/0/1 | 0/1/4 | 0 |
| v4_last | 2/0/1 | 1/1/3 | 0 |
| v5_best | 2/1/1 | 1/0/3 | 0 |
| v5_last | 2/1/1 | 1/0/3 | 0 |
| v6_best | 3/1/0 | 4/0/0 | 5 |
| v6_last | 2/0/1 | 4/0/0 | 1 |

## 读表要点（区分证据与推断）
- 证据：v6 在同数据/同起点/同 40 轮/同 432 步条件下，训练反馈由 v5 的 1/0/3 提升到 4/0/0；旧 val 上 v6_best 达 3/1/0（召回最高），v6_last 为 2/0/1 且无 FP。
- 推断（需声明机制变化）：该提升可能来自后段适配 + BN affine/冻结机制变化，而非单一因素。
- v6_best 在未知 19 图上产生 5 图响应、v6_last 产生 1 图响应；未知烹饪羽流无负例真值（JSON role 明示），5 或 1 不能称为误报率，只能描述为“有响应图数/框数”。
- 决策（JSON decision）：旧双头仍保留 v4；v5 不晋升；v6_last 仅作为新增独立烟雾网络的开发候选；v6_best 仅作更高召回/更多响应的对照。shared_fire_backbone_compatible=false，即不能把 v6 的改动特征塞回旧双头结构。

## 下一步（≤2 项）
1. 补充带采集说明的燃烧烟雾、蒸汽、油烟对照事件；仅凭现有图片不能建立燃烧原因真值。先记录羽流外观和框几何，不将未知组改成负例。
2. 为 v6_last 增设独立留出集（新来源/事件在开发前冻结，不将已曝光图重新命名为独立测试），验证 4/0/0 的训练反馈是否外推，再决定是否推进独立烟雾网络。

## 复现入口
- 结果 JSON：`docs/smoke_extension_v5_v6_results_20261028.json`（comparison_sha256 106df299…）
- 权重：v5 run `model_work/runs/smoke-extension-v5-20261028/weights/`（best 0b061873… / last 39afe227…）；v6 run `model_work/runs/smoke-partial-features-v6-20261028/weights/`（best c8a0c349… / last d4b69cf8…）；起点 `model_work/out/independent_smoke_start_v4_20261002/start.pt`
- 逐图结果与拼图页：各模型 per_image_files（val_dev.json / extension_train.json / unknown_cooking_plume.json）及 contact_* 页，文件名见 JSON。



## 实际视觉复核与边界

主控实际查看 v5 last 的3页val，v6 best的2页含烟val、v6 best/last各1页新增正例，以及覆盖全部6个未知组响应的3页。v6 best补出远处深色烟柱，但同图出现覆盖相邻天空的第二个框；v6 last的4个新增正例均对应可见烟雾区域。未知组best5个框主要覆盖明亮背景/羽流区域，另有一个偏右的非羽流背景框；last的1个框同时包入羽流和较宽墙面。这里只描述几何与外观，不能判断蒸汽、油烟或燃烧烟雾。

GLM原始v6代码误将后段卷积也列为固定状态，已在训练前修正为前缀、特征BN缓冲及DFL。Copilot静态复核提出DFL可能解冻的猜测，但本机初始化及40轮实际检查均通过，不作为已发生缺陷。模型生成、静态意见与实际执行证据分别记录。
