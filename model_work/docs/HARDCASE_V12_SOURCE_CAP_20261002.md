# v12 灶火来源采样消融｜2026-10-02

用户授权持续推进电脑端模型。无开发板连接，本实验只在电脑训练和评测，不替换 v3、不修改原数据或旧实验。

## 运行前固定的假设和范围

既有训练抽查显示，GC 灶火来源包含蓝焰，却有高度重复的商用厨房背景，部分框包括较大锅体；本次重新查看抽查第一张联系表也确认这些现象。抽查不是全量场景/颜色统计。v11 中该来源占 1045/2432 个文件、1045/2810 个采样条目；这可能限制其他来源的学习，但原因尚未得到实验验证。

v12 只改变该来源的采样：用 seed=20261002，将 `kitchen_stove_fire` 1045 张原训练图确定性抽到 128 张。排序键是 seed、provenance、文件名、图像 SHA 的 SHA-256，取最小 128 个；不按预测表现挑图，不再追加不同上限追逐五张蓝焰分数。其他全部保留样本、标签、采样权重不变。未修正锅体框，也未把剩余图改成负例。父版本 v11 审核链和原始 train/val/test 均保留。

这叫来源采样消融，不叫厨房场景独立划分；文件名不是独立厨房身份。v3 的原始训练暴露也不会因本次少用图片而消失。旧测试、Commons 原作、私人视频均为已曝光开发诊断。

| 输入 | 身份或计数 |
|---|---|
| 起点 v3 best.pt | `48f284f9094fe334e02c66647f95fed8bfb3396b09f4488753508f514ed38980` |
| v11 父清单 | `858e0140b56a26f65476249896adc69f3de296856e99c21f8d3b58d0916f24e3` |
| v12 清单 | `d7c0a4853495a2dcc68935c55269ec41b57ba086eb8a4c69e21a1959b3fb6cfd` |
| v12 train.txt | `03bf40c5cee4f75f1583154d25dd437c90b99f62ed9b6ec99c747a6b426ae2b0` |
| 图像与标签绑定 | 1515 文件、1893 条目、1515 标签；父版本 2432 文件同时重新核验 |
| 派生图 | 原有 55 张；无新增原作、场景或标签 |

训练采用 v11 相同配置：从 v3 重启微调、最多 12 epoch、patience=8、640、batch=4、AdamW、lr0=2e-5、warmup_bias_lr=2e-5、seed=20260930；不冻结主干。数据量减小导致每 epoch 优化步数减少，这个差异属于本次采样方案，不能称作优化步数相等的因果隔离。实际 args 与 v11 将在启动后核对。

## 运行与判断

```powershell
python -B model_work/src/build_source_capped_dataset.py --parent model_work/data/hardcase_v11_negative_curation_20260930 --out model_work/data/hardcase_v12_stove_cap_20261002 --source kitchen_stove_fire --cap 128 --seed 20261002 --rationale "Bounded development ablation: reduce dominating repeated-commercial-stove source from 1045 to 128 unique original train images; no label repairs, no independent-scene claim; all other sources and weights unchanged."
python -B model_work/src/train_hardcase_v5.py --data model_work/data/hardcase_v12_stove_cap_20261002/data.yaml --start model_work/runs/fire-binary-user-video-v3/weights/best.pt --expected-sha256 48f284f9094fe334e02c66647f95fed8bfb3396b09f4488753508f514ed38980 --out model_work/runs/fire-binary-hardcase-v12-stove-cap-20261002 --epochs 12 --lr0 0.00002 --warmup-bias-lr 0.00002
```

完成后对 best/last 与原 v3 比较：55 个曝光派生图、五图蓝焰主要炉位、286 图旧 mAP、来源固定阈值 TP/FP/FN、Commons 曝光分组、9 段开发视频。v11 原结果作参考；v11 best/last 也在同一 55 图清单上重新检查。

沿用前一阶段冻结筛选门槛：相对 v3，旧图 mAP50 下降不超过 .01、蓝焰定位至少增加 1/5、原无可见火焰 100 图 FP 框增加不超过 1。满足这些仅允许进一步复核，不能自动晋升或等同计划书 F1 验收。曝光正例、食物/烟雾误检及逐来源退化继续报告；贴纸按用户已缩小演示范围作为附带诊断。

本文件在训练前建立；运行失败、早停、未改善都追加记录，不覆盖旧实验，不自动重启。当前状态：数据构建和完整身份核验通过，训练尚未启动。

## 启动记录

冻结方案提交 `7f1ea9a` 后，于本机 2026-10-02 13:37（Asia/Shanghai）启动训练，PID=137752；以 psutil 核对 Python 路径、完整命令与创建时间。实际 args.yaml 和 v11 只有 data/name/save_dir 三项差异，训练配置未漂移。完成后的顺序比较进程 PID=36208，绑定该训练进程，不会重启训练。

评测包含固定阈值旧测试来源计数和 v11 best/last 曝光派生图重测；每项评测限 1800 秒，超时或失败保存日志。训练已有第 1 轮完成结果；这还不能判断改进。运行目录：`model_work/runs/fire-binary-hardcase-v12-stove-cap-20261002`，比较目录：`model_work/out/hardcase_v12_stove_cap_20261002/comparison`。脚本、preflight、args、CSV及独立stdout/stderr日志留本地。

数据审计新增父子版本端到端测试：排除的父样本标签变动也必须拒绝，循环父引用必须拒绝；选样测试保证固定集合、继承权重和标签不变。基础CI安装与本机一致的 PyYAML 6.0.3 来实际执行这些数据配置测试。

## 完整结果与决定

9 轮完成，patience=8 正常早停，best 在 epoch 1。训练日志报告 .229 小时；两份完成权重身份核对通过：

- best.pt：`31b376c81f343d9720ae99acf32db99aeb01b1744d4af4a4bd91246ba5650d55`
- last.pt：`d6a21ced8b474bbe362c4e694edb725d53278c03e15152466c1696fce8f86f1a`

自动对照全部完成，包含三份模型的 6 组评测、v11 两份同清单曝光帧参考、三份模型固定阈值来源定位计数。两个实际 PID 均已退出，experiment_meta.json 和 comparison_complete.json 已保存。公开结果见 [身份绑定汇总](../../docs/source_cap_v12_results_20261002.json)。

| 模型 | 五图蓝焰定位 | 旧 286 图 mAP50 | 原无火 100 图 FP 框 | 队友曝光正例 TP/FP/FN | 新增蓝焰原作训练图 TP/FP/FN |
|---|---:|---:|---:|---:|---:|
| v3（重测） | 0/5 | .9732386 | 1 | 3/5/7 | 1/3/12 |
| v12 best | 0/5 | .9527251 | 3 | 4/5/6 | 1/3/12 |
| v12 last | 3/5 | .8884888 | 5 | 9/5/1 | 6/1/7 |

best 的旧图 mAP50 下降 .0205135，蓝焰未增益，无火 FP 多 2；last 的 mAP50 下降 .0847499，虽蓝焰增加 3/5，无火 FP 多 4。**两份均不通过预设筛选，保留 v3，不晋升。** 不补试其他来源上限、不放宽门槛；此次没有证明减小重复背景采样就能解决跨厨房问题。

固定阈值来源进一步揭示：ks_flame 的 v3/best/last TP 分别 57/56/57，FP 1/1/2；14 张旧灶火三者均 18 TP/5 FP/2 FN。75 张已确认无可见火焰来源与 40 张锅具来源三者都无 FP。来源结果是曝光开发诊断，不能转换为独立测试 F1 或100小时误报。

v11 best/last 在相同 55 图上的参考重测已完成：队友曝光正例为 5/5/5 和 10/2/0，原作蓝焰训练图为 2/3/11 和 9/1/4；v12 last 相比 v11 last 在这两组也更差。旧 24 帧 v12 保持18 TP/0 FP/0 FN，但不足以抵消其他退化。1 个食物裁剪均0 FP；贴纸裁剪 FP由v3的3到v12的2，仅作用户演示范围外附带诊断。

三者各评测 9 段开发视频、307 个采样，旧4段预测数量一致，队友5段数量有变化；没有补充逐帧/事件真值，不用“有框帧更多”报告准确率提高。Commons 的两份训练来源与十一份未用于此轮训练来源仍分开统计，未训练来源也是曝光诊断。

最终核对原 v3 与 v11 清单 SHA 均未改变。候选、日志、审核清单、预览和私人媒体留本地，公开仅代码和匿名汇总。该采样消融本阶段已结束；持续目标下一步转向统一可见火焰框口径、明确 Smoke/蒸汽语义和有来源的新数据，详见 [烟雾审核](SMOKE_SOURCE_SEMANTICS_20261002.md) 与 [候选源记录](../../docs/SMOKE_DATA_NEXT_SOURCES_20261002.md)。新版计划书整体要求仍未完成。

公开汇总命令（已有输出拒绝覆盖）：

```powershell
python -B model_work/src/summarize_source_cap_experiment.py --data model_work/data/hardcase_v12_stove_cap_20261002 --run model_work/runs/fire-binary-hardcase-v12-stove-cap-20261002 --comparison model_work/out/hardcase_v12_stove_cap_20261002/comparison --expected-data-sha256 d7c0a4853495a2dcc68935c55269ec41b57ba086eb8a4c69e21a1959b3fb6cfd --out docs/source_cap_v12_results_20261002.json
```
