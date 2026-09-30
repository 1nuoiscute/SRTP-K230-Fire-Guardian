# 负例审核清理与权重对照 v11｜2026-09-30

v10 移除整个通用来源没有改善五图蓝焰定位，且锅内食物误检重新出现；[按来源核对](LEGACY_SOURCE_LOCALIZATION_20260930.md) 显示开发候选的另一个问题是烟雾/反光被当成火焰。v11 回到 v9 的数据，主要因素是负例准入与权重清理；不追加贴纸专项数据。

## 已实际审核的变化

1. 隔离三张画面有明确火焰但原标签为空的通用来源图，避免继续作为无可见火焰背景学习。
2. 隔离两张火焰框位于烟雾/灰雾、在原分辨率下无法确认火苗的图，待后续语义复核；没有直接改为空标签。
3. 从当前 train 的烟雾背景来源按固定 seed 随机抽取 12 张生成拼版，选出其中三张再打开原图复核：烟囱烟羽、暗背景蓝灰烟羽、室内薄烟。其原始空火焰标签均实际读取确认，每份权重由 1 增至 12。

三张加权图都是已有训练原件，不是 test 错误图，不增加独立场景。可见烟雾/阴燃并不表示事件安全，标签只说明此帧未见清楚火焰。来源名称 nofire_real_indoor 也不能替代目视判断，抽样中包含室外场景。

其余 795 张通用来源、55 份 v9 派生图、蓝焰照片及食品无可见火焰裁剪全部保留，不改图像或标签。五份隔离原件继续留在原路径，不删除。隔离与加权是同一轮负例数据组合，不单独归因于其中某一个操作。

## 数据身份与检查

共 2432 个文件、2810 个采样条目：v9 的 2782 条目减去 5，加上三份各多采样 11 次，共增加 33。全部 2432 份图像及标签绑定身份，保留 lineage 除三个明确权重字段外与 v9 逐字段相同，包括 Unicode 署名。

build_reviewed_source_ablation.py 新增 --changes-review，与整来源 --exclude-provenance 模式互斥。reviewed_sampling_changes.py 拒绝未批准修改、错误父数据/证据、替换图像/标签、不属于已有 train 的条目、非空标签冒作负例、非法或过大权重、重复和路径越界名称。构建器另外核验加权图确实在身份绑定的目视抽样中。

最终审核拼版、抽样清单、隔离与加权审批在 source_reviews 留本机完整快照。初始 pending 审批保留；审批前 preflight 拒绝训练，审批后实际全量核验通过。dataset_preflight.py 亦校验最终负例拼版身份。公开 [数据汇总](../../docs/hardcase_v11_negative_curation_data_identity.json) 不含私有逐文件名称/哈希/坐标或原图。

manifest SHA-256：858e0140b56a26f65476249896adc69f3de296856e99c21f8d3b58d0916f24e3；最终审批 SHA-256：f4ec2324aaa39930c00e0ddba5d18809f63f1569a3d7960a51d2e1c981255c6b。

## 实际训练与复现

v11 已完成，从 v3 SHA-256 48f284f9094fe334e02c66647f95fed8bfb3396b09f4488753508f514ed38980 出发，仍为不冻结主干、640、batch=4、seed=20260930、AdamW、lr0/warmup_bias_lr=0.00002、最多 12 轮/patience=8。已比较实际 args.yaml 与 v9：仅 data/name/save_dir 不同。新的数据审核检查不改变梯度或增强配置，但实际采样条目/批数与早停轮数需报告。

[启动身份](../../docs/hardcase_v11_experiment_identity.json) 记录实际校验依赖脚本和训练身份。完成后的 runner 已绑定实际 PID，已比较 v3/v11 best/last 的同一 55 派生图、五图蓝焰、旧 286 图、来源分组 Commons 和九段视频，补测了 v9 best/last 的同一 55 图。另已按来源检查烟雾/反光误检和完整画面食物误检，不能仅看裁剪。

```powershell
python -B model_work/src/build_reviewed_source_ablation.py --parent model_work/data/hardcase_v9_sources_20260930 --changes-review model_work/data/negative_curation_review_20260930.json --evidence model_work/out/training_color_review_20260930_v2/review_complete.json --out model_work/data/hardcase_v11_negative_curation_20260930
python -B model_work/src/train_hardcase_v5.py --data model_work/data/hardcase_v11_negative_curation_20260930/data.yaml --start model_work/runs/fire-binary-user-video-v3/weights/best.pt --expected-sha256 48f284f9094fe334e02c66647f95fed8bfb3396b09f4488753508f514ed38980 --out model_work/runs/fire-binary-hardcase-v11-negative-curation-20260930 --epochs 12 --lr0 0.00002 --warmup-bias-lr 0.00002
```

构建仍先输出未批准数据；命令不自动审批或训练，已存在目录拒绝覆盖。私有详细审批不在公开仓库，其他机器需自行取得和审核数据，不能用公开汇总冒充逐图审批。

本机 58 项测试通过（新增 6 项审核采样守卫、1 项负例拼版篡改测试；基础 50 项加 8 项 PyTorch 审计）。训练及完整对照已完成，结果见下节；候选按 [用户指定的演示范围](../../docs/DEMO_SCOPE.md) 判断，贴纸抑制不是必修目标。独立厨房完整事件、同步传感器、量化和实机验收仍未完成。

## 完整结果与决定

实际完成 10 轮早停，CSV 记录耗时 1521.19 秒。best SHA-256 为 17ba88fbcaca0e5e1aaf436bee2c84f0b64ae4c11def96f37613d964cdb38126；last 为 cbcb5f9300679df7d10f55f2878cabb4405130b8cf0b1a6103b32329a720366c。训练、完整 runner 与来源定位均 exit 0。与 v9 实际 9 轮的比较包含采样批数和早停轮数差异，不能把全部变化只归因于某张负例。

| 诊断 | v3 | v9 last | v11 best | v11 last |
|---|---:|---:|---:|---:|
| 五图蓝焰定位 | 0/5 | 3/5 | 0/5 | 3/5 |
| 旧 286 图 mAP50 | 0.97324 | 0.90044 | 0.95838 | 0.90824 |
| 同一 10 个已曝光正例 TP/FP/FN | 3/5/7 | 10/3/0 | 5/5/5 | 10/2/0 |
| 同一 24 个旧视频帧 TP/FP/FN | 18/0/0 | 18/0/0 | 18/0/0 | 18/1/0 |
| 已入训练的 10 张 Commons/13 GT TP/FP/FN | 1/3/12 | 7/2/6 | 2/3/11 | 9/1/4 |
| 原无可见火焰/烟雾来源 100 图 FP 框 | 1 | 9 | 2 | 10 |

v11 last 蓝焰还是同样三图定位，剩余 Cardoner 和 stovetop 无框。旧 Commons 中本轮未训练蓝焰源仍仅 1/2 正确定位（conf .25/.5），不比 v9/v10 改善。训练来源和未训练开发来源分组见 [公开完整汇总](../../docs/hardcase_v11_comparison_results.json)。所有这些图都曾用于开发分析，不是独立验收。

食品无火裁剪仍为零误检，但打开对应完整画面确认锅内食物框 conf .3318，不能用裁剪成绩宣称解决食物误检。另一个完整画面的 v10 重复部分火焰框本轮未出现。24 旧视频帧新增一处 conf .2581 反光前景框；该帧灶台无可见火焰。逐图审核、预测与图像身份保存在本机 manual_review.json，不公开私有画面和逐文件身份。

来源固定阈值复核显示 v11 last 保持 ks_flame 57 TP/0 FN，但 FP 增至 4；kitchen_stove_fire 为 17/3/3；无可见火焰/烟雾来源 FP 为 10，没有达到本轮降低误检的目的。详细汇总见 [v11 来源复核](LEGACY_SOURCE_LOCALIZATION_V11_20260930.md)。固定阈值计数不能完全解释 mAP 排序与高 IoU 框质量。

决定：不替换 v3，不把 v11 作为新的演示候选。原因是蓝焰未再改善、烟雾来源误检未减少且食物/关火反光误检存在；贴纸仍只是可选边界，未作为单独否决理由。下一步先做固定权重的新来源短视频探查，不继续凭这一小组已曝光图片叠加微调。完整独立厨房事件、真实同步融合、量化与实机验收仍缺证据。
