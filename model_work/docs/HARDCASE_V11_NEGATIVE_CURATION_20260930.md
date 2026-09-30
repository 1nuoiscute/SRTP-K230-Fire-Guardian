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

v11 已启动，从 v3 SHA-256 48f284f9094fe334e02c66647f95fed8bfb3396b09f4488753508f514ed38980 出发，仍为不冻结主干、640、batch=4、seed=20260930、AdamW、lr0/warmup_bias_lr=0.00002、最多 12 轮/patience=8。已比较实际 args.yaml 与 v9：仅 data/name/save_dir 不同。新的数据审核检查不改变梯度或增强配置，但实际采样条目/批数与早停轮数需报告。

[启动身份](../../docs/hardcase_v11_experiment_identity.json) 记录实际校验依赖脚本和训练身份。完成后的 runner 已绑定实际 PID，将比较 v3/v11 best/last 的同一 55 派生图、五图蓝焰、旧 286 图、来源分组 Commons 和九段视频，补测 v9 best/last 的同一 55 图。另需按来源检查烟雾/反光误检和完整画面食物误检，不能仅看裁剪。

```powershell
python -B model_work/src/build_reviewed_source_ablation.py --parent model_work/data/hardcase_v9_sources_20260930 --changes-review model_work/data/negative_curation_review_20260930.json --evidence model_work/out/training_color_review_20260930_v2/review_complete.json --out model_work/data/hardcase_v11_negative_curation_20260930
python -B model_work/src/train_hardcase_v5.py --data model_work/data/hardcase_v11_negative_curation_20260930/data.yaml --start model_work/runs/fire-binary-user-video-v3/weights/best.pt --expected-sha256 48f284f9094fe334e02c66647f95fed8bfb3396b09f4488753508f514ed38980 --out model_work/runs/fire-binary-hardcase-v11-negative-curation-20260930 --epochs 12 --lr0 0.00002 --warmup-bias-lr 0.00002
```

构建仍先输出未批准数据；命令不自动审批或训练，已存在目录拒绝覆盖。私有详细审批不在公开仓库，其他机器需自行取得和审核数据，不能用公开汇总冒充逐图审批。

本机 58 项测试通过（新增 6 项审核采样守卫、1 项负例拼版篡改测试；基础 50 项加 8 项 PyTorch 审计）。训练与效果尚未完成；候选按 [用户指定的演示范围](../../docs/DEMO_SCOPE.md) 判断，贴纸抑制不是必修目标。独立厨房完整事件、同步传感器、量化和实机验收仍未完成。
