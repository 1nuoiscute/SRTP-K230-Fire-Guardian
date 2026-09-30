# 通用火灾来源移除对照 v10｜2026-09-30

v9 last 的五张蓝焰诊断定位由 v7 的 1/5 到 3/5，但旧图 mAP50=0.90044、贴纸误检仍在，拒绝替换。原训练集抽查显示厨房来源已有不少蓝焰，另发现 generic_fire 的三张空标签图实际有火，以及可疑烟火框语义；详见 [实际抽查](TRAINING_COLOR_QUALITY_REVIEW_20260930.md)。v10 检验在此次微调中移除 generic_fire 整个来源的影响。

## 单因素与数据身份

- 父数据仍为 v9（manifest SHA-256 7a262442e6227a0da6277b03c3b9211e65effc3d8e3e68e71995922fc636ecc2）。
- 按实际 base manifest 的 train provenance 移除 generic_fire 800 份原图和 800 个条目，不靠文件名猜来源。
- 剩余 1582 份旧原图、55 份派生文件，共 1637 个文件、1982 个采样条目；每份保留 lineage 字段、标签/图像字节及权重与 v9 一致。
- 新增八份蓝焰、食物裁剪、两份旧蓝焰原作、队友及旧视频帧均保留；没有重标旧锅体框，不混入第二个因素。
- 两份旧审核拼版、逐原作渲染、source_reviews 文件复制后身份一致；本轮审批只确认来源移除与继承身份，不冒充新逐图标注。
- 800 份原数据继续留在原路径。移除整个来源不意味着 800 张全错，亦不是修复了 800 个标签。

起点仍为 v3；v3 的训练沿革曾使用 generic_fire。此次消融只移除当前微调的该来源，不是整个训练链从未接触该来源。所有诊断均为开发用途。

最终数据目录 model_work/data/hardcase_v10_source_ablation_utf8_20260930；manifest SHA-256 794d7623fa422049b0d8ab6694d92afe278de67e6f16a97a31763f8a52645fde，审批 SHA-256 3f79d95f591338e3b37322f759f2faeedf24f5871d5a03ffb44076069a13a4ef；预检查逐份绑定全部 1637 个图像和标签。公开 [数据身份](../../docs/hardcase_v10_source_ablation_data_identity.json)、[实际训练身份](../../docs/hardcase_v10_experiment_identity.json)。

## 实际启动与评测

v10 已实际启动。核对 args.yaml 与 v9，只有 data/name/save_dir 不同；v3 初始化哈希、seed、增强、AdamW、640、batch=4、lr0/warmup_bias_lr=0.00002、最多 12 轮/patience=8、freeze=None 一致。每轮批数由 v9 的 696 变为 496，样本组成和更新次数一起变化；不能声称等更新预算的颜色因果试验。

run_hardcase_comparison.py 已绑定实际训练进程，完成后顺序比较 v3/v10 best/last 的同一 55 派生图、五图蓝焰、旧 286 图、按原作曝光分组的 Commons 和九段视频；同一 55 图再比较 v9 best/last。训练成功不自动接受候选，结果待完整评测，不改板端启动产物。

```powershell
python -B model_work/src/build_reviewed_source_ablation.py --parent model_work/data/hardcase_v9_sources_20260930 --exclude-provenance generic_fire --evidence model_work/data/hardcase_v10_source_ablation_utf8_20260930/source_reviews/training_quality_evidence.json --out model_work/data/hardcase_v10_source_ablation_utf8_20260930
python -B model_work/src/train_hardcase_v5.py --data model_work/data/hardcase_v10_source_ablation_utf8_20260930/data.yaml --start model_work/runs/fire-binary-user-video-v3/weights/best.pt --expected-sha256 48f284f9094fe334e02c66647f95fed8bfb3396b09f4488753508f514ed38980 --out model_work/runs/fire-binary-hardcase-v10-source-ablation-20260930 --epochs 12 --lr0 0.00002 --warmup-bias-lr 0.00002
```

构建命令先输出未批准状态，不能直接训练。需检查移除清单、全部保留字段/字节/权重、审核图身份，再记录审批并通过 preflight。工具不自动训练、不重启、拒绝覆盖。

## 保留的失败与修复

首个目录 hardcase_v10_source_ablation_20260930 始终未批准、未用于训练。首次审批脚本因 Windows GBK 默认读取 UTF-8 而停止；改为 UTF-8 后，逐字段比对又发现构建脚本默认读取把 André 署名改成了乱码。即使图像和标签字节相同，也不能接受来源元数据悄悄改变。

修复构建器所有 JSON/YAML 读取为明确 UTF-8；重建到带 _utf8 的新目录。新的 1637 份 lineage 完全等于 v9 对应记录，包括非 ASCII 署名。检查新目录仍未批准时 preflight 确实拒绝，再审核/批准后预检查全部通过。旧目录和 pending 审批保留，原件没有修改。

本实验仍缺独立厨房完整事件、无火长时误报率、同步传感器与实机验收。它检验下一步数据方向，不是最终系统验收。

公开审核 JSON 已改为汇总版本，逐文件身份及来源细节仅留在本机。实际训练数据中的 source_reviews/training_quality_evidence.json 保留构建时完整快照，没有修改，preflight 与正在运行的训练身份不受影响；复现构建可使用这份本地冻结证据，已有输出目录仍会拒绝覆盖。

用户在本轮运行期间调整演示范围：贴纸干扰不作为必须解决目标，演示采用无此干扰灶台。v10 数据、配置和完整评测保持不变，完成后仅贴纸误检不能单独否决演示候选；真实火焰与演示内其他误检仍需判读。参见 [范围说明](../../docs/DEMO_SCOPE.md)。
