# 电脑端复现

已用环境：Windows、Python 3.13、PyTorch 2.12.0+cu132、Ultralytics 8.4.70、OpenCV 4.13.0，RTX 5060 Laptop 8GB。版本是本机实测记录；CUDA 构建按运行机器配置，不由普通 requirements 强制安装。

```sh
python -m pip install -r requirements-pc.txt
python -B -m unittest discover -s model_work/src -p 'test_*.py' -v
```

融合演示所需的 CSV 与 JSON 位于 model_work/data。它们是模拟数据，未用于实际报警：

```sh
python -B model_work/src/fusion_replay.py replay --input model_work/data/fusion_demo_scenarios_20260927.csv --config model_work/data/fusion_demo_config_20260927.json --out model_work/out/new_fusion_run
```

原视频和大型数据不在仓库，按 references 里的来源/许可自行获取。旧模型与数据沿革见实验报告。新实验构建、训练和评测均输出到新目录并拒绝覆盖；v5 命令及数据身份记录在本轮实验报告中。

## v5 对照工具

- build_hardcase_v5.py：以原数据清单和已审核帧构建开发集，拒绝覆盖。
- train_hardcase_v5.py：校验起点权重与数据审核身份，记录参数、环境、best/last 哈希。
- eval_hardcase_frames.py：按一对一框匹配统计开发样本 TP/FP/FN。
- eval_blue_localization.py：五张诊断照片的主要炉位近似定位。
- eval_development_videos.py：仅接受 exposure registry 中已用于开发的视频，保存预测与带框视频。
- run_hardcase_comparison.py：只等待命令行与创建时间均核实的训练进程，不自动重训；完成后顺序比较三个权重。

各评测输出到独立新目录。runner 不部署、不推送，失败时保留日志，需查明后继续未完成项目。

## 两份既有蓝焰诊断原作的训练数据扩展

构建方法、人工审核和哈希见 [数据准备记录](../model_work/docs/BLUE_SOURCE_DATA_PREPARATION_20260930.md)。重复权重不增加独立照片数；v7 训练与完整评测已经完成，具体配置与拒绝替换结论见 [v7 实验](../model_work/docs/HARDCASE_V7_EXPERIMENT_20260930.md)。

新增 dataset_preflight.py 在训练前验证实际图像/标签身份、采样权重和配置路径；带有 label_sha256 的清单会拒绝审核后的标签修改。历史 v5 清单没有逐标签哈希，记录会如实报告其绑定覆盖数，不把它当成同等强度校验。

## 视觉结果转融合输入

新版转换需指定 --config，输入采用带 width/height/predictions 的开发预测 CSV；ROI JSON 逐视频声明尺寸及 roi_xyxy。配置、输出哈希在回放时核验。命令和迁移说明见 [边界修复记录](../model_work/docs/VISUAL_FUSION_BOUNDARY_FIX_20260930.md)。

Commons 诊断脚本现要求 --comparison-data 指向本轮训练 build_manifest.json；按原作身份分别统计训练来源与本轮未训练的开发诊断来源。不能用不同下载尺寸绕过这个分组。

可选 --freeze-backbone 按实际 checkpoint 的 backbone 长度冻结，并在训练运行中核验主干参数与 BN 缓冲。默认不启用，既有实验日程不变。命令与范围见 [v8 冻结对照](../model_work/docs/HARDCASE_V8_FREEZE_EXPERIMENT_20260930.md)。

第二批来源和无火裁剪的构建器 build_reviewed_source_queue.py 保留父数据、排除暂缓照片，并输出未批准的数据目录。逐图复核及训练方法见 [v9 数据与实验](../model_work/docs/HARDCASE_V9_SOURCE_DATA_EXPERIMENT_20260930.md)。dataset_preflight.py 现在还核对 source_reviews 实际文件和最终新增审核图；训练记录绑定本地校验依赖脚本哈希。

原训练集分层目视抽查工具 prepare_training_color_review.py 和来源移除构建器 build_reviewed_source_ablation.py 已加入。前者只生成固定抽样拼版，不产生颜色真值；后者只派生未批准数据，不自动重标或训练。具体审核、失败与身份见 [抽查](../model_work/docs/TRAINING_COLOR_QUALITY_REVIEW_20260930.md) 与 [v10](../model_work/docs/HARDCASE_V10_SOURCE_ABLATION_20260930.md)。

负例审核构建器新增 --changes-review 模式，拒绝未批准采样、将有火标签当负例及身份/权重变动；审批还绑定最终负例拼版。具体身份与流程见 [v11](../model_work/docs/HARDCASE_V11_NEGATIVE_CURATION_20260930.md)。eval_legacy_source_localization.py 输出按来源的固定阈值 TP/FP/FN 与本机错误图，公开只保留汇总，见 [来源定位](../model_work/docs/LEGACY_SOURCE_LOCALIZATION_20260930.md)。
