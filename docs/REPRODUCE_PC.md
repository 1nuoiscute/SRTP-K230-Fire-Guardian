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
