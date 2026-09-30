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
