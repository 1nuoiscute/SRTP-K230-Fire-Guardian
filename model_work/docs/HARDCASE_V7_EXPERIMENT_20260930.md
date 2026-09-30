# 蓝焰来源增补 v7｜2026-09-30

v6 完成后仍没有解决外部蓝焰定位，故不替换基线。本轮检验两张新来源照片的开发训练增补效果。

## 已固定的实验

起点仍为 v3，SHA-256：48f284f9094fe334e02c66647f95fed8bfb3396b09f4488753508f514ed38980。不从 v6 继续训练，避免混入额外适配阶段。

与 v6 相同：seed=20260930、imgsz=640、batch=4、AdamW、lr0=0.00002、warmup_bias_lr=0.00002、lrf=0.1、weight_decay=0.0005、计划最多 12 轮、patience=8，以及所有增强参数。唯一有意改变的训练因素是数据：在原 2654 条目上加入两张人工审核蓝焰照片各 12 次，共 2678 条目/2428 个文件。实际早停轮数须按完成记录报告。

新增照片、近似框、许可及来源见 [数据准备](BLUE_SOURCE_DATA_PREPARATION_20260930.md)。数据清单 SHA-256：0a40ce9345652e322c8af0cd2d8a5abb1831e94432065c4b546828d64ee47a3b。

训练前的新校验器核对审核清单/图、2428 个图像及标签、2678 个条目、权重，以及 YAML 的训练和验证分区。训练步长、增强或优化器未因校验器而改变。新校验和源代码身份写入 preflight。

```powershell
python -B model_work/src/train_hardcase_v5.py --data model_work/data/hardcase_v7_sources_20260930/data.yaml --start model_work/runs/fire-binary-user-video-v3/weights/best.pt --expected-sha256 48f284f9094fe334e02c66647f95fed8bfb3396b09f4488753508f514ed38980 --out model_work/runs/fire-binary-hardcase-v7-sources-20260930 --epochs 12 --lr0 0.00002 --warmup-bias-lr 0.00002
```

实际运行目录 model_work/runs/fire-binary-hardcase-v7-sources-20260930，日志和对照放在 model_work/out/hardcase_v7_sources_20260930。已启动训练，无完成结果。

## 预排对照

训练进程结束且 best/last 哈希验证通过后，顺序评测 v3、v7 best/last：原 44 派生样例、新两张照片、五张旧蓝焰、旧 286 图、Commons 13 图及九段开发视频。另在同一份 46 图上补测 v6 best/last，明确两张照片加入前后的差别；旧 v6 图/视频结果仍由其完整记录提供。

现有 best 仍由旧验证分区选择，并不自动代表厨房蓝焰最优。所有旧图/视频与两张新增来源均是开发材料；改善只可称开发回归或诊断改善。独立厨房/完整事件验收、每小时误报和板端量化/运行没有完成。
