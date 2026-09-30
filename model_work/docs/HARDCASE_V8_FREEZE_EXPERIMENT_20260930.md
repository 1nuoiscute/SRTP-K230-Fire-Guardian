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

训练已启动，效果待评测。来源边界沿用 v7：Commons 两个蓝焰原作已进训练，剩余未训练诊断仍非盲测；独立厨房/完整事件验收、量化和板端运行继续未完成。
