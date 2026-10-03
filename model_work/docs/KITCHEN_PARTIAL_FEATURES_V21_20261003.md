# v21：全画面后段特征适配｜2026-10-03

**当前只完成CPU预演和比较协议，正式训练尚未启动。** 演示v16 best、定位开发v18 last保持；不以4张训练正例拟合晋升。

采用新来源可学习性诊断的下一条路线，在完整1445唯一图/2409采样条目上适配模块10–22和检测头23，前10模块保留。所有特征BN运行统计固定、后段BN仿射参数可训练；检测头BN沿用普通训练。这与诊断中所有BN固定不同，不能把正式收益归因于唯一因素。与v20正式路线相比，数据/采样、初始化、增强、轮数/优化器日程保持，仅开放后段特征。

8图真实CPU预演已exit0：4新正例、2新负例、2旧火焰，1次实际更新。499状态精确重建、60个后段特征权重张量和42个头权重张量改变；冻结前缀、全部特征BN、DFL保持，所有训练梯度有限、冻结梯度缺席。4977155可训练参数。预演不是候选模型，不保存权重。

正式计划：v16 best起点、12轮、AdamW lr=.00005/lrf=.1、batch/nbs8、seed20261006、imgsz640、workers0、freeze10。全部新4预留/3隔离身份继续排除；新14训练图分开计反馈。每轮核验实际特征/头更新、冻结前缀/BN状态/EMA，最终best/last分别核验保存权重的固定状态和实际适配。

比较协议在启动前冻结：保留v16 best、v18 last、v19 last、v20 best/last五参照，两份v21均做原286、原55、蓝焰5、物理15框、新14训练来源；9视频307同帧比较及实际视觉复核。不得按旧AP提前丢弃，也不得凭训练拟合采用。导出/CPU入口按原一致性要求另验。

用户提出安装后手动选择重点区域。建议另作“重点区域识别+定期全画面巡视”的电脑端验证，避免把依赖已知目标的标注裁剪当可部署功能；摄像头移动后需重新确认区域，区域外事件仍要覆盖。当前只是方案讨论，尚未实现ROI功能，不声称有实际收益。

[预演冻结记录](../../docs/kitchen_partial_features_v21_preflight_20261003.json)、[比较协议](../../docs/kitchen_partial_features_v21_comparison_protocol_20261003.json)、[前一诊断](KITCHEN_SOURCE_LEARNABILITY_V1_20261003.md)。

```powershell
python -B model_work/src/train_kitchen_partial_features_v21.py --start model_work/runs/fire-scratch-semantic-v16-r2-20261003/weights/best.pt --data D:/SRTP_Datasets/fire_kitchen_source_v20_20261003/data.yaml --out model_work/out/v21_partial_cpu_rehearsal_20261003 --rehearse-cpu
python -B model_work/src/train_kitchen_partial_features_v21.py --start model_work/runs/fire-scratch-semantic-v16-r2-20261003/weights/best.pt --data D:/SRTP_Datasets/fire_kitchen_source_v20_20261003/data.yaml --out model_work/runs/fire-kitchen-partial-features-v21-20261003 --rehearsal model_work/out/v21_partial_cpu_rehearsal_20261003
python -B model_work/src/screen_kitchen_partial_features_v21.py --run model_work/runs/fire-kitchen-partial-features-v21-20261003 --data D:/SRTP_Datasets/fire_kitchen_source_v20_20261003 --review model_work/out/legacy_visible_flame_audit_20261002 --previous model_work/out/kitchen_source_v20_comparison_20261003 --out model_work/out/kitchen_partial_features_v21_comparison_20261003
```

已有输出拒绝覆盖；正式源必须与最终CPU预演一致。训练实际启动/完成与推理结果另记录，不把预演当正式运行。
