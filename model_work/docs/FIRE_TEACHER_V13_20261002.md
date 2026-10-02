# v13：固定特征的火焰头训练与旧输出约束

用户要求持续工作，得到实测优于 v3 的版本。本次直接处理此前蓝焰适配与旧火焰退化的矛盾；烟雾头兼容实验不视为已完成该目标。板子未连接，本次全部工作在电脑上完成。

## 开跑前确定的方案

源权重仍为原 v3：SHA256 `48f284f9094fe334e02c66647f95fed8bfb3396b09f4488753508f514ed38980`。冻结前 23 个特征模块，只训练原 fire Detect 的 819779 个参数。源文件不覆盖。除了固定特征模块，固定检测头 BatchNorm 统计；每轮核对 live 状态，并将这些固定状态精确复制到 EMA 后核对。复制一个 v3 检测头作教师，禁止其梯度、统计量或参数更新；它从同一批固定特征计算旧输出，无须运行第二个完整主干。

监督数据来自已审核 v11 全量版本，无来源抽样削减，共 2432 张/2810 次采样。应用此前人工复核的 5 张、6 个可见火焰包络修正，复制到新目录而保留原始文件。未批准的提案不变更标签。父集图像、标签、采样权重、审核文件和渲染身份逐一核对。新 manifest SHA256 `56496195e463bd11f1071bb7299dab8198175911403016d11ef79bc03c61ccc7`。派生目录为本机 `D:/SRTP_Datasets/fire_teacher_v13_20261002`。

2372 张未修正的 legacy train 图片接受教师约束；5 张修正图和 55 张已审核派生难例只接受监督标签，防止教师保留旧漏检。分类约束为 Bernoulli KL，框约束为教师概率大于 .05 的 anchor 上按概率加权的 16-bin categorical KL；两项 gain 均为 1，不把教师输出当人工真值。禁用 mosaic/mixup，以保证每张图的约束归属明确；其他增强仍施加于同一学生/教师特征。

固定 12 轮、不早停、不预热、AdamW lr0=.0001/lrf=.1/WD=.0005，640、batch8/nbs8、GPU0/workers0/seed20261002。保留真实 batch 和 optimizer.step 计数，记录每轮教师损失及实际生效图片数。现有环境 Torch 2.12.0+cu132、Ultralytics 8.4.70、RTX 5060 Laptop；不升级依赖。

预先声明比较 best、last、epoch3、epoch6、epoch9。Ultralytics 的周期文件使用从零开始的编号，分别对应完成第 4、7、10 轮的快照。框架另存 epoch0；其早期诊断只检查退化，不加入上述候选名单。best 由原 legacy val 的框架 fitness 选择，此 val 已曝光。

沿用旧门槛：原 286 图 mAP50 降幅不超过 .01，5 张蓝焰主火焰 IoU≥.5 定位至少增加 1 张，无火室内 100 图 FP 增量不超过 1。再检查原 ks_flame 57 框 TP≥56、kitchen_stove 20 框 TP≥18。全部使用 conf=.25/NMS IoU=.6/匹配 IoU=.5。合格候选按蓝焰定位数、旧 mAP50、室内 FP 顺序排列；门槛不随结果调整。另保存 55 张训练暴露难例的定位和负例计数，合格候选还须复查开发视频。

这些图片、视频和来源已用于开发，比较只能支持版本改进判断，不能替代独立跨厨房或事件验收。原正式计划书指标仍按证据矩阵记录。

## 复现入口与运行状态

```powershell
python -B model_work/src/fire_label_revision_dataset.py --parent model_work/data/hardcase_v11_negative_curation_20260930 --review model_work/out/fire_envelope_review_20261002/proposal/review_complete.json --out D:/SRTP_Datasets/fire_teacher_v13_20261002
python -B model_work/src/train_fire_teacher_v13.py --data D:/SRTP_Datasets/fire_teacher_v13_20261002/data.yaml --expected-manifest-sha256 56496195e463bd11f1071bb7299dab8198175911403016d11ef79bc03c61ccc7 --start model_work/runs/fire-binary-user-video-v3/weights/best.pt --expected-start-sha256 48f284f9094fe334e02c66647f95fed8bfb3396b09f4488753508f514ed38980 --out model_work/runs/fire-teacher-head-v13-20261002
python -B model_work/src/screen_fire_teacher_v13.py --run model_work/runs/fire-teacher-head-v13-20261002 --data D:/SRTP_Datasets/fire_teacher_v13_20261002 --out model_work/out/fire_teacher_v13_screen_20261002
```

这些命令拒绝覆盖已存在的数据、run 或评测目录；本机已有启动记录，不重复开跑。训练日志与逐图预测只留本地，Git 保存源码和不含原图/权重的汇总。

当前训练进行中，前 3 轮特征/教师/头 BN 状态检查均通过；最终质量尚未评测，不据此宣称超过 v3。新增数学梯度测试 3 项与数据修正防篡改测试 4 项通过。后续在本记录追加实际结果。
