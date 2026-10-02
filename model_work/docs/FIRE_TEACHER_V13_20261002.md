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

训练期间已完成以下固定早期诊断；全部候选的最终比较尚未结束。新增数学梯度测试 3 项与数据修正防篡改测试 4 项通过；全量 114 项本机测试通过，375 个公开文件检查通过。源码提交 `3b8f28f` 已推送开发分支。

| 固定快照 | 旧 286 图 mAP50 | 蓝焰主火焰定位 | ks_flame TP/FP/FN | kitchen_stove TP/FP/FN | 室内无火 FP |
|---|---:|---:|---|---|---:|
| v3 | .973239 | 0/5 | 57/1/0 | 18/5/2 | 1 |
| v13 epoch0（只作诊断） | .957222 | 未另测 | 未另测 | 未另测 | 未另测 |
| v13 epoch3（完成第 4 轮） | .971276 | 1/5 | 57/2/0 | 18/4/2 | 1 |

epoch3 的 SHA256 为 `484d9857b02d238facf4f1594823658ccc2e5632e2a5d28306e784e63e0d0bcc`，早期固定门槛通过；不以它代替未结束的完整候选比较。old 75 张灶台无火与 40 张锅来源均 0 FP，和 v3 相同。人工查看 blue_oven 叠加图，预测覆盖可见蓝焰环，近似主框 IoU=.707762；另两个检出的蓝焰图框偏大，未计正确定位。

55 张暴露难例中，Commons 蓝焰 13 个框由 v3 的 TP/FP/FN=1/3/12 改为 4/1/9；队友火焰由 3/5/7 改为 5/4/5；旧视频火焰 18/0/0 保持，食物负例 0 FP 保持，贴纸 crop 3 FP 降为 2。仍有 5 个队友火焰漏检，不称完整跨厨房识别。

epoch3 静态与动态 ONNX 各完成 80/80 图预处理、raw 和 NMS 一致性，固定容差未放宽。静态 SHA256 `2dd6fc6d82714fc827443fc3fde0628bc3b0f20dd1d7bc22278b115e25b233f3`；动态 SHA256 `cf3c9210505f749e96ae048c358d5c1f4ddca788a0096a7831dc60819c397705`。输出仍为单类原 Detect，参数量和输出布局未增加。四段旧视频及五段队友视频完成固定 1fps 回放；有框数量只作行为检查，不计准确率或事件误报警。

## 完整结果与版本保存

12 轮正常完成，4224 batch/4216 次实际更新；14 次 live 特征核对、12 次教师与 EMA 核对通过。另逐张量检查所有声明候选的保存状态：432 个固定特征/BN 状态相同，64 个变化张量均在 fire 头内。best 与 epoch3 所有模型张量完全一致；最终选择 v13 best。

| 候选 | 蓝焰/5 | 旧 mAP50 | 固定筛选 |
|---|---:|---:|---|
| best | 1 | .971276 | 通过，选定 |
| epoch3 | 1 | .971276 | 通过，与 best 张量相同 |
| epoch6 | 1 | .957979 | 旧 mAP 未通过 |
| epoch9 | 1 | .940804 | 旧 mAP 未通过 |
| last | 1 | .939614 | 旧 mAP 未通过 |

更多训练轮数继续提升训练难例，却再次损伤旧图表现；没有采用 last，也没有放宽门槛。本实验联合标签修正、冻结、BN 保护和教师项，未隔离各操作的因果贡献。平均分类 KL 约 .01、框 bin KL 约 .01，与监督项的量级不同，后续如调整约束强度必须作为单独实验，保留本结果。

9 段画框视频统一抽开始/中间/结束各 1 帧，3 张联系图已实际目视复核：旧无火视频抽样无框，三段旧灶火抽样框在可见火焰附近；新开口蓝焰环三个抽样均正确画框；锅下小蓝焰在另外视频抽样仍漏检，蓝色贴纸仍有错误框。该检查覆盖 27 个画面，无法代替连续逐帧真值。复核图和逐视频身份留在 `model_work/out/fire_teacher_v13_video_review_20261002/`。

独立 CPU 图片入口实际检出 blue_oven，动态视频入口完整解码一段队友视频 719 帧、采样 24 帧/24 个框；输出与 PyTorch 同一模型状态，采样有框数不是正确率。

选定文件复制到 `model_work/out/fire_teacher_v13_selected_20261002/`，新权重 SHA256 `c2b2a8bdc66fe452c80a1786f918f62f8686ad3b0f11061a761969ee02b09966`。ONNX 沿用已核验的等价 epoch3 导出并在版本清单明确两容器身份；v3、所有候选和原数据保留。见 [v13 模型卡与运行入口](../../docs/PC_FIRE_V13_MODEL_CARD.md)、[公开完整计数](../../docs/fire_teacher_v13_results_20261002.json)。v13 是综合开发诊断更好的候选，独立厨房、全蓝焰和事件验收仍不足。
