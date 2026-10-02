# v14：随机初始化全网络训练与路线对照

用户授权继续模型推进，并要求可以从零训练作比较。这里的从零指全部可学习卷积随机初始化，不载 COCO 预训练或 v3/v13 的参数；架构仍取 YOLO11s，以免把架构变化混进路线比较。电脑没有连接板子，v13 开发候选与原 v3 保留。

## 初始化与固定训练方案

初始化 seed=20261003，9428179 参数。从原 v3 只读取网络 YAML；87 个非 DFL 卷积全部与源权重不同，同种子第二次构建全部状态逐张量相同。保存 random_initial.pt 使用 FP32，不通过 FP16 舍入；载入核对一致。DFL bin 投影是固定算子，BN 初始统计属于确定性初始化，这些不叫继承训练权重。

- 结构 YAML SHA256：`1d21fee6d41a56681324bd0d1be5d9932a4f03a95b6e4bfb775e98b30156514e`。
- 随机初始文件 SHA256：`2707c671df6990902b9e316071778d26546807380d30791eadb1a8f181eb21a3`。
- 随机模型状态 SHA256：`6932a979359fe6b3bba6ffbc5b5cadde37132dd123b583d58ce4a1d5519a4d57`。
- 初始化清单 SHA256：`defc84bda87b71574b3b0282f35930de19a9958b40da151407d5b88dacb8b3f7`。

训练器明确 pretrained=False/resume=False，get_model 拒绝任何 supplied weights，用隔离 CPU RNG 的固定种子直接构建 DetectionModel。首个 batch 前比较实际模型与记录的随机模型全部状态，已通过，实际可训练 9428163 参数，仅 DFL 固定 16 个参数。这避免只检查调用参数却不知道训练器是否重建、载入了另一套模型。

使用 v13 同一个已审核标签修正数据，2432 张/2810 次采样，manifest SHA256 `56496195e463bd11f1071bb7299dab8198175911403016d11ef79bc03c61ccc7`。标签、父数据身份和采样不改，val 沿用 243 张已曝光 legacy selector，不称独立验证。新图中 smoke 缺席没有判真值，任务仍单类可见 fire。

固定100轮、patience=0、640、batch8/nbs8、AdamW lr0=.001/lrf=.01/WD=.0005，warmup3轮、warmup_bias_lr=.001、GPU0/workers0；HSV=.015/.3/.25，degrees3/translate.05/scale.2/fliplr.5/mosaic.5/mixup0，最后10轮关闭mosaic。实际optimizer.step另计，AMP跳过更新不得算实际更新。每轮核对 feature 与 head 均发生学习变化，保存状态摘要和更新次数。

100轮相对 v13 的12轮头部适配有更长优化过程；增强和学习率也不同。它比较实际训练路线，不能把结果解释为仅初始化变量的因果消融。若另作等预算初始化对照，必须新建固定方案，而不是复用本轮作该声明。

## 对照范围

最终 primary candidates 预先固定 best/last；每10轮保存的 epoch0/10/20/…90 仅用于收敛诊断，编号从零开始。正式比较 v3、已选 v13 best、scratch best/last，统一旧286图mAP50/50–95和分来源固定阈值TP/FP/FN、五张蓝焰主定位、55张暴露难例。阈值仍conf=.25/NMS IoU=.6/定位匹配IoU=.5。不能用训练loss下降、val变好或更多有框画面代替该比较。

沿用已冻结旧图/蓝焰/室内误检门槛，同时与 v13 实际指标作成对检查。若候选不合格，保留 v13；只有经过完整图片、无火及视频检查的候选才能成为新电脑端候选。上板、独立厨房、完整两类和事件要求继续按证据矩阵；不在这轮提前宣称。

## 当前实际状态与复现

已经启动单个本地训练任务，真实随机初始化检查通过。第一轮特征权重126张量、头权重42张量变化，352个batch/343次实际更新；首轮val mAP50=.00491，尚处随机起点学习阶段，不以首轮低分否定从零路线。GPU当次显存约3.6GB，训练正在进行，没有同时启动第二份GPU训练。

```powershell
python -B model_work/src/prepare_random_fire_start.py --reference model_work/runs/fire-binary-user-video-v3/weights/best.pt --expected-reference-sha256 48f284f9094fe334e02c66647f95fed8bfb3396b09f4488753508f514ed38980 --out model_work/out/fire_random_v14_initial_20261002
python -B model_work/src/train_fire_from_scratch_v14.py --data D:/SRTP_Datasets/fire_teacher_v13_20261002/data.yaml --expected-manifest-sha256 56496195e463bd11f1071bb7299dab8198175911403016d11ef79bc03c61ccc7 --initial model_work/out/fire_random_v14_initial_20261002 --expected-initialization-sha256 defc84bda87b71574b3b0282f35930de19a9958b40da151407d5b88dacb8b3f7 --out model_work/runs/fire-from-scratch-v14-20261002
```

本机已有初始目录、preflight和运行进程，不能重复执行这两条启动命令。查看同一任务的日志/句柄，等待结束或明确错误再处理；观察超时不等于训练停止。初始化、源数据、run和详细日志只留本地，Git仅收源码和记录。
