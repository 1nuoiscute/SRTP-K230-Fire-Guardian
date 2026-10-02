# v14：随机初始化全网络训练与路线对照

**最新状态：100轮训练、best/last统一图片对照及五模型同帧视频比较均已完成。** 35200 batch/35180次实际更新；两份蓝焰主定位均4/5，锅底小蓝焰视频改善，但旧无火误检和锅体框问题仍在。按用户新的项目综合标准，优先推进last的电脑端运行验证，保留v13回退。动态ONNX 80/80图一致性及独立CPU图片/视频入口已完成。最终计数与取舍见 [完整路线比较](PROJECT_MODEL_ROUTES_20261002.md)。下方运行中快照保留时间顺序，原筛选门槛已改为历史诊断，不再作为采用的自动否决条件。

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

## 训练期间状态与复现（历史过程）

已经启动单个本地训练任务，真实随机初始化检查通过。第一轮特征权重126张量、头权重42张量变化，352个batch/343次实际更新；首轮val mAP50=.00491，尚处随机起点学习阶段，不以首轮低分否定从零路线。GPU当次显存约3.6GB，训练正在进行，没有同时启动第二份GPU训练。

61轮时val mAP50=.55790，前述23轮读数为.57660；这些只是运行中的开发验证进展，最终100轮和best/last对照尚未结束。全网络更新审计持续记录。初始化与固定比较工具的GitHub push/PR两项基础CI均通过（head `1510f1c`），训练图复核提交`5bd4f04`两项CI也均通过；加入后续训练前置检查后本机全依赖129项测试通过。旧灶火原框存在锅体范围差异，已独立完成 [补充诊断](../../docs/PC_FIRE_LABEL_SEMANTICS_20261002.md)，原标签和本轮训练保持。

### 固定周期快照诊断

从已声明的周期保存中固定`epoch20/50/90.pt`做收敛观察，均不进入最终primary候选池，不据此调训练或阈值。使用独立CPU进程、OMP/MKL各2线程、禁用该诊断进程的CUDA可见设备；GPU继续原100轮日程。`epoch20.pt`是零起编号，实际完成21轮，SHA256 `1846dd76c4de478a7aa02d7b3a737d169e7eb6cf7dfe246e223adf01c4f4264c`。其当轮val mAP50=.50175，不与最新运行中的val混淆。

第一个完整CPU诊断已结束：5张蓝焰主定位2/5（两张开口灶头，两个锅下蓝焰仍无框，另一张开口蓝焰被分成局部框）；55张曝光训练难例中Commons蓝焰TP/FP/FN=13/2/0、队友10/0/0、旧视频18/0/0、贴纸与食物FP=0。旧286图固定阈值ks_flame=53/0/4、kitchen_stove=15/4/5、无火室内FP=2，其余两组负例FP=0。它仍没达到旧火焰召回门槛，不因训练难例拟合好就晋升。

同一10图/15处近似可见火焰诊断：原标签11/3/4，紧火焰IoU≥.5为0/14/15、≥.3为4/10/11，平均最佳IoU=.219941。实际看完全部3张语义预测页和5张蓝焰预测图，仍见高锅体范围预测与蓝焰碎框。最终候选也应检查这一语义问题；它不是换初始化即可解决的证据。这里是CPU中途诊断，原v3/v13筛选是历史GPU测量；最终成对路线评测仍统一用同一设备，不把这组跨设备中途计数写作最终因果对照。

完整本地文件 `model_work/out/fire_random_v14_epoch20_diagnostic_20261002/diagnostic_complete.json`，公开 [收敛计数](../../docs/fire_scratch_v14_convergence_20261002.json)。工具拒绝未完成周期或已存在输出，绑定随机实际起点、原训练代码/依赖、checkpoint和补充复核身份；不会触碰训练目录。后续只在对应已保存周期完成后执行同一工具。

第二个固定快照`epoch50.pt`（实际完成51轮）也已完整结束，SHA256 `1875e3a374229751965acd139900610dc8d059d5d5b9fc03b7eaaca4323fa73d`，该轮val mAP50=.56231。蓝焰主定位3/5，55张曝光派生图全部41个正框匹配、FP/FN=0；旧ks_flame=57/0/0，工业灶火17/3/3，原无火室内命名来源FP=11（10张图），另两组负例FP=0。它改善了中途召回和蓝焰定位，但没有通过无火误检及工业灶火召回条件，不能晋升；尚未测本快照的正式旧图AP，不能从固定阈值计数推算AP。

补充10图/15框仍为原标签13/2/2，紧火焰IoU≥.5为0/15/15、≥.3为4/11/11，平均最佳IoU=.242962。已实际看完3页补充预测、5张蓝焰图和全部10张无火来源误检图：高锅体范围预测保持；两张锅下蓝焰仍无框，开口灶头3张能匹配，其中一张还有重叠局部框。误检包括视频水印、日落、纸板边缘、桌面手机线缆、工业蓝色物体和烟雾区域；另3张重复桌面亮区的光源性质待核。原来源名称不代表每图均室内，重复帧也不等于独立失败场景。原标签继续保留，不借中途输出重标验证集。完整结果和18张已查看文件身份位于本地`fire_random_v14_epoch50_diagnostic_20261002`，汇总已追加到同一公开收敛记录。

第三个固定快照`epoch90.pt`（实际完成91轮）已完整CPU诊断，SHA256 `a3c17f7b2a3e7b36f45194e31580c6641dbb063193f2fb4e3c670f0ba5e971b7`。蓝焰主定位4/5，新增Ka23锅下蓝焰匹配、Cardoner水壶图仍无框；Bogtar有一个重叠局部框。55张派生难例仍41 TP/0 FP/0 FN，但旧ks_flame=50/6/7、工业灶火18/4/2，无火室内命名来源16 FP/13图，另外两组负例FP=0。该快照改善蓝焰且仍未通过旧场景召回与误检门槛，不能据蓝焰数晋升。

同一补充范围原标签13/3/2，紧火焰IoU≥.5为0/16/15、≥.3为5/11/10，平均最佳IoU=.255081。全部3页补充预测、5张蓝焰及13张无火来源误检图实际看完，共21个文件，身份记录SHA256 `e97222240ef962883a8741d88f3faed4a98c1fbc56a7d99ad2dab7a4bf19642c`；锅体范围保持并有重叠预测。无火误检包括7张重复桌面亮区/羽流图（9框，实际光源性质待核）、2张水印、人/羽流及手机线缆（2框）、日落、工业蓝色物体、烟雾区域。原标签、阈值和100轮日程保持；重复帧不代表独立失败场景。本地完成记录`fire_random_v14_epoch90_diagnostic_20261002/diagnostic_complete.json` SHA256 `fe7d5b19d00a2baefb573b0f00dc393814f654d05ae7a5075db549e609dcc702`，三次预定周期观察已完成，最终best/last正式对照仍待训练结束。

```powershell
python -B model_work/src/diagnose_fire_scratch_snapshots.py --run model_work/runs/fire-from-scratch-v14-20261002 --epoch 50 --review model_work/out/legacy_visible_flame_audit_20261002 --out model_work/out/fire_random_v14_epoch50_diagnostic_20261002
```

```powershell
python -B model_work/src/prepare_random_fire_start.py --reference model_work/runs/fire-binary-user-video-v3/weights/best.pt --expected-reference-sha256 48f284f9094fe334e02c66647f95fed8bfb3396b09f4488753508f514ed38980 --out model_work/out/fire_random_v14_initial_20261002
python -B model_work/src/train_fire_from_scratch_v14.py --data D:/SRTP_Datasets/fire_teacher_v13_20261002/data.yaml --expected-manifest-sha256 56496195e463bd11f1071bb7299dab8198175911403016d11ef79bc03c61ccc7 --initial model_work/out/fire_random_v14_initial_20261002 --expected-initialization-sha256 defc84bda87b71574b3b0282f35930de19a9958b40da151407d5b88dacb8b3f7 --out model_work/runs/fire-from-scratch-v14-20261002
```

本机已有初始目录、preflight和完成的run，不能重复执行这两条启动命令覆盖产物。初始化、源数据、run和详细日志只留本地，Git仅收源码和记录。
