# v16：从v14特征继续做可见火焰适配

2026-10-03，承接 [v14/v15项目比较](PROJECT_MODEL_ROUTES_20261002.md)。v14实际锅底小蓝焰视频表现较好，但旧来源误检多、锅体大框仍在。v15有限修正数据能增加无遮挡炉头紧框，却未解决锅体框。因此另立v16：从v14 last继续学习全网络，使用相同的已审核语义派生集。它是随机起点路线的后续微调，不是新一次从零训练，也不与v15构成单因素因果消融。

## 固定方案与完成状态

- 起点v14 last SHA256：`a46690756e65826060392688e98ceb268c9ecfa3f007e5796fa3c38ef4b5833e`。
- 数据manifest：`f3237c9ae34804918968a34945d51886266313aed83b7938e012586e11471069`，1407张/1890次采样，原val243张保持；没有重划独立测试。
- 固定12轮、patience=0、640、batch8/nbs8、AdamW lr=.0001/lrf=.1、warmup0、mosaic0、seed20261005。9428163个可学习参数，只有16个固定DFL参数不更新。
- 不使用教师约束，不冻结特征或BN。数据中的旧教师掩码仅作为沿革保留。
- 仅best/last参与最终比较，不挑中间快照，不按中间分数追加轮数。旧val含锅体框，不能仅按它选择几何候选。

正式训练已完成12轮，2844 batch/2837次实际优化更新。每轮全网络更新审计、实际起点逐张量检查及完成文件已写入。best SHA256 `e770cf68d12cd43b4494a1439dc9c65f14d1d5fc570b2e2e379cc084384418fd`；last SHA256 `3381a759d77ea390c11ae5034cfe14f122b62326938644f72d71df126a3de8f1`。训练完成不等于模型改善，图片与视频结果后续追加。

## 完成比较与项目取舍

统一图片比较、9段视频307共享采样/1535次推理均完成。实际查看两份候选的全部6页补充物理框、9页视频比较（27个源画面）。与上次相同采样逐项核对，307帧像素摘要相同，v13/v14/v15三份参考逐框预测也全部精确相同。完整 [公开结果](../../docs/fire_scratch_semantic_v16_results_20261003.json) 绑定原始评测、训练和实际查看记录。

| 模型 | 蓝焰主定位/5 | 物理框IoU≥.5 TP/FP/FN | 物理框平均最佳IoU | 旧mAP50 | 旧无火来源FP |
|---|---:|---|---:|---:|---:|
| v14 last | 4 | 0/15/15 | .250246 | .787650 | 18 |
| v15 last | 3 | 5/18/10 | .406334 | .886195 | 0 |
| v16 best | 4 | 6/11/9 | .384912 | .682962 | 17 |
| v16 last | 3 | 10/9/5 | .591463 | .376500 | 18 |

best的补充框F1=.375，last=.588235，均高于v14的0。实际查看确认best在第2/5张锅底及无遮挡炉头有更紧框，last还在第1/3/6张锅底新增紧框，但大锅体框继续存在，有时与小框并存；不是完全修正。旧ks_flame best仍46 TP、last降至34 TP；工业灶火best14 TP、last9 TP，旧综合micro F1分别.785992/.646154。几何标签差异不能解释所有退化，原始计数保持公开。

两份v16对原无火视频32个采样均0框，视频2–5的122采样均有火焰框；视频6为21/23（v14 19/23），视频7为25/27（v14 24/27），视频8为43/43，视频9为60/60（v14 59/60）。这些仍是有框计数，不是准确率。复核视频7–9起/中/末画面确认v16框住可见锅底蓝焰；视频6的起始火焰可见性仍未判真值。视频6/9里best总框22/64，last24/70；视频7里best37框，last34框，额外框代价因场景不同，不能说best所有视频重复框更少。

按项目用途优先保存 **v16 best作为下一电脑端开发候选**：保留4/5蓝焰，延续锅底视频优势，物理框有部分改善，相对last在部分片段额外框较少。接受其旧AP下降进入有界开发，不要求所有旧指标改善。last保留为更强物理定位的研究候选；v14/v13回退均保留。17个旧无火来源误检、锅体大框和火焰碎框仍需解决，没有宣称通用可靠性或独立验收。

best动态ONNX已通过80/80图预处理、raw和NMS一致性，最大坐标误差.000839输入像素、概率误差2.3842e-6，未放宽容差。独立CPU视频入口完整解码734帧，27采样25帧有框/37框；与GPU计数相同不代表逐框GPU数值相同。平均ONNX前向61.45ms仅是本机该次测量。独立图片叠加图实际查看，锅底火焰被框住，食物及贴纸未画框。见 [运行记录](../../docs/fire_scratch_semantic_v16_pc_runtime_20261003.json)。动态ONNX SHA256 `6d3d2ae52679988dd3f48a2d6932fa987ad2bd7b713168a638997cc86d8d683a`，不是板端静态或量化模型。

142项本机全依赖测试通过；源代码CI缺少模型依赖时相关测试跳过，不能等价于本机训练验收。下一阶段优先扩大训练图中物理框一致性审核，并检查无火/羽流错误，而不是只追加相同数据的轮数。旧smoke头依赖另一套固定特征，不直接拼到v16上。

## 预演、失败与修复

第一次8图CPU预演完成一步真实更新，126个feature/42个head权重张量变化，全部梯度有限，权重丢弃。但它没有覆盖正式入口重建模型的行为。第一次GPU启动使用`pretrained=False`，本机Ultralytics 8.4.70的`Model.train`因此把传给训练器的weights设为None。实际起点检查发现随机重建，**在首个batch前抛错，无训练更新、无候选权重**。

失败目录`model_work/runs/fire-scratch-semantic-v16-20261003`保留preflight、日志、failed.json及原始launch_source.py。原源码SHA256 `1b7ee52d60e95960d4c386d364b4c09e2ba2e9374e5a5efca6b360db3c6c38fa`；失败文件SHA256 `3c036c055c7e75ed14235faf0a615f7b1edfbfaf89f788f2785dc58d5ca9bf29`。

修复为显式指定本地v14 checkpoint，并在CPU预演前调用真实`DetectionTrainer.get_model`重建后比较499个状态张量，全部相等。二次8图预演再次完成一步，框/分类/DFL loss为2.600619/2.719205/2.469203，126个feature/42个head权重变化，没有保存候选。预演完成SHA256 `ae06bdc37ed92ad28de44261ba3b96b2f87a002e4367ec21018c11c0b1527bc0`。

修复后源码SHA256 `40d353034bf88d458b70d1a589dc91966236203fcaf0233c9bf772f655c0cdf8`。新正式目录`model_work/runs/fire-scratch-semantic-v16-r2-20261003`通过GPU实际起点检查，状态摘要`f9982a84366f5d80b2bf508bba863f4795a648d49c06335f60f345286d2b633e`与预演重建相同。失败目录未覆盖。

准备后续评测工具与文档时，自动审批服务曾因用量限制无法完成审核，该次文件写入未执行。用户要求继续后恢复工具访问，先确认原训练已完成，再补评测工具，没有重启训练。补充5项测试覆盖未完成轮数、假预演候选、实际更新数、缺少特征学习及显式本地权重参数，全部通过。

## 复现与评价

```powershell
python -B model_work/src/train_scratch_semantic_v16.py --start model_work/runs/fire-from-scratch-v14-20261002/weights/last.pt --data D:/SRTP_Datasets/fire_semantic_pilot_v2_20261002/data.yaml --video-comparison model_work/out/project_five_model_videos_20261002/comparison_complete.json --semantic-comparison model_work/out/fire_semantic_v15_comparison_20261002/comparison_complete.json --out <new-local-run>
python -B model_work/src/screen_scratch_semantic_v16.py --run model_work/runs/fire-scratch-semantic-v16-r2-20261003 --data D:/SRTP_Datasets/fire_semantic_pilot_v2_20261002 --review model_work/out/legacy_visible_flame_audit_20261002 --out <new-local-comparison>
```

预演另加`--rehearse-cpu`并使用新目录。训练和比较都拒绝覆盖已有产物。统一比较v13、v14 last、v15 last、v16 best/last；保留旧图、物理火焰框、无火来源和同帧视频的收益与代价，不以旧AP自动否决。全部素材仍是曝光开发诊断。板子未连接，v13回退、v14运行候选与smoke组合均保留。
