# v20：不同来源扩展的完整结果与采用取舍

2026-10-03完成。**演示保留v16 best，定位开发保留v18 last；v20 last保留作局部收益与退化的研究参照，本轮不晋升。** 两份候选均完成全部固定图片和视频比较，没有按旧AP提前筛掉。

## 训练与证据

1445唯一训练图/2409采样条目，原243验证图保持；新增14开发图/13暂定来源组，其中4火焰区域、10无可见火焰图。4预留图和3隔离图全部排除，未预测、未训练。新增采样不等于独立厨房覆盖。数据manifest SHA `15a509075be0d2ff062894cd7de1160699bbdc7b657d8199bfba77cb80bf36fa`。

从70705ce启动的正式进程已exit0：12轮/3624 batch/3618次实际更新。499状态张量精确初始化、每轮42个头权重张量改变、23特征模块及BN保持、EMA恢复、保存best/last特征保持均通过。16份实际源码快照与冻结哈希一致，训练源码和旧依赖没有在比较阶段修改。

best SHA `a8186bc227207e9daedb4fe7a59f8e967c9acc225fc1b205ccb2b0bbe1ebe24c`；last SHA `37d18ca746cf83fb9e19fd07c0faaaaa6e04f96916f9864b12d415a3acdf55c0`。一次后续命令曾因审批服务额度无法完成检查而未执行；正式训练继续运行，本轮重新观察旧句柄并确认正常结束，没有因旧启动记录重启训练。

## 图片收益与代价

旧286、蓝焰5、物理火焰15框、旧55派生图保持原身份；全部为已曝光开发诊断。新14张训练图另计。

| 模型 | 蓝焰主定位/5 | 紧框TP/FP/FN（15框） | 紧框F1 | 平均最佳IoU | 旧mAP50 | 旧ks_flame TP/57 | 旧空标签100图预测框 |
|---|---:|---|---:|---:|---:|---:|---:|
| v16_best | 4 | 6/11/9 | 0.3750 | 0.3849 | 0.6830 | 46 | 17 |
| v18_last | 4 | 10/5/5 | 0.6667 | 0.5237 | 0.5476 | 43 | 18 |
| v19_last | 4 | 11/3/4 | 0.7586 | 0.5603 | 0.4615 | 39 | 25 |
| v20_best | 4 | 7/12/8 | 0.4118 | 0.4341 | 0.5870 | 42 | 18 |
| v20_last | 4 | 10/3/5 | 0.7143 | 0.5038 | 0.4161 | 36 | 22 |

v20 last较v18 last保持10个紧框匹配、补充FP5→3，但平均最佳IoU .5237→.5038；旧ks_flame TP43→36、286+55 micro F1 .70635→.63780。相对v19 last则少一个匹配。best仍有锅体大框和重叠框，last仍漏索引3和7的左锅底，索引4/8/10仍框入锅体；开放炉头的三处匹配不应全部称作锅底火焰。

新14图的4个火焰区域，五模型均0个IoU≥.5匹配。这不等于四张都无火焰预测：context3_01完整商业厨房是真漏检；context4_04覆盖下方蓝焰但缺上侧范围；context4_05蓝焰预测偏宽，v20 last IoU=.495715，固定门槛不改；context4_11只框橙色尖部，大部分蓝色炉头仍漏。各模型逐图IoU已记录，不能把几何失配都当非火误检。

新10张空标签负例：v16/v18/v20 best各4张有错框，v19 last/v20 last各3张。v20 last消除了context4_03食物高光框，但餐具/红托盘、橙色食物、灶台标牌错误仍在。混合14图的last FP=6含3处正例定位失配，不能说6张负例误报。

旧空标签来源best18框/15图、last22框/19图已逐页复核。水印、日落、纸巾、建筑及彩色对象存在明确非火对象错误；羽流/强光场景的燃烧语义保持未确认。22框不是22次报警或22%真实误报警率。

## 视频复核与项目采用

9段视频307共同采样、5模型1535次推理；307条像素/帧/时间/尺寸与v19一致，三个旧参考921条输出精确复现。实际查看9页固定首/中/末、全部13个计数差异样本5页，另补同计数video9样本6/37/39两页。新14图4页、4正例放大页、两份物理框6页、空标签9页、CPU预览1页及图片1份，合计41文件SHA保留。

- 视频6：两份v20均21/23有框、21框，与v18/v19 last相同；固定末帧保留蓝焰。样本3少碎片框，不能从总框数直接计算正确率。
- 视频7：best/last均25/27有框，共35/36框。样本0两份恢复v18/v19 last漏掉的中间蓝焰，右分量仍漏；样本1两份仍只左分量，漏掉v18 last可见的中间分量。样本12两份保留左/中，last样本15恢复v18/v19 last漏掉的右分量，best未恢复。存在局部收益，连续覆盖尚未解决。
- 视频9：best/last均60/60有框，共64/69框（v18 last70、v19 last71）。在明确复核子集中last误框挂件33/34/35/36/40/44/45，较v18/v19 last减少38；44消除v19 last第二个挂件框，但一个误框仍在，45仍两个嵌套挂件误框。best错40/44/45。样本6所有模型的两框均覆盖真实炉头区域，37/39无挂件错误；同框数仍需实际查看。

决定以项目用途作综合取舍：v20 last的两处蓝焰分量恢复和一处挂件减少有研究价值，但同场景中间蓝焰漏检、其余挂件错误和新来源定位未解决，尚不能证明稳定的整体项目收益。旧AP下降作为代价记录，没有作为单独否决条件。保留当前候选与所有v20权重。

## 电脑端运行与限制

best/last动态矩形FP32 ONNX各94/94原始输出及最终框一致性通过（旧80加新14开发图），固定.1像素/.0001概率容差未改。last独立CPU图片与视频实跑完成，原视频734帧完整解码、采样27帧、25有框/36框；CPU PyTorch→独立入口27/27通过。生成预览重新解码27帧，首/中/末和图片实际查看。

GPU→CPU严格数值只有12/27通过，框数保持，最大坐标差.0205078、概率差.00133318超过.0001；保留不通过事实，未放宽阈值。平均ONNX forward57.33ms仅本次电脑调用，不含解码/预处理/NMS/画框，也不是受控旧版速度比较、摄像头或端到端报警P95。没有KModel、量化、新板端或长时事件验收。

收尾核验：182项完整本机测试通过、无跳过；484文件公开仓库检查与暂存差异检查通过。旧准确提交4ad38be的push/PR实际任务和各步骤均成功。本轮只提交源码、JSON小汇总和记录，原图/权重/视频保留本地。

## 下一阶段

新来源下头部更新的适应收益有限；现有证据不足以归因于冻结特征、样本份额或标注范围中的某一项。下一步先做训练曝光正例的可学习性诊断，明确全画面小目标与检测头冻结的限制，再选择特征适配路线；继续补按厨房来源划分的完整场景，四张预留仍不参与调参。烟雾/蒸汽与正常灶火事件边界另行推进。

[结构化结果](../../docs/fire_kitchen_source_v20_results_20261003.json)、[电脑运行记录](../../docs/fire_kitchen_source_v20_pc_runtime_20261003.json)、[原训练冻结方案](KITCHEN_SOURCE_HEAD_V20_20261003.md)、[未达项目要求](../../docs/PLAN_REQUIREMENT_MATRIX_20261002.md)。

## 复现命令

沿用已记录Python环境；原图/权重只读，输出存在时拒绝覆盖。所有留出/隔离身份继续排除。

```powershell
python -B model_work/src/screen_kitchen_source_v20.py --run model_work/runs/fire-kitchen-source-head-v20-20261003 --data D:/SRTP_Datasets/fire_kitchen_source_v20_20261003 --review model_work/out/legacy_visible_flame_audit_20261002 --previous model_work/out/train_context_v19_comparison_20261003 --out model_work/out/kitchen_source_v20_comparison_20261003
python -B model_work/src/eval_project_model_videos.py --models model_work/out/kitchen_source_v20_comparison_20261003/models.json --videos-dir viedos viedos/视频/视频 --out model_work/out/project_v20_videos_20261003
python -B model_work/src/prepare_kitchen_source_v20_review.py --images model_work/out/kitchen_source_v20_comparison_20261003 --videos model_work/out/project_v20_videos_20261003 --previous-videos model_work/out/project_v19_videos_20261003 --data D:/SRTP_Datasets/fire_kitchen_source_v20_20261003 --out model_work/out/v20_visual_review_20261003
python -B model_work/src/prepare_video_count_difference_review.py --comparison model_work/out/project_v20_videos_20261003 --out model_work/out/v20_all_video_count_differences_20261003
python -B model_work/src/prepare_nofire_error_review.py --cache model_work/out/kitchen_source_v20_comparison_20261003/legacy_sources --model v20_best --out model_work/out/v20_best_nofire_error_review_20261003
python -B model_work/src/verify_onnx_export.py --weights model_work/runs/fire-kitchen-source-head-v20-20261003/weights/best.pt --expected-sha256 a8186bc227207e9daedb4fe7a59f8e967c9acc225fc1b205ccb2b0bbe1ebe24c --data D:/SRTP_Datasets/fire_kitchen_source_v20_20261003 --dynamic-rectangular --out model_work/out/kitchen_source_v20_best_dynamic_onnx_20261003
python -B model_work/src/prepare_nofire_error_review.py --cache model_work/out/kitchen_source_v20_comparison_20261003/legacy_sources --model v20_last --out model_work/out/v20_last_nofire_error_review_20261003
python -B model_work/src/verify_onnx_export.py --weights model_work/runs/fire-kitchen-source-head-v20-20261003/weights/last.pt --expected-sha256 37d18ca746cf83fb9e19fd07c0faaaaa6e04f96916f9864b12d415a3acdf55c0 --data D:/SRTP_Datasets/fire_kitchen_source_v20_20261003 --dynamic-rectangular --out model_work/out/kitchen_source_v20_last_dynamic_onnx_20261003
python -B model_work/src/pc_onnx_video.py --model model_work/out/kitchen_source_v20_last_dynamic_onnx_20261003/model.onnx --expected-sha256 bb35f050b56b5f670414fa84ae9cc6960790d0dc5bd34aedb6a891f5b59c58ff --video viedos/视频/视频/82f51e33ff7ddbdcda9ea4c6754b9983.mp4 --rectangular --boxed --out model_work/out/kitchen_source_v20_last_cpu_video_20261003
python -B model_work/src/verify_pc_video_cpu_parity.py --comparison model_work/out/project_v20_videos_20261003 --runtime model_work/out/kitchen_source_v20_last_cpu_video_20261003 --weights model_work/runs/fire-kitchen-source-head-v20-20261003/weights/last.pt --out model_work/out/kitchen_source_v20_last_cpu_video_20261003/cli_prediction_parity.json
python -B model_work/src/pc_onnx_infer.py --model model_work/out/kitchen_source_v20_last_dynamic_onnx_20261003/model.onnx --expected-sha256 bb35f050b56b5f670414fa84ae9cc6960790d0dc5bd34aedb6a891f5b59c58ff --image model_work/out/project_v20_videos_20261003/source_review_frames/video_07_sample_0013.png --rectangular --out model_work/out/kitchen_source_v20_last_cpu_image_20261003
python -B model_work/src/prepare_kitchen_source_v20_detail.py --data D:/SRTP_Datasets/fire_kitchen_source_v20_20261003 --images model_work/out/kitchen_source_v20_comparison_20261003 --runtime model_work/out/kitchen_source_v20_last_cpu_video_20261003 --out model_work/out/v20_detail_review_20261003
python -B model_work/src/prepare_video_sample_comparison_review.py --comparison model_work/out/project_v20_videos_20261003 --video-index 9 --samples 37 39 --out model_work/out/v20_same_count_review_20261003
python -B model_work/src/prepare_video_sample_comparison_review.py --comparison model_work/out/project_v20_videos_20261003 --video-index 9 --samples 6 --out model_work/out/v20_same_count_sample6_review_20261003
```

生成页面只准备复核输入，不自动证明已看过；结构化结果里的41份文件是本轮实际查看后的独立人工观察记录。当前训练/比较/CPU入口均已正常结束，无仍在运行的v20任务。持续目标未完成。
