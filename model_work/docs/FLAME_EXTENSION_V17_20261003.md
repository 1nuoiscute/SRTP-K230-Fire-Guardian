# v17：扩展可见火焰框与复核无火采样

2026-10-03，承接v16。目标是改善锅底真实火焰定位、降低可见无火/羽流误检，保留锅底蓝焰视频优势。板子没有连接。按项目整体收益与代价判断，不以旧AP全面超过作为采用条件。

## 训练前固定方案

从v16 best继续全网络微调，起点SHA256 `e770cf68d12cd43b4494a1439dc9c65f14d1d5fc570b2e2e379cc084384418fd`。固定12轮、patience0、640、batch8/nbs8、AdamW lr0=.00005/lrf=.1、warmup0、mosaic0、seed20261006，只有DFL固定参数不学习。仅best/last进入比较，不按中途指标追加轮数或挑快照。相对v16同时改变数据、起点、学习率和种子，不能解释为单因素消融。

审核原训练图48张：按原火焰框数量1..4各抽8图，排除上一批32图及已保留修订；另从两类空标签来源各取8图。选择按来源组排序的中点分位，未看模型预测。全部12页原图和12页最终标注页实际查看。批准17图/44个近似可见火焰框、15张可见无火图增加采样权重；15张遮挡/亮点火焰图仍隔离，1张暗光羽流图不增加权重，其继承空标签/权重1保持但未确认。后者不是新批准负例，也没有被描述为已隔离。

新数据 `D:/SRTP_Datasets/fire_semantic_extension_v17_20261003`：1424独立文件、2131采样条目，新增火焰图权重8，15张已审核负例从1到8；保留父集1407图全部标签和其他采样权重，1008张工业来源仍隔离，原val243图身份/标签不变。新manifest SHA256 `8e70dea69f39beafa0f17b3a6d4b483c537622fdf52271755fd8f3e2447bd54d`。全部来源已曝光，属于相同工业厨房及旧来源的开发改进，没有独立厨房或盲测。

来源名nofire_real_indoor包含室外、羽流和隐蔽燃烧装置，不能按名称判断正常事件。负例仅指整图没有可见火焰；smoke缺席继续未知。原始图片/标签未改动。审核文件、原图和具体预测留本地，公开记录只保存汇总与工具。

## 检查与修正

首次准备因无火图原始尺寸不统一而报错，未形成批准或训练；失败目录保留。修复为按原尺寸校验并等比例展示，在r2目录继续。首版叠加页发现第7/8图候选框纵向位置错误，在批准前修正；首版候选和全部页保留，最终12页逐项重新实际查看。模型不得使用首版框。

训练前须完成一次真实CPU梯度预演，含2张新增火焰、2张新增负例、2张既有修订、1张旧来源和1张派生火焰。预演临时模型丢弃，无性能结论。正式训练逐状态检查真实v16起点、全网络每轮变化、实际优化步数与有限梯度。

## 复现与后续比较

```powershell
python -B model_work/src/train_flame_extension_v17.py --start model_work/runs/fire-scratch-semantic-v16-r2-20261003/weights/best.pt --data D:/SRTP_Datasets/fire_semantic_extension_v17_20261003/data.yaml --expected-data-sha256 8e70dea69f39beafa0f17b3a6d4b483c537622fdf52271755fd8f3e2447bd54d --rehearse-cpu --out model_work/out/fire_extension_v17_cpu_rehearsal_20261003
```

正式运行去掉rehearse-cpu，使用新run目录。统一比较v16 best/v17 best/last：旧286图、蓝焰5图、派生55图、补充10图15个物理火焰框、旧无火来源，以及9段视频307共享采样。保留原阈值与近似标注，所有best/last均复核；实际目视检查后才决定开发候选。没有自动替换、上板、两类smoke融合或独立验收。

本节保留训练前冻结方案；完成结果和采用判断见下文。

CPU预演已完成：8图真实一步更新，框/分类/DFL损失2.221290/1.961447/1.983217，全部可训练梯度有限；126个特征权重张量和42个头权重张量改变。实际起点499个状态张量与v16 best一致；预演权重丢弃。新增6项防错检查通过。


## 训练启动阶段及无火错误语义复核（历史记录）

正式GPU起点检查通过：9428163个可训练参数，实际499个状态与v16 best相同，未加载其他学习权重。12轮固定日程已启动；148项本机全依赖测试通过。阶段准备已提交并推送`2af62ee`，候选训练尚未完成，不能提前宣称性能改进。

另对v16全部15张旧无火来源误检图/17框实际查看全部4页。10图/12框明确落在羽流、物体、太阳、纸巾、水印或灯光上；5图/5框来自相似桌面亮物体，火焰与照明仍不确定。原空标签FP=17计数保持，不把不确定5框改判成正确火焰来提高分数，也不把所有nofire来源解释为正常事件。完整公开汇总见[错误口径复核](../../docs/nofire_error_semantics_v16_20261003.json)。这批测试图没有加入训练。

补充视频6全部23个1fps采样可见性审核：重新完整解码并逐项核对23帧像素，与既有比较全部相同；实际查看全部6页原画面和下半幅细节，采样0/1高度遮挡仍不确定，2..22共21帧可分辨锅底蓝焰，没有确认无火帧。此存在性标注在v17评测前冻结，没有框真值，不能将有框计数转换成定位准确率。原23采样分母完整保留，不把未知帧当负例。见[可见性记录](../../docs/video6_sample_visibility_20261003.json)。


## 完成结果与项目取舍

12轮正式训练完成：3204个batch、3198次实际优化器更新，真实v16起点、每轮全网络变化和梯度审计通过。只有事先声明的best/last进入比较，没有从中间轮挑选额外权重。结果见[公开完整汇总](../../docs/fire_flame_extension_v17_results_20261003.json)，权重、数据、源码和完成审计均以SHA256绑定。本轮不是单因素消融，不能将全部变化归因于新增17图。

| 同一阈值与输入 | v16 best | v17 best | v17 last |
|---|---:|---:|---:|
| 5图蓝焰主定位 IoU≥.5 | 4 | 3 | 4 |
| 补充10图15个近似物理火焰框 TP/FP/FN | 6/11/9 | 11/8/4 | 12/7/3 |
| 补充物理框F1 | .3750 | .6471 | .7059 |
| 物理框平均最佳IoU | .3849 | .5873 | .6425 |
| 旧286图mAP50 | .682962 | .609314 | .418980 |
| 旧ks_flame 57框 TP/FP/FN | 46/13/11 | 46/10/11 | 35/21/22 |
| 旧kitchen_stove_fire 20框 TP/FP/FN | 14/8/6 | 9/14/11 | 8/15/12 |
| 原空火标签nofire来源100图 FP框/有框图 | 17/15 | 19/16 | 15/14 |

原标签在相同补充子集上TP分别为10/7/7；补充可见火焰框匹配分别为6/11/12，说明几何口径变化影响旧AP。两种口径同时保留，不替换原指标。实际查看全部6页best/last物理框叠加：紧框改善，但锅体大框、嵌套框仍在，last第4/8/10张补充图仍没有匹配到紧火焰框。不能将12/15写成锅体误检已解决。

9段视频共307个共享采样，三模型921次推理。逐项核对视频SHA、帧号、时间、尺寸与解码像素SHA，全部与v16上轮一致，v16全部预测记录也精确复现。实际查看9页首/中/末对照（27个固定原画面），再查看所有13个预测数量不同的画面，共5页；后者是预测差异诊断，不是新的独立测试。全部结果与28页实际复核的哈希记录保存在本地 `model_work/out/project_v17_videos_20261003/review_complete.json`。

视频6保留原23采样：三模型都是21个有框采样，在事先标注的21个可见蓝焰采样上均有框，2个未知采样均无框；这只能说明存在性采样诊断，不能证明定位准确率。末帧锅底框从1→2→3，重叠更明显。视频7有框采样25→25→24、总框37→35→34；sample1实际可见蓝焰，last新增漏检，sample0/15减少组件框不能直接当召回提高。

视频9三模型均60个有框采样，总框64→69→72。完整画面复核发现best在sample33/35/36/37/38/40新增挂着的彩色厨具误检，last还在sample39误检；last的sample5也多出重叠蓝焰框。这个错误属于演示场景内代价，需要优先处理，不能因为锅底仍有一个正确框就忽略。首/中/末画面之外仍无全密度框真值，1fps采样无法验证短时事件和整段连续准确率。

另外实际查看v17 best全部16张原空标签误检图/19框、last全部14图/15框，共8页。best明确非火焰8图/10框、亮桌面物体不确定8图/9框；last明确非火焰7图/8框、不确定7图/7框。原FP计数17/19/15保持，未把不确定火焰当正确输出，也未修改测试标签。相似来源存在近重复，不能把图片数解释为独立场景数。烟雾是否存在另行未知。

**采用判断：当前电脑端演示继续保留v16 best；v17 last作为下一物理火焰定位开发候选，best保留对照，v14/v13回退保留。** last的真实定位收益支持继续这条路线；但视频内厨具误检、更多重叠框和新增可见蓝焰漏检，尚不支持取代当前演示模型。旧AP下降单独不构成否决，决定依据是项目收益与实际演示内代价。没有覆盖旧权重、自动默认替换或板端部署。

## 电脑端独立运行验证

best/last均导出动态矩形FP32 ONNX，在同一80图、同一预处理、原始输出和NMS容差下各80/80通过。last在独立CPU图片CLI和视频CLI实跑通过：视频完整解码734帧，27个采样、24个有框、34个框；CPU PyTorch对独立ONNX CLI的27/27采样数值一致性通过。实际查看图片叠加和盒框视频首/中/末3画面。电脑上观测单图前向约61ms，视频平均前向约59ms；不含完整处理时延，不能当K230速度或实时验收。

另行尝试将CPU CLI与之前GPU预测按相同严格数值容差比较，27采样框数相同，但严格数值只有15/27通过；最大坐标差.022034像素、置信度差.000524，超过.0001置信度容差。该未通过结果完整保留，没有放宽阈值。随后用CPU PyTorch复核27采样全部通过，确认CPU导出与独立入口的既定口径；GPU与CPU不能宣称数值完全一致。见[电脑运行及差异记录](../../docs/fire_flame_extension_v17_pc_runtime_20261003.json)。

```powershell
python -B model_work/src/train_flame_extension_v17.py --start model_work/runs/fire-scratch-semantic-v16-r2-20261003/weights/best.pt --data D:/SRTP_Datasets/fire_semantic_extension_v17_20261003/data.yaml --expected-data-sha256 8e70dea69f39beafa0f17b3a6d4b483c537622fdf52271755fd8f3e2447bd54d --out model_work/runs/fire-flame-extension-v17-20261003
python -B model_work/src/prepare_video_count_difference_review.py --comparison model_work/out/project_v17_videos_20261003 --out model_work/out/v17_all_video_count_differences_20261003
python -B model_work/src/pc_onnx_infer.py --model model_work/out/fire_extension_v17_last_dynamic_onnx_20261003/model.onnx --expected-sha256 2156d6522581404b9678be0b03887adc851ef34fadaa0cefa7c5fb43e9ec8134 --image model_work/out/project_v16_videos_20261003/source_review_frames/video_07_sample_0013.png --rectangular --out model_work/out/fire_extension_v17_last_cpu_image_20261003
python -B model_work/src/pc_onnx_video.py --model model_work/out/fire_extension_v17_last_dynamic_onnx_20261003/model.onnx --expected-sha256 2156d6522581404b9678be0b03887adc851ef34fadaa0cefa7c5fb43e9ec8134 --video viedos/视频/视频/82f51e33ff7ddbdcda9ea4c6754b9983.mp4 --rectangular --boxed --out model_work/out/fire_extension_v17_last_cpu_video_20261003
python -B model_work/src/verify_pc_video_cpu_parity.py --comparison model_work/out/project_v17_videos_20261003 --runtime model_work/out/fire_extension_v17_last_cpu_video_20261003 --weights model_work/runs/fire-flame-extension-v17-20261003/weights/last.pt --out model_work/out/fire_extension_v17_last_cpu_video_20261003/cli_prediction_parity.json
```

以上为实际命令，输出目录已存在时拒绝覆盖；复现应使用新输出目录，并保留原始证据。数据、权重和逐图逐帧结果本地保留，公开仓库仅放源码、汇总与哈希。

## 下一阶段工作依据

优先从原train范围审核有彩色厨具、反光、遮挡和小火焰的完整场景；有可见火焰的全帧必须保留火焰正标签，不能为了压制厨具而变成空标签。保持已曝光test/视频不进入训练，补一致物理框，解决大锅体框和重叠输出后再做同帧项目比较。v17新增17图仍少，继续堆叠训练轮数没有本轮证据支持。另需厨房/事件隔离的新数据，才能验证跨场景和事件可靠性；两类烟雾、融合、KModel与板端验收仍未完成。


完成收尾：148项本机全依赖测试通过。本机实际源码字节与Git LF规范化字节可能不同，已将22份训练、评测、运行工具和采用规则逐份按原哈希复制到run内source_snapshot，保留清单；严格复核原实验使用该本地快照。数据和权重仍留本地。
