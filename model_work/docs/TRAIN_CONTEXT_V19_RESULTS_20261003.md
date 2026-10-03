# v19：训练全场景补充、七模型比较与采用取舍

2026-10-03完成。**演示继续保留v16 best，定位开发候选继续保留v18 last；v19 last作为紧框定位取舍参照，v19 best作为较少挂件错误的对照，不晋升。** 判断依据是完整画面中的蓝焰分量漏检和厨具错误，而非旧AP单项否决。

## 数据和训练完成证据

[冻结方案与逐图审核](TRAIN_CONTEXT_V19_20261003.md)先于结果提交；40图/21页审核后采用15图27个物理火焰框、14张无可见火焰完整图。新增7张工业厨房原图与现有场景相关，8张ks_flame重标；10张冗余单锅图保持原权重，1张模糊锅底图仍隔离。没有把本次项目视频或旧测试图放进训练。

1431唯一图/2341采样条目，原243验证图身份及非获批图像/标签/权重保持。manifest SHA `9971c9530a04f1dd962a7c2d9e229d73d5b7e1a47aab5d3d966975d6e78d831a`。提高到8的采样是实验设定，不证明数据覆盖或效果；实际挂件难例并未由这些训练图覆盖。

正式训练与v18仅data和run名称改变，同v16 best起点、12轮、seed20261006、AdamW lr0=.00005、batch/nbs8、imgsz640、warmup0/mosaic0，前23特征模块及BN缓冲固定、819779个检测头参数学习，DFL固定，无教师损失。12轮/3516 batch/3510次实际更新完成；499状态初始化、逐轮头更新/冻结/EMA、保存两份权重的特征核对均通过。实际源码15份逐项与训练前冻结字节相同，复制到run/source_snapshot，保留CRLF与Git LF哈希。

best SHA `d06f584f545c2f71a50caafa505d3bc1ca82a52ca97adbf6809f9b44064ed1c0`；last SHA `9a13523c8f66f8a9b60e8bcafda9b1cdc270cf71e0269b4019708b7beabd80af`。本机预演阶段158项测试已通过，完整训练/比较审计另有实际结果，不把代码测试当识别准确率。方案1669fbd、数据预演3840b5d、七模型方案与初始化1cffb7d均在比较结果前推送且push/PR CI通过。

## 图片收益和代价

七模型的固定来源计数和补充物理15框实际重算；新两份重跑蓝焰5、旧286、55派生图。旧五份的AP/蓝焰/55图缓存经过权重、数据成员/标签、评测源码和政策身份验证。全部为曝光开发诊断。

| 模型 | 蓝焰主定位/5 | 物理紧框TP/FP/FN（15框） | 紧框F1 | 平均最佳IoU | 旧mAP50 | ks_flame TP/57 | 原空标签100图预测框 |
|---|---:|---|---:|---:|---:|---:|---:|
| v16 best | 4 | 6/11/9 | .3750 | .3849 | .6830 | 46 | 17 |
| v17 best | 3 | 11/8/4 | .6471 | .5873 | .6093 | 46 | 19 |
| v17 last | 4 | 12/7/3 | .7059 | .6425 | .4190 | 35 | 15 |
| v18 best | 4 | 7/13/8 | .4000 | .4624 | .6016 | 46 | 18 |
| v18 last | 4 | 10/5/5 | .6667 | .5237 | .5476 | 43 | 18 |
| v19 best | 4 | 6/13/9 | .3529 | .4351 | .5798 | 43 | 20 |
| v19 last | 4 | 11/3/4 | .7586 | .5603 | .4615 | 39 | 25 |

last较v18 last多1个紧框匹配、少2个补充FP，物理F1提高；旧AP、固定来源TP及286+55图micro F1退化（.70635→.64865），并非全面提高。实际查看两份各3页：last索引3较v18收紧，1/2/5/6保持物理框；7左锅底仍漏，4/8/10仍有锅体大框。11个匹配包含14的3个开放炉头，不能称作11个锅底火焰。

原空标签来源增加的是25框/20张有框图，不是25次报警或25%真实误报率。全部20图5页另行实际查看，新增有框图身份为复核索引3/4/14/15/18、未减少原v18有框图。水印、日落、纸巾、建筑和云区有明确非火对象；烟状羽流、明亮灯光及相关房间重复图的事件语义另留边界。另看索引1/20原尺寸，没有据此改标签、判断无燃烧或建立独立测试。

## 视频实际复核与采用决定

9视频307个共同采样，七模型2149次推理。与v18的307条视频/帧/时间/尺寸/像素身份相同，五份旧参考1535条预测精确复现。实际查看9页固定首/中/末三模型页（v16/v19 best/last，27个原画面）、七模型全部18个计数差异样本的6页可读四模型页（v16/v18 last/v19 best/last），加6页物理框和5页空标签来源，共26页。选择行/列和原始七模型页SHA均保留，旧参考逐帧输出仍完整；计数差异只安排审核，不能计算准确率。

- 视频6：v19两份均21/23有框、21框，与v18相同；样本3没有v16的碎片框，22保留可见蓝焰。没有新增密集真值。
- 视频7：v19 last为25/27有框、33框，v18 last为25/35。样本0两份last均只覆盖左分量（v19横坐标135.56–243.89，v18为134.25–244.81），中/右分量仍遗漏；样本1仅左分量而v18还覆盖中间可见火焰，12漏掉左分量。**少框不能归为重复框改善。** 本次逐分量复核补充了此前粗略框数/重叠描述，旧记录不追改。
- 视频9：v19 best/last均60/60有框，共64/71框，v18 last70框。差异集合里last误框挂件33/34/35/36/38/40/44/45，与v18 last的八张错误身份相同；44还新增第二个挂件框，45仍两个嵌套误框。按此前8个固定样本33/35/36/37/38/39/40/45，best错40/45，last错33/35/36/38/40/45。best有局部挂件收益，但紧框表现不支持整体替换。

因此保留当前演示与开发候选。v19 last有局部几何价值，但真实厨房蓝焰分量损失及挂件无改善，尚不足晋升。下一步应补真正不同厨房的挂件/反光完整场景和遮挡火焰，按场景划分训练/留出；不再仅凭同一批原图重加权、延长训练或调阈值。烟雾/事件口径仍另行推进，不能把fire检测直接当危险报警。

## 电脑端可运行性与边界

best/last动态矩形FP32 ONNX各80/80原始输出和最终框一致性通过，固定坐标.1像素/概率.0001容差不变。last独立CPU图片实跑，视频实际解码734帧/采样27帧、25有框/33框；CPU PyTorch→独立ONNX入口27/27通过。GPU→CPU严格数值只13/27，框数一致，最大坐标差.02209、概率差.00103068超过.0001，保留不通过事实。

boxed视频重新解码确认27帧，图片与首/中/末三帧实际查看。平均ONNX forward67.36ms是在GPU同帧比较同时进行时测得，不能与v18做受控速度比较，也不包含解码/预处理/NMS/画框。没有板端、KModel、量化、实时摄像头、端到端P95或长期误报警验收。

辅助差异页首次默认GBK读取含中文的plan失败，在创建输出前结束；改为显式UTF-8后成功，没有改训练/评测数据或模型。训练句柄61850已exit0结束，不能再用旧状态文件声称在训。

[训练/图片/视频结果](../../docs/fire_train_context_v19_results_20261003.json)、[ONNX和CPU结果](../../docs/fire_train_context_v19_pc_runtime_20261003.json)。大文件仍在本地，Git保存源码、小汇总和经过，持续目标仍未达到[独立厨房与事件要求](../../docs/PLAN_REQUIREMENT_MATRIX_20261002.md)。

## 复现

使用已记录本机Python/依赖；输入均只读，输出存在时拒绝覆盖，复现应换新输出目录。训练/数据命令见冻结方案。实际比较、导出、运行命令如下：

```powershell
python -B model_work/src/screen_train_context_v19.py --run model_work/runs/fire-train-context-head-v19-20261003 --data D:/SRTP_Datasets/fire_train_context_v19_20261003 --review model_work/out/legacy_visible_flame_audit_20261002 --previous model_work/out/frozen_flame_v18_comparison_20261003 --out model_work/out/train_context_v19_comparison_20261003
python -B model_work/src/eval_project_model_videos.py --models model_work/out/train_context_v19_comparison_20261003/models.json --videos-dir viedos viedos/视频/视频 --out model_work/out/project_v19_videos_20261003
python -B model_work/src/prepare_video_count_difference_review.py --comparison model_work/out/project_v19_videos_20261003 --out model_work/out/v19_all_video_count_differences_20261003
python -B model_work/src/prepare_nofire_error_review.py --cache model_work/out/train_context_v19_comparison_20261003/legacy_sources --model v19_last --out model_work/out/v19_nofire_error_review_20261003
python -B model_work/src/verify_onnx_export.py --weights model_work/runs/fire-train-context-head-v19-20261003/weights/best.pt --expected-sha256 d06f584f545c2f71a50caafa505d3bc1ca82a52ca97adbf6809f9b44064ed1c0 --data D:/SRTP_Datasets/fire_train_context_v19_20261003 --dynamic-rectangular --out model_work/out/train_context_v19_best_dynamic_onnx_20261003
python -B model_work/src/verify_onnx_export.py --weights model_work/runs/fire-train-context-head-v19-20261003/weights/last.pt --expected-sha256 9a13523c8f66f8a9b60e8bcafda9b1cdc270cf71e0269b4019708b7beabd80af --data D:/SRTP_Datasets/fire_train_context_v19_20261003 --dynamic-rectangular --out model_work/out/train_context_v19_last_dynamic_onnx_20261003
python -B model_work/src/pc_onnx_infer.py --model model_work/out/train_context_v19_last_dynamic_onnx_20261003/model.onnx --expected-sha256 062ea2aba78a0d82d5126627076bb6f955e5c54ea8676ebd00cdbd0b49720b75 --image model_work/out/project_v16_videos_20261003/source_review_frames/video_07_sample_0013.png --rectangular --out model_work/out/train_context_v19_last_cpu_image_20261003
python -B model_work/src/pc_onnx_video.py --model model_work/out/train_context_v19_last_dynamic_onnx_20261003/model.onnx --expected-sha256 062ea2aba78a0d82d5126627076bb6f955e5c54ea8676ebd00cdbd0b49720b75 --video viedos/视频/视频/82f51e33ff7ddbdcda9ea4c6754b9983.mp4 --rectangular --boxed --out model_work/out/train_context_v19_last_cpu_video_20261003
python -B model_work/src/verify_pc_video_cpu_parity.py --comparison model_work/out/project_v19_videos_20261003 --runtime model_work/out/train_context_v19_last_cpu_video_20261003 --weights model_work/runs/fire-train-context-head-v19-20261003/weights/last.pt --out model_work/out/train_context_v19_last_cpu_video_20261003/cli_prediction_parity.json
```

可读派生页仅保留固定七行原页的第1/6/7行、差异页七列的第1/5/6/7列，每行460像素/每列640像素；来源和输出页SHA在本地review_complete.json中。推理政策、原图、模型和标签未由页排列改变。逐项实际查看后才写复核记录，不能通过运行命令自动宣布已看过。
