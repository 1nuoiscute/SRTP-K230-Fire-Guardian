# v18：保留v16特征，适配物理火焰检测头

2026-10-03完成。**v18 last作为下一轮定位开发候选，当前演示保留v16 best；v17 last保留为更强几何参照，v18 best保留为较少挂件误检的对照。** 综合锅底定位、连续性、场景误检、旧召回和电脑端可运行性决定用途，不要求所有旧指标全面改善。仍有锅底漏检、锅体大框和彩色挂件误检，尚无独立厨房或板端验收。

## 固定方案和真实训练

承接已完成的v17（a0559ea）。固定起点v16 best：`e770cf68d12cd43b4494a1439dc9c65f14d1d5fc570b2e2e379cc084384418fd`。完全复用v17数据1424张唯一图/2131个采样条目，manifest `8e70dea69f39beafa0f17b3a6d4b483c537622fdf52271755fd8f3e2447bd54d`；保留原标签与243验证图身份，没有加入本轮曝光图片/视频。

保持v17的12轮、seed20261006、batch/nbs8、AdamW lr0=.00005、imgsz640、warmup0、mosaic0和增强方案；冻结前23个特征模块及全部BN缓冲，检测头819779个参数学习，DFL固定，head BN正常学习。无教师损失、无阈值改动、无中途追加轮数。实际配置对照仅freeze与run名称不同；EMA逐轮恢复固定特征原值也是方法区别，不能把结果归因于单一权重冻结动作。

8图CPU真实一步预演通过：损失box/cls/dfl为2.001191/1.871194/1.872403，42个头权重张量改变、全部头梯度有限、特征无梯度且参数/缓冲逐项未变。预演权重丢弃。GPU实际499状态起点核对通过，12轮共3204 batch/3198次真实更新完成；逐轮冻结、BN模式、头更新、优化器节拍均通过。保存best/last的特征与实际v16起点逐项一致。13份实际源码字节快照留在run的source_snapshot，同时记录Git LF规范化哈希；跨平台复现要区分工作区字节与Git内容。此前保存v17快照时默认GBK读取失败，未写入输出，改用显式UTF-8后成功。

本机152项完整依赖测试通过。方案/训练源码9ee5010、比较方案2ae2536、监督来源审计cc6eef9均在结果出现前或训练阶段提交推送。上述各提交的push/PR CI均成功。

## 图片比较和代价

旧286图AP、5图蓝焰主定位与55派生图沿用经权重/数据/评测源码绑定的v16/v17缓存；v18两份实际重跑。五模型的固定来源计数和补充物理火焰框全部重新推理。旧标签指标与补充物理火焰指标分别保留，未改旧测试标签。全部是曝光开发诊断。

| 模型 | 蓝焰主定位/5 | 物理紧框TP/FP/FN（15框） | 物理F1 | 平均最佳IoU | 旧mAP50 | ks_flame TP/57 | 原无火100图FP |
|---|---:|---|---:|---:|---:|---:|---:|
| v16 best | 4 | 6/11/9 | .3750 | .3849 | .6830 | 46 | 17 |
| v17 best | 3 | 11/8/4 | .6471 | .5873 | .6093 | 46 | 19 |
| v17 last | 4 | 12/7/3 | .7059 | .6425 | .4190 | 35 | 15 |
| v18 best | 4 | 7/13/8 | .4000 | .4624 | .6016 | 46 | 18 |
| v18 last | 4 | 10/5/5 | .6667 | .5237 | .5476 | 43 | 18 |

v18 last较v16提高4个紧框匹配，较v17 last少2个，但少2个补充FP、旧来源ks_flame多8TP。旧AP仍低于v16；v18 last与v17 last的旧AP评测recall恰好相同(.55844)，不能声称旧recall全面恢复。286+55图固定阈值micro F1：v16 .78599、v17 last .66403、v18 last .70635。v18两份原无火FP均18，未消除无火问题；原空标签不全是重新审核的无火事件。

实际查看v18 best/last各3页物理框（共6页）：last在索引1/2/5/6更紧，锅底7仍漏，3/4/8/10仍有过大框；10个匹配包含索引14的3个开放炉头，不能说锅底定位已解决。best仍有锅体范围与嵌套框。

## 同帧视频实际复核

9段视频307个共同采样、五模型1535次推理，约1fps。全部307条视频SHA/帧号/时间/尺寸/像素SHA与v17比较一致，v16/v17 best/last全部921份预测精确复现。实际查看9页固定首/中/末画面（27个原画面，保留v16/v18 best/last三行并记录原图哈希），再查看所有15个计数不同画面的5页五模型比较。加上6页图片，共20页实际检查；计数不同是诊断选择规则，不能推导准确率。

视频6：23个采样中v18两份均21有框/21框，v16为21/22，v17 best为21/23、last为21/24；样本3和22的重叠/碎片框减少，整段覆盖未验收。

视频7：27个采样，v16为25有框/37框、v17 last为24/34，v18两份均25/35。实际确认样本1的可见蓝焰由v18恢复，但last在该处仍重叠；样本0 last只框住左侧蓝焰分量，少框不能解释为完整定位。其余共同空样本没有新增全帧标签。

视频9：60采样全部有框，v16/v17 best/last/v18 best/last框数为64/69/72/66/70。实际查看挂件出现的差异样本33/35/36/37/38/39/40以及共同错误45：v17 best在前7个中误框6处、last7处；v18 best误框35/40两处，last误框33/35/36/38/40五处。样本45全部模型误框挂件，v18 last还给同一挂件两个嵌套框。这里只是该复核子集，不能作为完整视频误报率。

基于这些实景收益与代价，推进v18 last修正定位，保留v18 best作为头更新程度对照；演示保持v16 best。采用理由不是单独旧AP门槛，也不能仅用框数或匹配数宣布替换。

## ONNX与独立CPU入口

best/last动态矩形FP32 ONNX各80/80通过同一预处理、原始输出与NMS一致性校验（坐标.1像素/概率.0001）。原始坐标最大误差分别.00097656/.00091553，概率.00000364/.00000417。导出不含NMS；运行参数imgsz640、conf.25、iou.6。

last独立CPU图片实跑，独立视频入口实际解码734帧、抽样27帧、25有框/35框，CPU PyTorch与ONNX CLI固定容差27/27通过。GPU与CPU严格数值匹配只有12/27，框数全部一致；坐标最大差.01727、置信度最大差.00107548超出.0001。保留不通过事实，未放宽容差。输出boxed视频重新解码确认27帧，图片及首/中/末三帧全部实际查看并留哈希。

本机视频ONNX平均单次forward约59.46ms，不含解码/预处理/NMS/画框，不能据此宣称完整实时或K230速度。没有连接开发板，没有转换KModel、量化或进行任何实机试验。

## 监督来源与下一步

[监督构成](../../docs/flame_supervision_composition_v18_20261003.json)：审核原来源物理火焰37图/261条目，占2131条目的12.25%；另有55份审核派生记录。generic_fire795图、ks_flame184图，原空标签未全量重新目视确认。该审计没有改变本轮数据，不能证明误检的单一原因。

下一步优先审核训练侧原始全场景的挂件/反光与遮挡火焰，保持物理框口径及曝光身份，再冻结下一轮采样方案。继续保留v16原特征路线和v17几何参照；未经审核不把本轮视频塞入训练，不依据同一开发图反复筛中间轮次。烟雾组合与事件层仍是独立缺口，现有fire/smoke组合不能直接套到新fire特征。

## 产物和复现

[训练/图片/视频结果汇总](../../docs/fire_frozen_flame_head_v18_results_20261003.json)、[导出和CPU验证](../../docs/fire_frozen_flame_head_v18_pc_runtime_20261003.json)。本地权重位于`model_work/runs/fire-frozen-flame-head-v18-20261003/weights`，大模型/图片/视频/逐帧记录仍在本地；Git保存代码、小汇总和经过。last权重SHA为`2534a8c727157d0333bf7e0b785cd09e950f5e6e0192831947ba26cfe865b3ce`；last ONNX SHA为`bbbeeeb4372727b75e97cc372065de703b3ae709108df3ab1fca4dfe037686ff`。

使用本机miniconda3 Python及已记录依赖；下面为本轮命令，输出已存在时拒绝覆盖，复现须换新输出目录：

```powershell
python -B model_work/src/train_frozen_flame_head_v18.py --start model_work/runs/fire-scratch-semantic-v16-r2-20261003/weights/best.pt --data D:/SRTP_Datasets/fire_semantic_extension_v17_20261003/data.yaml --rehearse-cpu --out model_work/out/frozen_flame_v18_cpu_rehearsal_20261003
python -B model_work/src/train_frozen_flame_head_v18.py --start model_work/runs/fire-scratch-semantic-v16-r2-20261003/weights/best.pt --data D:/SRTP_Datasets/fire_semantic_extension_v17_20261003/data.yaml --out model_work/runs/fire-frozen-flame-head-v18-20261003
python -B model_work/src/screen_frozen_flame_head_v18.py --run model_work/runs/fire-frozen-flame-head-v18-20261003 --data D:/SRTP_Datasets/fire_semantic_extension_v17_20261003 --review model_work/out/legacy_visible_flame_audit_20261002 --previous model_work/out/fire_extension_v17_comparison_20261003 --out model_work/out/frozen_flame_v18_comparison_20261003
python -B model_work/src/eval_project_model_videos.py --models model_work/out/frozen_flame_v18_comparison_20261003/models.json --videos-dir viedos viedos/视频/视频 --out model_work/out/project_v18_videos_20261003
python -B model_work/src/prepare_video_count_difference_review.py --comparison model_work/out/project_v18_videos_20261003 --out model_work/out/v18_all_video_count_differences_20261003
python -B model_work/src/verify_onnx_export.py --weights model_work/runs/fire-frozen-flame-head-v18-20261003/weights/last.pt --expected-sha256 2534a8c727157d0333bf7e0b785cd09e950f5e6e0192831947ba26cfe865b3ce --data D:/SRTP_Datasets/fire_semantic_extension_v17_20261003 --dynamic-rectangular --out model_work/out/frozen_flame_v18_last_dynamic_onnx_20261003
python -B model_work/src/pc_onnx_infer.py --model model_work/out/frozen_flame_v18_last_dynamic_onnx_20261003/model.onnx --expected-sha256 bbbeeeb4372727b75e97cc372065de703b3ae709108df3ab1fca4dfe037686ff --image model_work/out/project_v16_videos_20261003/source_review_frames/video_07_sample_0013.png --rectangular --out model_work/out/frozen_flame_v18_last_cpu_image_20261003
python -B model_work/src/pc_onnx_video.py --model model_work/out/frozen_flame_v18_last_dynamic_onnx_20261003/model.onnx --expected-sha256 bbbeeeb4372727b75e97cc372065de703b3ae709108df3ab1fca4dfe037686ff --video viedos/视频/视频/82f51e33ff7ddbdcda9ea4c6754b9983.mp4 --rectangular --boxed --out model_work/out/frozen_flame_v18_last_cpu_video_20261003
python -B model_work/src/verify_pc_video_cpu_parity.py --comparison model_work/out/project_v18_videos_20261003 --runtime model_work/out/frozen_flame_v18_last_cpu_video_20261003 --weights model_work/runs/fire-frozen-flame-head-v18-20261003/weights/last.pt --out model_work/out/frozen_flame_v18_last_cpu_video_20261003/cli_prediction_parity.json
```
