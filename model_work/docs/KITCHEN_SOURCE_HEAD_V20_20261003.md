# v20：不同厨房来源的平衡扩展与固定检测头训练

2026-10-03：正式训练和固定图片/视频/CPU运行验证已完成，见[完整结果与采用取舍](KITCHEN_SOURCE_V20_RESULTS_20261003.md)。演示保留v16 best、定位开发保留v18 last，v20不晋升。以下保留训练前方案、预演和启动时的历史记录；不能把历史“尚未比较”当当前状态。

## 目的与固定口径

v19提高部分紧框指标，却仍漏掉视频中蓝焰分量、误检挂件。此次只使用队列3/4预先冻结的开发角色，补充不同来源的锅下蓝焰、炉内小蓝焰、炉头、食物、反光和挂置厨具。保留原v19训练图/标签/采样权重及原243验证图，不重新命名旧曝光测试为独立留出。

新增14图/13暂定来源组，其中4火焰图/4可见炉头区域、10无可见火焰图。每个新正例组8条采样、负例组4条；同作者Shixart两图各2条，避免该组翻倍。因此新增68采样，最终1445唯一训练图/2409采样。全部原图保留完整下载画面。4张留出图及3张隔离图的图像SHA、group、author和original身份排除，二批联合角色检查保持。暂定组不是独立厨房证明。

[预先方案](../../docs/kitchen_source_extension_v20_plan_20261003.json) SHA `ea09fd68fb3db53fb50114705afb35bfa8d977579bb0ae5d1023e21cb691f86f`；数据manifest SHA `15a509075be0d2ff062894cd7de1160699bbdc7b657d8199bfba77cb80bf36fa`。当前数据路径为D:/SRTP_Datasets/fire_kitchen_source_v20_20261003。

保持v16 best起点、固定前23特征模块和BN缓冲、只学习819779个检测头参数，DFL固定；与v19同12轮、seed20261006、AdamW lr0=.00005、batch/nbs8、640、warmup0/mosaic0，无教师损失。同日起点和日程主要用于解释数据变化收益，不断言有限采样即可因果证明或跨厨房泛化。

比较预先声明v16 best/v18 last/v19 last及v20 best/last；新两份都完成图片与9视频307同帧比较，不按旧AP提前剔除，不挑中间快照。旧286图、蓝焰5图、物理紧框15区域和旧55派生图按原身份比较；新增14训练图单独计开发诊断，不与独立准确率混合。留出4图仍不预测、不参与阈值/模型选择，火焰输出不当危险报警。

## 核验、失败和预演

数据构建首稿把验证身份字段误读为image_sha256，实际为sha256；错误在创建输出前触发，没有训练更新。失败源码及信息留在model_work/out/v20_build_preflight_failure_20261003，修复并新增相同验证schema/重叠拒绝测试后构建成功。原队列、旧数据和构建器未改。

CPU首次预演完成真实更新；随后完善**正式训练**快照hook和最终源码预演必需条件，所以在新r2目录重新预演最终源码，不能直接用首次结果替代最终版本。8图覆盖全部4新火焰、队列3/4各1新负例、2旧火焰。真实1次更新，42个检测头权重张量改变，梯度有限、特征无梯度且全部保持；loss box/cls/DFL为3.201894/2.769903/2.813685。未保存候选权重。

正式启动要求预演的源码/依赖/数据/起点身份全部匹配。实际训练器重建499状态张量精确核对，逐轮检查特征/BN保持及头更新，保存权重再次核对。源码快照在实际训练目录建立后写入，拒绝框架静默改名；训练前不会创建run目录导致Ultralytics自动改名。实际字节与Git换行身份分开保留。

182项完整本机测试通过、无跳过，新8项覆盖分组采样、留出/隔离身份拒绝、验证字节重叠及训练日程保持。[预演汇总](../../docs/kitchen_source_extension_v20_preflight_20261003.json)绑定数据与源码哈希。测试和预演不是识别准确率。

## 复现

原数据和审核文件只读，输出目录存在则拒绝覆盖；同样本复现保持哈希，重新下载则需要新的审核。首次失败和两次预演路径保留。

```powershell
python -B model_work/src/build_kitchen_source_extension_v20.py build --plan docs/kitchen_source_extension_v20_plan_20261003.json --out <new-data-directory>
python -B model_work/src/train_kitchen_source_head_v20.py --start model_work/runs/fire-scratch-semantic-v16-r2-20261003/weights/best.pt --data D:/SRTP_Datasets/fire_kitchen_source_v20_20261003/data.yaml --out <new-rehearsal-directory> --rehearse-cpu
python -B model_work/src/train_kitchen_source_head_v20.py --start model_work/runs/fire-scratch-semantic-v16-r2-20261003/weights/best.pt --data D:/SRTP_Datasets/fire_kitchen_source_v20_20261003/data.yaml --rehearsal model_work/out/v20_head_cpu_rehearsal_r2_20261003 --out <new-formal-run-directory>
```

完整目标仍包括独立厨房、烟雾/蒸汽事件、低误报警和板端链路，均不由本轮训练准备达成。

## 正式启动与比较实现

正式训练已从70705ce启动，实际目录fire-kitchen-source-head-v20-20261003，当前启动观察记录见[launch JSON](../../docs/kitchen_source_v20_launch_20261003.json)。499状态张量精确等于v16 best，819779头参数学习、23特征模块固定；实际args与声明data/epochs/batch/imgsz/optimizer/seed/freeze/lr0/nbs/warmup/mosaic/name/pretrained逐项一致。16份实际源码快照已逐项核对。观察时轮数和更新数是带时间的快照，不当训练完成声明。

351a023的基础CI曾因新增配置比较测试在跳过处理前导入cv2失败。70705ce只修正测试的可选导入边界，模型训练源码和最终CPU预演身份不变；准确push/PR任务均完成成功。本机182项无跳过；基础CI中缺模型依赖的75项跳过，二者范围分开。失败CI记录保留。

比较实现已冻结到[protocol](../../docs/kitchen_source_v20_comparison_protocol_20261003.json)，包括准确旧图片/视频结果、评测源码和采用政策SHA。新入口加载检查通过，但完整候选评测尚未运行，不能从--help宣布比较通过。旧55派生图沿用原v19数据，不因新增14图改变历史计数；新增图单独评分。训练完成后运行：

```powershell
python -B model_work/src/screen_kitchen_source_v20.py --run model_work/runs/fire-kitchen-source-head-v20-20261003 --data D:/SRTP_Datasets/fire_kitchen_source_v20_20261003 --review model_work/out/legacy_visible_flame_audit_20261002 --previous model_work/out/train_context_v19_comparison_20261003 --out <new-comparison-directory>
python -B model_work/src/eval_project_model_videos.py --models <new-comparison-directory>/models.json --videos-dir viedos viedos/视频/视频 --out <new-video-comparison-directory>
```

全部best/last继续进行项目视频复核，留出仍不预测。GPU进程的session只用于重新观察实际进度，句柄缺失或终止须检查产物，不从旧launch快照重启。
