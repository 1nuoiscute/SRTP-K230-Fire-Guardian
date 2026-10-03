# v18：保留v16特征，适配物理火焰检测头

承接已完成的v17（Git a0559ea）。v17 last在补充物理火焰框6→12/15的收益值得继续，但项目视频9新增彩色厨具误检、视频7多漏一帧，当前演示保留v16 best。下一轮检验固定特征路线，不能事先宣称能消除这些错误。

固定方案：起点仍为v16 best（e770cf68d12cd43b4494a1439dc9c65f14d1d5fc570b2e2e379cc084384418fd），数据完全复用v17的1424图/2131采样条目（manifest 8e70dea69f39beafa0f17b3a6d4b483c537622fdf52271755fd8f3e2447bd54d）。保持v17的12轮、seed20261006、batch/nbs8、AdamW lr0=.00005、imgsz640、warmup0、mosaic0和增强方案；冻结前23个特征模块及其BN缓冲，只训练最后检测头，DFL固定。没有加入已曝光测试图或视频、没有教师损失、没有改阈值。head BN仍正常学习。

EMA逐轮恢复固定特征的原值，并复核保存checkpoint的特征与实际v16起点一致；这是固定特征保存措施，也需记录为与v17全网络路线的区别。只有事先声明的best/last评测，不按中间结果追加轮数。比较全网络与固定特征路线，不把结果解释为独立厨房可靠性。

CPU预演已完成：同类8图真实一步更新，819779个头参数学习，42个头权重张量改变，损失2.001191/1.871194/1.872403。全部头梯度有限，特征梯度不存在，特征参数和BN缓冲逐项未变；真实499个状态与v16起点一致。预演权重丢弃，没有候选模型结论。

正式GPU训练必须通过相同实际起点、每轮特征冻结/BN模式、有限状态、头更新和优化器步数审计。完成后对比v16 best、v17 best/last、v18 best/last：旧286图及来源计数、蓝焰5图、55派生图、补充10图15个近似火焰框，以及9段视频307同帧。实际检查厨具误检、遮挡火焰漏检和重复框；旧AP下降不单独否决，项目内表现决定继续或替换。候选仍需ONNX及独立CPU入口验证。

复现（输出已存在时拒绝覆盖）：

```powershell
python -B model_work/src/train_frozen_flame_head_v18.py --start model_work/runs/fire-scratch-semantic-v16-r2-20261003/weights/best.pt --data D:/SRTP_Datasets/fire_semantic_extension_v17_20261003/data.yaml --rehearse-cpu --out model_work/out/frozen_flame_v18_cpu_rehearsal_20261003
python -B model_work/src/train_frozen_flame_head_v18.py --start model_work/runs/fire-scratch-semantic-v16-r2-20261003/weights/best.pt --data D:/SRTP_Datasets/fire_semantic_extension_v17_20261003/data.yaml --out model_work/runs/fire-frozen-flame-head-v18-20261003
```

开发板未连接，全部大文件留本地，Git保存源码/汇总/经过。训练结果尚待完成和正式比较。


正式GPU训练已启动，实际499状态起点核对通过，819779个检测头参数学习，前23模块固定。152项本机测试通过，源码计划提交9ee5010已推送。图片比较脚本同时保留v16/v17缓存参考并实际重算五模型来源/物理框，视频五模型全部重跑同307帧；缓存与实际重跑各自标记。
