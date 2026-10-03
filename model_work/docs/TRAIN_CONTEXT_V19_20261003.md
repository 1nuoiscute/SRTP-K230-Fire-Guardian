# v19训练全场景审核与采样扩展：冻结准备方案

承接v18（9a01155）。重点是训练侧完整图中的非火对象、反光与锅底火焰；不添加已曝光项目视频，不修改旧val/test标签，不把审核数据称为独立测试。

先冻结选择规则，再查看结果：在v17数据的原始train库存中，使用v18 last（SHA 2534a8c727157d0333bf7e0b785cd09e950f5e6e0192831947ba26cfe865b3ce）对尚未人工审核的object_pan/nofire_real_indoor空标签图推理，各选置信度最高12图，零框时以原图SHA排序补足。模型只安排审核顺序，原空标签和预测都不作为自动真值。另按来源组与SHA排序中点分位，从未审核的隔离kitchen_stove_fire原图选8图、原ks_flame选8图。共40图，保留整个原画面，不裁出错误对象后冒充场景。

逐图审核可见物理火焰框；看不清、语义不明或边界无法可靠标注时held。negative仅指该完整图未见可见火焰，不证明没有燃烧/烟雾事件。全部原图和父标签绑定SHA，非train像素禁止加入，既有val身份保持。原始审核图与标注提案页都必须实际查看并绑定页SHA后才允许完成审核。

构建新数据仅改变获批图：审核火焰重标物理框并提高采样到8、审核negative空标签采样到8；held保持原父版本（隔离图仍隔离）。其余像素、标签、来源、权重逐项保持。记录已审核来源和采样构成；不自动宣称高权重可以消除挂件误检。

完成数据后冻结v19训练：与v18同v16起点、固定前23模块/BN、12轮/同seed与增强、仅头学习，无教师损失；改变审核数据和采样，保存实际初始化/头/冻结/优化器/保存权重审计。提前声明best/last均比较：旧286、蓝焰5、55派生、物理15框和9视频307同帧，实际复核误检、遮挡及重复框，之后ONNX/独立CPU验证。若本轮没有足够明确审核样本，先记录不足，不启动无实际数据变化的重复训练。

此阶段为开发反馈整理；独立新厨房、事件隔离、烟雾语义、完整实时和板端验收仍待推进。大文件在本地，Git保存工具、小证据、方案和经过。

## 已完成审核、构建和训练预演

方案1669fbd已先行推送，push/PR CI均成功。对337张尚未审核的原始train空标签图实际挖掘，只有2张nofire_real_indoor烟状羽流图有模型框，object_pan全部无框。10页40张完整原图实际查看；火焰原图及3个羽流/暗室图另看原尺寸。标注提案10页实际复核后，收紧索引33锅底框，再查看修订页（共21页，另外9页字节与已看首版相同，首版保留）。

最终采用15张火焰图/27个近似物理框（7张隔离工业厨房图新增、8张原ks_flame重新审核标签），14张完整图未见可见火焰（2个单锅对照、12个室内/羽流图）。1张模糊锅底图继续隔离，10张无错误且高度冗余的单锅图维持原权重，held共11张。held也可用于避免冗余追加，并非这些图都语义不明。

带烟状羽流/可能阴燃材料的negative仅表示未见可见火焰，不能作为无燃烧或无smoke标签。索引24有彩色挂衣，但这不是项目视频的真实厨具挂件困难场景。现有数据没有因此覆盖项目挂件错误，不宣称已解决。唯一图数也不表示独立厨房数，新增工业厨房与原场景相关。

新数据1431唯一图/2341采样条目全部校验，原243验证图身份、非获批图像/标签/权重和隔离列表保持。数据manifest SHA：`9971c9530a04f1dd962a7c2d9e229d73d5b7e1a47aab5d3d966975d6e78d831a`。本轮权重8是训练反馈实验设定，其效果等待同帧结果。

新8图CPU真实梯度一步：box/cls/dfl 2.379748/1.970842/1.695368，37个头权重张量改变，头梯度有限，固定特征无梯度且逐项保持；预演权重丢弃。正式配置与v18仅data和run名称不同，前23模块及BN固定、819779头参数学习、12轮。158项完整依赖本机测试通过。新增日程测试初次因预期相对路径而实际函数规范化成绝对路径失败，修正测试预期后全部通过，没有修改训练参数规避失败。

[逐图审核与数据证据](../../docs/train_context_v19_review_20261003.json)、[CPU预演与正式配置](../../docs/train_context_v19_preflight_20261003.json)。当前仅审核/构建/预演完成；新权重训练、正式图片和视频比较、运行验证尚待完成，不能称作模型性能提高。

复现命令（输出已存在时拒绝覆盖；复现须用新目录）：

```powershell
python -B model_work/src/curate_train_context_v19.py prepare --parent D:/SRTP_Datasets/fire_semantic_extension_v17_20261003 --weights model_work/runs/fire-frozen-flame-head-v18-20261003/weights/last.pt --out model_work/out/train_context_v19_review_20261003
python -B model_work/src/curate_train_context_v19.py render --review model_work/out/train_context_v19_review_20261003
# decisions与全部实际查看页SHA必须逐项审核后传给finalize，不自动宣布看过
python -B model_work/src/curate_train_context_v19.py build --parent D:/SRTP_Datasets/fire_semantic_extension_v17_20261003 --review model_work/out/train_context_v19_review_20261003 --out D:/SRTP_Datasets/fire_train_context_v19_20261003
python -B model_work/src/train_context_flame_head_v19.py --start model_work/runs/fire-scratch-semantic-v16-r2-20261003/weights/best.pt --data D:/SRTP_Datasets/fire_train_context_v19_20261003/data.yaml --rehearse-cpu --out model_work/out/train_context_v19_cpu_rehearsal_20261003
python -B model_work/src/train_context_flame_head_v19.py --start model_work/runs/fire-scratch-semantic-v16-r2-20261003/weights/best.pt --data D:/SRTP_Datasets/fire_train_context_v19_20261003/data.yaml --out model_work/runs/fire-train-context-head-v19-20261003
```


正式GPU训练已启动，499状态实际v16起点和819779个头参数核对通过；训练尚未完成。比较入口`screen_train_context_v19.py`在结果前冻结，比较v16、v17 best/last、v18 best/last和v19 best/last共7模型；新增数据的55派生图成员/标签必须与父版本逐项一致才允许参考缓存。五份参考的旧AP/蓝焰/55图缓存身份绑定，固定来源和物理15框实际重算七模型，视频七模型全部重跑同307帧。15份实际源码字节快照保存于本地`model_work/out/v19_frozen_sources_20261003`，保留原CRLF与Git LF规范化哈希。

```powershell
python -B model_work/src/screen_train_context_v19.py --run model_work/runs/fire-train-context-head-v19-20261003 --data D:/SRTP_Datasets/fire_train_context_v19_20261003 --review model_work/out/legacy_visible_flame_audit_20261002 --previous model_work/out/frozen_flame_v18_comparison_20261003 --out model_work/out/train_context_v19_comparison_20261003
python -B model_work/src/eval_project_model_videos.py --models model_work/out/train_context_v19_comparison_20261003/models.json --videos-dir viedos viedos/视频/视频 --out model_work/out/project_v19_videos_20261003
```
