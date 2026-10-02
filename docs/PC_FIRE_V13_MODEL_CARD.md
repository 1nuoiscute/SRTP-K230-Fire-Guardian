# 电脑端 fire v13 开发版本

2026-10-02，v13 best 在冻结筛选门槛内改善已曝光蓝焰和队友火焰定位，作为电脑端新的单类开发候选。v3 原文件保留回退。开发板没有连接，既有板端模型不变。本版本检测可见火焰，正常灶火仍可能检出，未验证异常火情报警。

| 固定对照 | v3 | v13 best |
|---|---:|---:|
| 5 张蓝焰主火焰正确定位 | 0 | 1 |
| 旧 286 图 mAP50 | .973239 | .971276 |
| 旧 ks_flame 正确框/57 | 57 | 57 |
| 旧 kitchen_stove 正确框/20 | 18 | 18 |
| 无火室内 100 图误检框 | 1 | 1 |
| 队友 10 个已审核火焰框 TP/FP/FN | 3/5/7 | 5/4/5 |
| 新增训练蓝焰 13 框 TP/FP/FN | 1/3/12 | 4/1/9 |
| 286 旧图 + 55 暴露难例的微平均 F1 | .832618 | .871795 |

最后一行只合计这两组已有完整框真值的开发图片，不加入仅标主火焰的 5 张蓝焰诊断图。合计 TP/FP/FN 由 97/18/21 改为 102/14/16，按 `2TP/(2TP+FP+FN)` 计算；不同场景的图片数仍有重复/曝光偏差，不是独立总体 F1。

固定 conf=.25、NMS IoU=.6、定位匹配 IoU=.5。所有这些图已经曝光或参与开发；旧 mAP 小幅下降，改进来自难例定位，不能称每项指标都优于 v3，也不能称独立厨房验收完成。仍漏检 4 张蓝焰主火焰、5 个队友火焰框；带蓝色贴纸的视频仍误检。训练集、原始视频、权重和 ONNX 只留本地。

模型架构与 v3 相同，9428179 个参数；只训练 fire Detect，前 23 块特征和全部头 BN 统计保持。12 轮训练实际 4224 batch/4216 次更新，14 次 live/12 次 EMA 与教师检查通过。保存后的 432 个固定状态张量核对通过；best 和 epoch3 所有模型张量完全一致。来源标签的 5 张修正、旧样本教师 KL 约束和完整候选淘汰结果见 [实验记录](../model_work/docs/FIRE_TEACHER_V13_20261002.md)、[汇总 JSON](fire_teacher_v13_results_20261002.json)。

## 本机文件与身份

补充 [旧灶火定位语义复核](PC_FIRE_LABEL_SEMANTICS_20261002.md) 已发现原标签框包含较大锅体：同一10图/15处近似可见火焰上，v13按原标签TP=13，按紧火焰包络IoU≥.5仅TP=1。新包络是曝光来源的单次手工近似标注，不能代替正式真值；这说明本卡原标签高分尚不能证明精确火焰范围定位。原指标和训练标签保留。

目录 `model_work/out/fire_teacher_v13_selected_20261002/`：

| 文件 | SHA256 |
|---|---|
| fire_v13.pt | c2b2a8bdc66fe452c80a1786f918f62f8686ad3b0f11061a761969ee02b09966 |
| fire_v13_dynamic.onnx | cf3c9210505f749e96ae048c358d5c1f4ddca788a0096a7831dc60819c397705 |
| fire_v13_static.onnx | 2dd6fc6d82714fc827443fc3fde0628bc3b0f20dd1d7bc22278b115e25b233f3 |

ONNX 从 epoch3 容器导出，与选定 best 的模型张量完全一致，容器哈希不同，证据记录在 version_manifest.json。FP32/opset12/无内置 NMS；静态输出 `[1,5,8400]`，动态输出支持 stride32 矩形。各 80/80 图原始输出和 NMS 一致性通过；已有图片 CPU CLI 实测，视频 CPU CLI 解码 719 帧、采样 24 帧完成。本机某次前向耗时不能外推 K230 或事件端到端时延。

## 电脑端运行

在仓库根目录用现有电脑 Python 环境执行，例如：

```powershell
python -B model_work/src/pc_onnx_infer.py --model model_work/out/fire_teacher_v13_selected_20261002/fire_v13_dynamic.onnx --expected-sha256 cf3c9210505f749e96ae048c358d5c1f4ddca788a0096a7831dc60819c397705 --image model_work/data/commons_blue_review_20260927/blue_oven.jpg --out model_work/out/v13_my_image --rectangular
python -B model_work/src/pc_onnx_video.py --model model_work/out/fire_teacher_v13_selected_20261002/fire_v13_dynamic.onnx --expected-sha256 cf3c9210505f749e96ae048c358d5c1f4ddca788a0096a7831dc60819c397705 --video viedos/1.mp4 --out model_work/out/v13_my_video --rectangular --boxed
```

输出目录必须是尚不存在的新目录。静态版本省略 --rectangular，参考 PyTorch 时相应设 rect=False。入口只用 ONNX Runtime/NumPy/OpenCV；fire_v13 本身只有 fire 类，已完成的独立 smoke v4 保留为另一开发候选，不在这里混称烟雾识别。正式两类、独立比例与事件要求见 [计划证据矩阵](PLAN_REQUIREMENT_MATRIX_20261002.md)。
