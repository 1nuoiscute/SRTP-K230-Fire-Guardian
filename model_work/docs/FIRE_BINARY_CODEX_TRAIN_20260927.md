# 单类火焰训练与厨房逐图验收（2026-09-27）

执行者：Codex。状态：训练完成，电脑端探索性验收完成；未编译 KModel，未替换板端模型。

## 数据和复现入口

- 原数据：`D:\SRTP_Datasets\kitchen_vision_v3_20260926`，原图和标签未改动。
- 新数据：`D:\SRTP_Datasets\kitchen_fire_binary_codex_20260927`，只保留 `0 fire` 标签；训练集每个源图组保留一张。图像使用 NTFS 硬链接，标签另存。
- 构建脚本：`model_work/src/build_fire_binary_codex.py`。
- 训练脚本：`model_work/src/train_fire_binary_codex.py`。
- 评测脚本：`model_work/src/eval_fire_binary_codex.py`。
- 起点权重：`C:\Users\ASUS\Documents\Codex\kitchen-fire-detection\runs\kitchen-vision-v3-640\weights\best.pt`，SHA-256 `f68d756674d2ad36bc9e0df1161a1477c2b51c9cf4792d9ba25c935c6c769d43`。
- 新权重：`model_work/runs/fire-binary-codex-v1/weights/best.pt`，SHA-256 `5f0fa05ebb92eb73cb39971159751e499df3f91d075c70ca818877c64ba05148`。

训练集 2382 张：GC 灶火 1045、Kitchen Safety 火焰 184、通用火焰 800、室内无火 153、锅具无火 200。验证集 243 张，测试集 286 张。训练和测试无相同 SHA-256 图像；Kitchen Safety 不同日期的画面仍来自同一厨房，不能称为独立厨房验收。

配置：YOLO11s 单类，640 像素，batch 4，AdamW，初始学习率 0.001，最多 60 轮，patience 15，seed 20260927。第 32 轮早停；最佳第 17 轮。最佳验证集 P=0.632、R=0.792、mAP50=0.570、mAP50-95=0.189。

## 冻结测试结果

标准 YOLO 测试集（286 张，77 个火焰框）：P=0.715、R=0.649、mAP50=0.639、mAP50-95=0.182。该指标只针对单类 fire，不能直接与原四分类模型的整体 mAP 比较。

厨房目标子集 146 张：71 张有火（KS 57、GC 14），75 张已确认无火（KS）。下表区分“画面出现火框”和“至少一个真值火焰框与预测框 IoU≥0.5”。

| 阈值 | 模型 | 有火图出现火框 | 有火图定位匹配 | 无火图误报 |
| --- | --- | ---: | ---: | ---: |
| 0.25 | 旧 v3 四分类 | 71/71 | 62/71 | 11/75 |
| 0.25 | 新单类 | 70/71 | 54/71 | 0/75 |
| 0.50 | 旧 v3 四分类 | 54/71 | 47/71 | 0/75 |
| 0.50 | 新单类 | 62/71 | 46/71 | 0/75 |

逐图原始结果：`model_work/out/fire_binary_codex_eval_20260927/per_image.csv`；汇总：同目录 `summary.json`。旧模型对照：`model_work/out/independent_kitchen_eval_20260927/per_image_scored.csv`。

## 判断

新模型在低阈值下消除了这批厨房无火图片的误报，在 0.50 阈值下多检出 8 张有火图；但 IoU≥0.5 的定位数量未改善，0.25 阈值还少 8 张。严格定位指标 mAP50-95=0.182 仍低。当前只能保留为电脑端候选，不能称作已达到厨房火焰识别要求，也不应据此替换板端模型。

下一步优先核对 GC/KS 小蓝焰与锅下火的真值框及新旧模型定位差异，补不同厨房、不同灶具的有火与无火照片，并用厨房级隔离划分重训和验收。完成电脑端验收后再考虑 KModel 编译与 K230 实测。

## 补充：用户早期手标厨房图片回归测试

此前未纳入上述 146 张测试。2026-09-27 补测 `C:\Users\ASUS\Documents\Codex\kitchen-fire-detection\datasets\kitchen-images` 共 64 张：normal_cooking 20、big_flame 24、kitchen_fire 11、no_fire 9。逐图结果在 `model_work/out/manual_kitchen_regression_20260927/per_image.csv`。

该批图片已经由旧 `prepare_data.py` 全部复制进旧模型训练集，且本次单类模型从旧链条权重继续微调，因此**只作回归诊断，不是独立测试**。原手标中 normal_cooking 只有 11/20 张带 fire 框，big_flame 只有 8/24 张带 fire 框；类别文件夹名不能替代逐图真值。

| 类别 | 图片 | 手标有火图片 | 阈值 | 旧模型出现火框 / 定位 IoU≥0.5 | 新模型出现火框 / 定位 IoU≥0.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| normal_cooking | 20 | 11 | 0.25 | 5 / 3 | 10 / 4 |
| normal_cooking | 20 | 11 | 0.50 | 2 / 1 | 1 / 1 |
| big_flame | 24 | 8 | 0.25 | 14 / 6 | 19 / 6 |
| kitchen_fire | 11 | 11 | 0.25 | 11 / 9 | 11 / 7 |
| no_fire | 9 | 0 | 0.25 | 0 / 0 | 1 / 0 |

`normal_cooking/4985.jpg_wh860.jpg` 中明显可见锅下橙焰，旧新模型都未检出；`normal_cooking/Snipaste_2026-07-07_18-25-34.jpg` 中有明显蓝焰，新模型虽然报出 fire，但与手标框 IoU 仅 0.076。该批回归测试进一步表明正常灶火识别远未可靠。
