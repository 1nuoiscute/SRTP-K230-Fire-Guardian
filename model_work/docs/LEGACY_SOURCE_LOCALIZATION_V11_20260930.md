# v11 完成后的旧图来源定位复核｜2026-09-30

顺序比较同一 286 个旧 test 图、相同标签和固定设置（640/conf=.25/NMS=.6/匹配 IoU=.5）。训练与原 runner 均完成后才启动，五模型全部 exit 0。私有逐图预测、错误画面及输入身份留在 model_work/out/legacy_source_localization_v11_20260930。公开只保存 [来源汇总](../../docs/legacy_source_localization_v11_results_20260930.json)。

| 来源 | 图/GT | v3 TP/FP/FN | v9 last | v10 last | v11 best | v11 last |
|---|---:|---:|---:|---:|---:|---:|
| ks_flame | 57/57 | 57/1/0 | 57/2/0 | 57/1/0 | 56/1/1 | 57/4/0 |
| kitchen_stove_fire | 14/20 | 18/5/2 | 17/2/3 | 17/3/3 | 18/5/2 | 17/3/3 |
| ks_nofire_confirmed | 75/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 |
| object_pan | 40/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 |
| nofire_real_indoor | 100/0 | 0/1/0 | 0/9/0 | 0/4/0 | 0/2/0 | 0/10/0 |

负例清理加权没有降低 last 在烟雾/无可见火焰来源的误检。这个来源名称不能证明画面全部在室内或事件安全；这里只统计可见火焰框。旧诊断已曝光，固定阈值 TP/FP/FN 不是 mAP，也不是每小时误报率。

复现模型配置留在 .local_archive/legacy_source_models_v11_20260930.json；包含 label/weights/expected_sha256。使用 eval_legacy_source_localization.py，输出须为新目录。不存在测试错误图直接追加训练的操作。
