# 旧 286 图按来源的固定阈值定位对照｜2026-09-30

为定位总 mAP 下降的具体表现，新增 eval_legacy_source_localization.py 顺序比较 v3、v9 last、v10 last。实际读取同一旧 test 原图/标签，核对全部图像 SHA、标签数、每模型推理前再次核对标签 SHA；匹配采用一对一 IoU>=.5，conf=.25、NMS=.6、640。不是重算 mAP，也不是新的独立测试。

| 原来源标签 | 图/GT 框 | v3 TP/FP/FN | v9 last TP/FP/FN | v10 last TP/FP/FN |
|---|---:|---:|---:|---:|
| ks_flame | 57/57 | 57/1/0 | 57/2/0 | 57/1/0 |
| kitchen_stove_fire | 14/20 | 18/5/2 | 17/2/3 | 17/3/3 |
| ks_nofire_confirmed | 75/0 | 0/0/0 | 0/0/0 | 0/0/0 |
| object_pan | 40/0 | 0/0/0 | 0/0/0 | 0/0/0 |
| nofire_real_indoor | 100/0 | 0/1/0 | 0/9/0 | 0/4/0 |

固定阈值下，主要厨房火焰来源保持了所有 57 个 GT 的匹配；另一灶台来源多漏 1 框。显著变化出现在烟雾/无可见火焰来源的额外框。这个表不能完全解释 mAP 曲线或高 IoU 下的框质量，也不能因为 TP 保持就认定整体没有退化。

已打开 v10 该来源全部四张错误原画面：三处框指向白/蓝灰烟羽，一处框指向反光台面，而非清楚可见火焰。原来源名称 nofire_real_indoor 不是“所有画面都在室内”或“事件安全”的保证，训练抽样里亦包含室外烟囱等场景。这里只检验可见火焰框；烟雾/阴燃需要独立的传感器与事件判断。

本机完整输入、逐图预测、GT/预测框错误图保留在 model_work/out/legacy_source_localization_20260930。公开 [来源计数](../../docs/legacy_source_localization_results_20260930.json) 只包含配置、权重身份和分类计数，不发布私有逐文件名称、哈希或图像。

复现：--models 指向本机 JSON 列表，每项含 label、weights、expected_sha256。输出目录必须是新的，工具不重标、不训练。三份冻结模型的实际配置留在 .local_archive/legacy_source_models_20260930.json。

```powershell
python -B model_work/src/eval_legacy_source_localization.py --data D:/SRTP_Datasets/kitchen_fire_binary_codex_20260927 --models .local_archive/legacy_source_models_20260930.json --out model_work/out/legacy_source_localization_20260930
```

下一步从当前 train 输入抽取并目视审核烟雾负例。旧 test 中的四张错误画面只做诊断，不直接追加训练。
