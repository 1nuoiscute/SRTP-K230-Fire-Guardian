# 旧灶火标签与可见火焰范围的补充诊断

2026-10-02，复核完整旧 `kitchen_stove_fire` test 来源：14 图、20 个原标签框。原框常覆盖较大的锅体和灶头区域。对实际任务而言，可见火焰范围与原数据的框口径存在差别；高原标签 mAP 不能直接证明火焰位置准确。

先检查没有预测框的原图，再固定手工近似火焰包络与暂缓项，渲染原框/新框对照，并实际检查全部4张复核页。批准10图/15处可见火焰，4图因遮挡、暗光和锅体反光边界不确定而暂缓。暂缓图明确存在可见火焰，**不是负例**，不计入本次补充评分。原训练标签、旧14图/20框评分和正在进行的从零训练均不改。

这些图以前已参与开发评测，标注者也有此前预测曝光；新复核页不显示预测，不能消除历史曝光。新框为单次人工近似包络，尚无第二标注者或专家一致性；结果只用于定位语义诊断，不当新独立真值、正式验收或模型替换依据。

## 同一10图/15框的成对结果

使用已有身份绑定预测，conf=.25、NMS IoU=.6，不重新筛阈值。以下TP/FP/FN均为框的一对一匹配；每个可见灶头的零散火焰合成一个包络，锅体反射排除。

| 指标 | v3 | 已选 v13 best |
|---|---:|---:|
| 原标签匹配 IoU≥.5：TP/FP/FN | 13/3/2 | 13/2/2 |
| 近似火焰包络 IoU≥.5：TP/FP/FN | 2/14/13 | 1/14/14 |
| 近似火焰包络 IoU≥.3：TP/FP/FN | 7/9/8 | 6/9/9 |
| 每处火焰与最佳预测的 IoU 平均值 | .326127 | .310422 |

随后实际查看两个模型全部6张预测复核页。主要差异是预测仍覆盖较高锅体或整段灶头，紧火焰包络只占预测的一部分。`.3` 是预先记录的近似几何敏感性诊断，未替代`.5`门槛。v13在原开发门槛内的难例改进仍成立，但这里没有改善火焰范围定位的证据。下一步训练路线比较也应保留这项补充口径，避免把原框拟合与物理火焰定位混为同一结论。

## 复现与身份

本地复核目录 `model_work/out/legacy_visible_flame_audit_20261002/` 保留原图/标签SHA、原始页、手工提案、原框/新框页、批准/暂缓理由及完整复核结果；本地评分目录 `model_work/out/legacy_visible_flame_scores_v3_v13_20261002/` 保留逐图预测、两种口径计数、6张预测页及SHA。媒体和逐图身份不公开，脱敏结果见 [汇总JSON](legacy_fire_visible_flame_results_20261002.json)。

复核结果 SHA256：`196927d6b67cfe01c89b94a2aedb3e1f8d876ef3bba17e4079033a773c685682`。手工提案SHA256：`a50e50fc2f7c24d345ec92a504f1a89f309dbb7fe7d04f96f9ac2d6ef610856d`。准备、复核与评分工具分别是 `prepare_legacy_fire_semantic_audit.py`、`legacy_fire_semantic_review.py`、`score_legacy_fire_semantics.py`，拒绝覆盖旧目录并检查原始像素、标签、渲染页、复核来源与缓存输入身份。

新评分目录的命令示例：

```powershell
python -B model_work/src/score_legacy_fire_semantics.py --review model_work/out/legacy_visible_flame_audit_20261002 --source-cache model_work/out/fire_teacher_v13_screen_20261002/legacy_sources --models v3 v13_best --out model_work/out/my_new_semantics_score
```

首次完成记录时，CLI传入正斜杠页路径与Windows生成的反斜杠清单不一致，校验拒绝；按清单的精确页路径记录后完成。首次评分遇到Windows默认GBK读取已有UTF-8缓存，尚未创建结果目录；修正评分工具显式UTF-8后重跑完成。两次失败未修改原标签或训练输入。
