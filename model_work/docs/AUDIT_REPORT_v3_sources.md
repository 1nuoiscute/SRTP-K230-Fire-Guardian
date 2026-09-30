# v3 新数据源审计报告（Tugas Akhir v3 / Kitchen Safety v3）

> ## ⚠️ 本文由 **DeepSeek** 生成，仅供参考
>
> - **撰写者**：DeepSeek 编码代理（`deepseek-flash`，运行于 DeepSeek Harness），**不是人工验收**。
> - **性质**：执行单第 8.1 节交付物（准入清单与排除清单）的说明，**仅供参考**。
> - **不构成依据**：不构成版权法律意见、不构成验收结论。**许可与侵权判断必须由人决定。**
> - **未改动任何原数据**；未训练、未刷板。

日期：2026-09-26
审计脚本：`model_work/src/audit_v3_sources.py`、`tugas_watermark_scan.py`、`tugas_source_verdict.py`
逐图清单：`model_work/out/audit_v3_sources/source_manifest_v3.csv`
抽样图：`out/audit_v3_sources/viz/`、`out/tugas_watermark/sheets/`、`out/watermark_probe/`

---

## 1. 结论速览

| 数据源 | 机械检查 | 内容判定 | 建议 |
| --- | --- | --- | --- |
| **Tugas Akhir v3**（260 图，2 类） | ✅ 图签配齐、无越界、无跨 split 同 SHA256 | 🔴 **混入图库水印图与短视频封面** | **整源排除** |
| **Kitchen Safety v3**（592 图，5 类） | ✅ 图签配齐、无越界、无跨 split 同 SHA256；README 840 差额 **248** 已查明 | ⚠️ 需按拍摄段分层审图；val/test 有近重复 | 可作候选，**必须按拍摄段重切** |

---

## 2. Tugas Akhir v3 —— 建议整源排除

### 2.1 机械层面是干净的

```
train 182 / valid 52 / test 26 = 260 图（README 声明 260，差额 0）
图签配齐：缺标签 0；空标签 0；坐标越界 0
跨 split 相同 SHA-256：0；源内重复：0
近重复 vs train：valid 3/52、test 1/26（dHash≤4）
类别：0 Blue Flame = 215 框、1 Orange Flame = 111 框
预处理：无增强，Resize 1920x1080 拉伸
```

**注意**：全部 260 个文件名都是 `Api-<Blue|Orange>-<N>-` 前缀 —— 与 Kitchen Safety
的相机风格命名（`20240902_212505`、`PXL_20240902_154429792`）形成对照，说明 Tugas
是同一批次来源，不像实拍。

### 2.2 内容层面：已人眼确认的问题样例

| 文件 | 确认的问题 | 证据 |
| --- | --- | --- |
| `train/Api-Blue-116-…` | **iStock 图库水印** | 画面中部 `iStock` 水印 + `Credit: zoom-zoom`；左下角图片编号 `91515926` |
| `train/Api-Blue-87-…` | **iStock 图库水印** | `iStock` + `Credit: kvkirillov`；左下角编号 `856908552` |
| `train/Api-Blue-127-…` | **pngtree 图库水印** | 全幅重复斜向 `pngtree` + 图标 |
| `test/Api-Blue-132-…` | **pngtree 图库水印** | 黑底重复斜向 `pngtree` + 2 个蓝焰 |
| `test/Api-Orange-99-…` | **短视频封面** | 印尼语大字块 `API KOMPOR KECIL TIDAK STABIL?` + `ADK CV. ANDHY KARYA` 企业 logo |

拼接图另见多张带 **iStock 署名水印**：`lonelyking`、`artitwpd`、`akiyoko`、`PitukTV`、
`Yummy_delight`、`Yuliia Vaisman`、`Pluk_Studio`；以及 **123RF** 水印；
以及多张 YouTube 封面版式（印尼语大字块 `+TEPAT`、`SELVIS SENDIRI` 等）。

### 2.3 判定与理由

**建议：Tugas Akhir v3 整源排除（0 张进入训练候选）。**（用户已裁定采纳）

理由：
1. **项目页的 CC BY 4.0 不覆盖嵌入的第三方图库内容。** iStock / pngtree / 123RF 是
   商业图库，其水印图出现在数据集里说明该数据集的"CC BY 4.0"标注与实际内容不符。
2. **另有短视频封面与带企业 logo 的画面**，同样不属于 CC BY 4.0 可授权范围。
3. 执行单 8.1-2 明确：**"若 0 张合格，就明确排除此源，不凑数。"**

> ⚠️ **本节不是法律意见。** 我给出的是"内容与许可声明不符"的技术观察，
> 是否可用的最终判断应由人（必要时咨询法务）决定。

### 2.3.1 一处论证更正（用户指正）

我原先额外写了"全 260 张同批次同前缀 ⇒ 无法切出干净子集"，把 **`Api-` 统一前缀**
当作排除理由之一。**这个论证过强，已按用户意见删除。**

理由：**文件名前缀是来源分布的观察，不是版权依据**。
一批图共用一个前缀只能说明它们可能来自同一采集/整理流程，
**不能据此推断版权状态**——同一批里完全可能存在已获授权或公有领域的图，
也可能恰好相反。版权结论只能建立在**逐张的内容证据**（水印、署名、编号）上。

因此本报告的排除结论**仅**建立在第 2.2 节人眼确认的水印/封面样例之上。
"同前缀"这一事实保留在 2.1 节作为**背景观察**，不参与判定。

### 2.4 我的检测器为什么不能当裁决依据（诚实说明）

我写了启发式水印检测（`tugas_source_verdict.py`），结果**不可靠**：

- 它给出 57 张 `hold_watermark_or_overlay`、17 张 `hold_suspect_stock_look`、**186 张 candidate**；
- 但 **`Api-Blue-132` 明明铺满 pngtree 水印，S1 却为 0（漏检）**；
- 同时 S3（纯色块）把"暗背景 + 蓝焰"大量误判（误报）。

**因此 186 这个 candidate 数字不可引用。** 结论建立在上面**人眼确认的样例**与
"全部同前缀"这一结构性事实上，不建立在检测器评分上。

---

## 3. Kitchen Safety v3 —— 候选，但必须重切

### 3.1 机械层面干净

```
train 418 / valid 137 / test 37 = 592 图
图签配齐：缺标签 0；坐标越界 0
跨 split 相同 SHA-256：0；源内重复：0
类别框：0 Burner=106、1 Flame=384、2 Smoke=189、3 Spill=370、4 safe=251
空标签：train 52、valid 16、test 7
```

### 3.2 README「840」差额 248 已查明

- README 头写 **840 images**，实际 **592**；
- 该 README 的版本头是 `Kitchen Safety - v3 2026-01-26 3:16pm`，而**导出日期是 2026-09-26**；
- 即：**README 描述的是 v3 版本创建时的概况，未被本次导出更新**。
- 结论：**不是数据损坏**，是 Roboflow 导出元数据陈旧。实际图数以 **592** 为准。

### 3.3 关键风险：val/test 与 train 有明显视觉重叠

| 评测集 | 图数 | 与 train dHash≤4 | 其中 dHash=0 |
| --- | ---: | ---: | ---: |
| valid | 137 | **52** | 9 |
| test | 37 | **25** | 7 |

**test 有 68%（25/37）与 train 近重复。** 与 GC 的问题同源：Roboflow 按增强/相邻帧切分。
→ **原 val/test 绝不能直接用于正式验收。**

### 3.4 拍摄段线索

文件名是相机风格的时间戳（`20240902_212505`、`PXL_20240902_154429792`），
可按**日期+时间**推断拍摄段。粗分桶（按编号/50）得到 train 6 / valid 6 / test 5 桶，
但**必须用完整时间戳做精确分段**，并人工处理边界。

### 3.5 类别语义（已人眼确认，影响映射口径）

抽看两张带框可视化（`out/audit_v3_sources/viz/`），配色为
红=0 Burner、绿=1 Flame、橙=2 Smoke、黄=3 Spill、紫=4 safe：

| 类 | 框落在哪 | 实例 | 对映射的影响 |
| --- | --- | --- | --- |
| `0 Burner` | **单个炉头** | `20240902_212558`：红灯头 | ✅ **不能映射为 `stove`**（四槽的 `stove` 是"整个灶具"）——与执行单 8.2 一致 |
| `1 Flame` | 可见火焰 | 同上：绿灯头火苗 | ✅ `→ 0 fire` |
| `2 Smoke` | **锅内飘起的蒸汽/白烟** | `PXL_20240915_…`：绿框标在锅上方的蒸汽 | ✅ **暂缓正确**——确实混入了烹饪蒸汽 |
| `3 Spill` | **台面上的白纸/铺开物** | `20240902_212558`：黄框白纸 | ✅ 不映射 |
| `4 safe` | **锅、盘子等"安全物体"** | 两图中紫框都在钢锅/盘子上 | ✅ 不映射（它不是"无火区域"，而是"安全物体"） |

> 关键确认：**`2 Smoke` 框的是蒸汽**，因此执行单"先排除蒸汽与混标"的要求是必要的；
> 而 `4 safe` 指的是物体而非区域，不能当作无火负例使用。

---

## 4. 跨源对照

| 对照 | 结果 |
| --- | --- |
| Tugas vs GC（dHash≤4） | **0 / 260** |
| Tugas vs 旧 fire-flame | **0 / 260** |
| Kitchen Safety vs GC | **0 / 592** |
| Kitchen Safety vs 旧 fire-flame | **0 / 592** |

> **哈希阴性不证明拍摄事件独立**，只说明没有近重复副本。执行单已明确此限制。

---

## 5. 逐图清单字段（`source_manifest_v3.csv`）

`source`、`orig_path`、`sha256`、`orig_split`、`orig_classes`、`n_boxes`、
`empty_label`、`seq_guess`、`watermark_suspect`、`license`、
`review_verdict`（待人工填）、`exclude_reason`（待人工填）、`target_split`（待人工填）

---

## 6. 下一步（待用户/Codex 裁决）

1. **Tugas Akhir**：接受"整源排除"？还是要求我按更细的人工流程逐张复核剩余 255 张？
   （我可以做，但**逐张版权判断超出我的能力**，且已有足够证据支持排除。）
2. **Kitchen Safety**：按拍摄段重切并映射（仅 `1 Flame → 0 fire`；
   `0 Burner` 不映射到 stove；`2 Smoke` 暂缓；`3 Spill`/`4 Safe` 不映射），
   产出 `kitchen_vision_v3_20260926`。
3. **纯无火难例**：Kitchen Safety 的空标签图（train 52/valid 16/test 7 = 75 张）
   需**逐张确认确实无可见火焰**后才可当负例——执行单 8.1-3 明确要求。

---

**再次声明**：本文由 **DeepSeek**（`deepseek-flash`，DeepSeek Harness）自动生成，**仅供参考**。
第 2 节的水印与版权判断是**技术观察，不是法律意见**；第 2.4 节已说明我的自动检测器不可靠，
结论依据的是人眼确认样例。任何排除/准入决定请人工复核后作出。
