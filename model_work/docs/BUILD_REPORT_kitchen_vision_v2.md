# kitchen_vision_v2_20260926 数据构建交付报告

> ## ⚠️ 本文由 **DeepSeek** 生成，仅供参考
>
> - **撰写者**：DeepSeek 编码代理（`deepseek-flash`，运行于 DeepSeek Harness），**不是人工验收**。
> - **性质**：执行单第 5 节交付物 ①②③④ 的交付说明，**仅供参考**。
> - **不构成依据**：不构成事实认定、验收结论或决策依据。所有判定须回查 manifest、脚本输出与原始数据。
> - **本次未刷板、未训练**。板端 K1/K2 配置未触碰。

日期：2026-09-26
数据集：`D:\SRTP_Datasets\kitchen_vision_v2_20260926`
构建脚本：`model_work/src/build_kitchen_vision_v2.py`（可复跑，拒绝覆盖）
验收脚本：`model_work/src/accept_build_v2.py`、`model_work/src/resplit_by_visual_family.py`

---

## 1. 结论摘要

| 项 | 结果 |
| --- | --- |
| 构建 | ✅ 完成，跨 split 泄漏 **全部清零**，验收 **PASS** |
| 目标域（GC 灶火）在 val/test 的可信独立样本 | ⚠️ **仅 27 / 14 张** —— 不足以支撑分项精度结论 |
| 最重大发现 | 🔴 **GC v2 上游自己的 valid/test 有 86% 与 train 视觉重叠** |
| 建议 | 先补真实独立厨房灶火样本，再谈正式验收；本轮只能给探索性结果 |

---

## 2. 数据集构成

```
D:\SRTP_Datasets\kitchen_vision_v2_20260926\
  data.yaml            4 类 0 fire / 1 smoke / 2 stove / 3 pan
  manifest.csv         6,092 行逐张记录（见第 5 节字段说明）
  build_report.json    构建参数、逐源映射前后统计、上游泄漏发现
  leak_report.json     组泄漏检查
  manifest_visual_family_fix.json   视觉重复族重切记录
  images/{train,val,test}/  labels/{train,val,test}/
```

| split | 图数 | 空标签 | 框 fire | 框 stove | 框 pan | 源组 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | **5,748** | 411 | 9,058 | 511 | 88 | 2,968 |
| val | 186 | 152 | 41 | 6 | 1 | 186 |
| test | 158 | 132 | 20 | 4 | 8 | 158 |

### 2.1 评测分层（必须分开引用）

| 层 | val | test | 可用性 |
| --- | ---: | ---: | --- |
| **formal**（GC 灶火 + stove + pan） | **66** | **58** | 可作正式验收，但样本极少 |
| exploratory（Home-fire 仅烟，**许可未声明**） | 120 | 100 | **仅探索性**，不进正式结论 |

> `smoke` 框数全为 **0** → **未训练 / 未验收**，不得声称烟雾识别能力。

---

## 3. 逐源映射前后统计（执行单第 2 项要求）

| 来源 | 收进图数 | 映射前框 | 映射后框 | 空标签 前→后 | 有框源组 | 许可 |
| --- | ---: | --- | --- | --- | ---: | --- |
| `gc_kitchen_v2` | 1,848 | `{2:2462, 1:1524, 3:1526, 0:1541}` | `fire: 3986` | 0 → 0 | 1,086 | CC BY 4.0 |
| `fire_flame` | 2,950 | `{0: 5133}` | `fire: 5133` | 9 → 9 | 1,357 | **Private** |
| `stove_detection` | 511 | `{0: 521}` | `stove: 521` | 0 → 0 | 79 | CC BY 4.0 |
| `pan_detection` | 410 | `{0:310, 1:97}` | `pan: 97` | 3 → 313 | 97 | CC BY 4.0 |
| `home_fire` | 373 | `{1: 401}` | **（空）** | 0 → 373 | 0 | **未声明** |

映射规则：
- GC `1 flame-under-pot` / `2 gas-stove-flame` → `fire=0`
- GC `0 flame-reflection` → **丢弃**（作困难负例区域）
- GC `3 no-flame` → **丢弃**（未点火炉位，**不**映射到 stove）
- `fire-flame 0` → `fire=0`；`stove 0` → `stove=2`；`pan 1` → `pan=3`（`pan 0 Black Circle` 丢弃）
- Home-fire：只取 `class 1`（smoke）作无火难例；`class 0`（fire）本轮未使用

### 3.1 完全排除的来源与原因（executed）

| 来源 | 排除原因 |
| --- | --- |
| 旧 `kitchen-images` 64 张 | 许可无据可查（目录内无 README/yaml）、含图库素材与伪标、且已泄漏进旧 `combined/train` |
| Home-fire 33 张空标签 | 类别分类不明，保守排除 |
| `fire-flame` 超限 6,388 张 | 执行单 3.2 要求「不把 9,237 全压进来」，按组抽样上限 3,000 |

---

## 4. 🔴 最重大发现：上游数据自带严重泄漏

### 4.1 实测（含阈值敏感性）

GC v2 上游自身的 valid/test 与 train 有视觉重叠。**重叠率高度依赖 dHash 阈值**，因此给出区间而非单点：

| dHash 阈值 | valid 命中 | test 命中 | 合计 | 占比 |
| --- | ---: | ---: | ---: | ---: |
| **≤0（哈希完全相同）** | 99 | 50 | **149** | **46%** |
| ≤1 | 136 | 64 | 200 | 62% |
| ≤2 | 162 | 75 | 237 | 73% |
| ≤3 | 179 | 83 | 262 | 81% |
| ≤4（本构建采用） | 187 | 92 | 279 | 86% |
| ≤6 | 199 | 99 | 298 | 92% |
| ≤10 | 211 | 107 | 318 | 98% |

**保守下限（哈希完全相同）：valid 99/215、test 50/109，合计 149/324 = 46%。**
即使把标准放宽到"完全相同"，也有近一半的评测图是训练图的重新编码副本。

> 口径说明：`≤0` 一档是最硬的证据 —— dHash 完全相同、但 MD5 与文件名不同，
> 说明是同一张源照片经 Roboflow 预处理后派生的多个身份。
> `1–4` 档可能混杂了同源视频的**相邻帧**，不能一律称为"同一张图"，故不单点断言 86%。

### 4.2 成因

Roboflow 在导出时：
1. 把**同一张源照片**派生出多个 `.rf.<hash>` 身份 —— 实测 `DSC_4779 / 4793 / 4800 / 4801 / 4808`
   与 `DSC_4780` 的 dHash **完全相同**，但文件名与 MD5 各不相同；
2. 按**增强变体**随机切分 train/valid/test，于是同一张照片的副本散落到不同 split。

因此**按 `.rf.` 前缀分组抓不全同源**（我第一版就是这么做的，遗留 512 处跨 split 近重复）。

### 4.3 采取的处理

按**视觉重复族**重切（`resplit_by_visual_family.py`）：
- 64-bit dHash + union-find，阈值 ≤4 建族；
- 含 GC 的族跟随 GC 源 split（保住目标域代表性）；其余族多数决；
- 129 个族整体重分配，物理移动 **527 张**。

**处理后残留：近重复 0、源组跨 split 0、val/test 多版本组 0。**

### 4.4 后果与建议

**按视觉重复族去重后，目标域（GC 灶火）在 val/test 的可信独立样本只剩 27 / 14 张。**
这意味着：

- 现有测试集**无法支撑四类分项精度结论**；
- 本轮只能给**探索性**结果，必须明确标注；
- **正式验收前需要补真实独立的厨房灶火样本**（不同厨房/不同拍摄时段/不同设备）。

---

## 5. manifest 字段说明

| 字段 | 含义 |
| --- | --- |
| `out_file` / `split` | 输出文件名与所属 split |
| `source` | 来源数据集 |
| `provenance` | 数据池（决定 `eval_tier`） |
| `eval_tier` | `formal`（可正式验收）/ `exploratory`（仅探索） |
| `source_split` | 原数据集的 split（含 group 标注） |
| `source_path` | 原始文件绝对路径（可回溯） |
| `sha256` | 原始文件内容哈希 |
| `group_id` | 源图组 ID |
| `orig_classes` / `new_classes` | 映射前 / 后的类别 |
| `n_boxes_pre` / `n_boxes_post` | 映射前 / 后框数 |
| `empty_label_post` | 映射后是否为空标签 |
| `license` / `license_evidence` | 许可与其证据出处 |
| `scene_review` | 场景人工审核状态 |
| `note` | 特殊说明 |

---

## 6. 验收结果（执行单 3.3）

```
train  图=5748  标签=5748  解码OK=5748  失败=0
val    图=186   标签=186   解码OK=186   失败=0
test   图=158   标签=158   解码OK=158   失败=0
结构问题（缺标签/类ID越界/坐标越界/框出界/字段数/数量不符）：{}  → 0
源组跨 split      : 0
SHA256 跨 split   : 0
近重复(跨组/跨源) : 0
val/test 多版本组 : 0
data.yaml 可读    : True
判定: PASS
```

保留意见：
- 近重复判定用 64-bit dHash 阈值 4，**对强裁剪/旋转不敏感**，不能排除此类隐蔽重叠；
- GC 标注质量仅做几何筛查（越界 0、极小框 3、极端长条 1）+ 少量人眼抽检，**未全量人工复核**。

---

## 7. 冒烟测试与正式训练命令（执行单第 4 节）

### 7.1 前置检查：15 项全部通过

```
[OK] 旧权重 SHA-256 与执行单一致        39d84feaef74d0c8
[OK] ultralytics 可导入                 8.4.70
[OK] torch 可导入                       2.12.0+cu132
[OK] CUDA 可用                          NVIDIA GeForce RTX 5060 Laptop GPU
[OK] ultralytics settings.json 可读     （沙箱警告在本会话不成立）
[OK] data.yaml 存在且四类顺序正确
[OK] images/train 5748 / val 186 / test 158 张
[OK] 正式 run 名 kitchen-vision-v2-640 未被占用
[OK] 冒烟 run 名未被占用
[OK] 旧权重可加载且类别正确             {0:fire, 1:smoke, 2:stove, 3:pan}
[OK] 旧权重在冒烟子集上可完成 val
=> 15/15，失败 0；报告见 out/preflight_v2.json
```

### 7.2 冒烟测试（1 epoch，fraction=0.05，独立试验名）

```
Epoch 1/1  GPU_mem 1.15G  box_loss 1.615  cls_loss 2.496  dfl_loss 1.671  72 batches
Val: all 186 图 / 48 实例 -> P 0.157  R 0.111  mAP50 0.0685  mAP50-95 0.0241
     逐类: fire 27/41 (mAP50 0.0002)  stove 6/6 (0.205)  pan 1/1 (0)
权重已保存（last.pt / best.pt 各 19.2 MB）
```

**判定：数据加载、loss 计算、验证、权重保存、CUDA 全部正常。**
（冒烟仅用 5% 数据、1 epoch，指标无意义，不得引用。）

输出：`runs\kitchen-vision-v2-smokecheck-2\`

### 7.3 正式训练命令

已封装为可复跑脚本 `model_work/src/train_kitchen_vision_v2.py`（含本机沙箱必需的
ThreadPool 串行替身补丁）。执行：

```powershell
cd C:\Users\ASUS\Documents\ChatGPT\SRTP\model_work
$env:PYTHONIOENCODING='utf-8'
& 'C:\Users\ASUS\miniconda3\python.exe' src\train_kitchen_vision_v2.py
```

等价的直接调用（与执行单模板一致）：

```powershell
& 'C:\Users\ASUS\miniconda3\python.exe' -c 'from ultralytics import YOLO; m=YOLO(r"C:\Users\ASUS\Documents\Codex\kitchen-fire-detection\runs\kitchen-fire\weights\best.pt"); m.train(data=r"D:\SRTP_Datasets\kitchen_vision_v2_20260926\data.yaml", epochs=80, patience=20, imgsz=640, batch=4, device=0, workers=0, seed=20260926, optimizer="auto", lr0=0.001, lrf=0.1, hsv_h=0.05, hsv_s=0.5, hsv_v=0.4, degrees=5, translate=0.1, scale=0.2, fliplr=0.5, mosaic=1.0, mixup=0.1, copy_paste=0.0, close_mosaic=5, project=r"C:\Users\ASUS\Documents\Codex\kitchen-fire-detection\runs", name="kitchen-vision-v2-640", exist_ok=False, save=True, plots=True)'
```

> ⚠️ 直接调用那条**在只读沙箱下会因命名管道被拒而失败**；用封装脚本可避免。

**状态：正式训练尚未启动**（冒烟已通过，run 名未被占用，随时可开）。
预计耗时为小时级；`GPU_mem 1.15G`，显存充裕，无需降到 batch 2。

---

## 8. 下一阶段的硬缺口

| 缺口 | 影响 |
| --- | --- |
| 目标域 val/test 仅 27 / 14 张可信独立样本 | **无法出分项精度结论** |
| 无独立厨房**视频** | **连续运行误报率不可评估**（执行单第 5 项已明确） |
| `smoke` 零正例 | 烟雾能力未训练/未验收 |
| Home-fire 许可未声明 | 其 220 张无火难例只能作探索性评测 |
| `fire-flame` 许可 Private | 仅本机训练，不对外再发布 |

---

**再次声明**：本文由 **DeepSeek**（`deepseek-flash`，DeepSeek Harness）自动生成，**仅供参考**。
第 2–6 节的数字均来自本机脚本输出与逐字节/逐图检查，但**判定与结论可能出错**；
正式使用前请回查 `manifest.csv`、`build_report.json`、`acceptance_report.json` 原始输出。
