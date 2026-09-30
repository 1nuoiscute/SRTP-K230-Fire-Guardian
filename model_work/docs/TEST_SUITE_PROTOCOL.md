# 冻结评测套件 test_suite_v1 — 协议与使用说明

> ⚠️ **本文与其中流程由 DeepSeek**（`deepseek-flash`，DeepSeek Harness）**生成，仅供参考**。
> 非人工结论、非验收依据。套件的泄漏判定、来源映射与类别语义假设均需人工复核后才能正式引用。

冻结日期：2026-09-25
套件路径：`C:\Users\ASUS\Documents\ChatGPT\SRTP\model_work\data\test_suite_v1`
构建脚本：`../src/build_test_suite.py`（可重跑，拒绝覆盖已存在目录）
审计证据：`data/test_suite_v1/MANIFEST_AUDIT.json`、`out/audit/overlap_kitchen_images_vs_combined.json`、`out/audit/clean_data_scan.json`

---

## 1. 为什么需要这份协议

原本打算把 `kitchen-fire-detection\results\` 下那 64 张按场景分类的图直接用作"独立测试集"。
**审计后否决了**：那 64 张就是 `datasets\kitchen-images`，已被 `prepare_data.py` 全部复制进
`combined\images\train`（文件名前缀 `manual_*`）—— **100% 泄漏**。

进一步审计发现整个工程**没有任何未泄漏的数据**：

| 候选来源 | 张数 | 与 combined 的关系 |
| --- | ---: | --- |
| `fire-flame` 的 test 分片 | 936 | 已被 `prepare_data.py:26` 的 `("test","val")` 映射进 **combined/val** |
| `stove-detection` 的 test 分片 | 26 | 同上，进 **combined/val** |
| `pan-detection` 的 test 分片 | 102 | 同上，进 **combined/val** |
| `kitchen-images` | 64 | 全部进 **combined/images/train**，**完全泄漏** |
| `hiennguyen9874_fire-smoke-detection` | 0 | 只有 `.incomplete` 残片，不可用 |
| **`D:\Codes\flame-detection\dataset_yolo`** | **239** | **MD5 与 dHash 双重核对：零重叠 ✅** |

因此本套件**按来源打泄漏标签**，并把唯一干净的部分单独标出，绝不假装整体独立。

---

## 2. 三个泄漏层级（务必分开引用）

| 层级 | 图数 | 含义 | 能用来证明什么 |
| --- | ---: | --- | --- |
| **clean** | 239 | MD5 与 combined 零重叠，dHash 汉明距离 > 6 | **唯一可作独立评测**的部分；可用于"未见数据"结论 |
| **val** | 1,064 | 图像已在 `combined/images/val` | 只可用于**相对比较**（新旧模型同一批图对比），**不能**证明泛化 |
| **train** | 64 | 图像已在 `combined/images/train` | **完全泄漏**，只作回归追踪（退化报警），数值必然虚高 |

> ⚠️ 严禁把 `train` 层的 0.995 mAP50 当作模型能力；它主要反映**记忆**。

---

## 3. 来源与类别映射

统一到旧模型的类别空间（保证与 `best.pt` 直接可比）：`0 fire / 1 smoke / 2 stove / 3 pan`

| 来源标签 | 根目录 | 映射 | 泄漏级别 | 收进 |
| --- | --- | --- | --- | ---: |
| `fireflame_test` | `fire-flame.v1i.yolov11/test/images` | `0→0` | val | 936 |
| `stove_test` | `stove-detection/test/images` | `0→2` | val | 26 |
| `pan_test` | `pan-detection/test/images` | `1→3`（丢弃 `0 Black Circle`） | val | 102 |
| `kitchen_images` | `kitchen-images/` | 原样（已是 0/2/3） | train | 64 |
| `cctv_clean` | `D:\Codes\flame-detection\dataset_yolo/images` | `0→0(fire)`, `1→1(smoke)` | **clean** | 239 |
| **合计** | | | | **1,367** |

标注转换沿用 `prepare_data.py` 的同一算法（多边形 → bbox），面积归一化值保留 6 位小数。

### 类别框统计

| 类别 | 框数 |
| --- | ---: |
| fire | 1,794 |
| smoke | 141 |
| stove | 75 |
| pan | 25 |

空标签（负例）图：103 张。

---

## 4. ⚠️ 需要人工确认的假设（未证实）

1. **CCTV 源的类别语义**：本套件按 `dataset_yolo/data.yaml` 的 `nc=2, names=[fire, smoke]` 解释，
   即 **0=fire、1=smoke**。但该源另一处 `dataset2/README.md` 写的是 `names: ['smoke','fire']`，
   **两者矛盾**。本套件选择相信 `data.yaml`（它是实际训练时使用的配置）。
   **请人工抽查若干张图确认。** 若语义反了，`smoke` 与 `fire` 的正例会互换，结论需整体重算。
2. **`cctv_clean` 的域差异**：该数据集是**合成（synthetic）CCTV 视角**数据，主题为
   "早期小火：冒烟的垃圾桶、人行道上的小纸张火、草地阴燃烟头"，高角度监控视角。
   它与本项目"家庭厨房、近景、烹饪场景"的**域差异很大**。
   因此 clean 层的绝对数值代表的是**跨域泛化能力**，不等于板端厨房场景的表现。
3. **许可**：`dataset/README.md` 标注 `license: cc-by-nc-4.0`（**非商业**）。
   参赛/展示用途需确认是否符合；商业产品化则不可使用。
4. `pan_test` 原源含 2 类，`0 (Black Circle)` 被丢弃 → 部分图成为空标签负例。
5. 重叠判定的强度：精确重复用 MD5；近重复用 64-bit dHash（汉明距离 ≤ 6）粗筛。
   dHash 对**强裁剪、旋转、大幅调色**不敏感，**不能排除**这类更隐蔽的重叠。

---

## 5. 使用方式

套件目录已包含 `data.yaml`，可直接交给 Ultralytics：

```
data/test_suite_v1/
  data.yaml            # path / train / val / test 均指向 images
  manifest.csv         # file, source, leaked, empty, md5, boxes, src
  MANIFEST_AUDIT.json  # 构建参数、来源统计、关键发现
  images/              # 1367 张
  labels/              # 对应 YOLO 标签
```

评测时**必须按 `manifest.csv` 的 `leaked` 列分层**，脚本见 `../src/eval_baseline.py`
（它会自动分层、写子集并输出 JSON）。

### 5.1 已知环境限制（沙箱）

Ultralytics 缓存标签时会开 `multiprocessing.pool.ThreadPool`，而本机沙箱**禁止进程间命名管道**
（`CreateFile` 报 `WinError 5`）。应对方式是**在该脚本内**把
`ultralytics.data.dataset.ThreadPool` 替换成串行替身，并把 `NUM_THREADS` 置 0。

> 注意：`dataset.py` 里 `ThreadPool` 与 `NUM_THREADS` 都是**直接 import 进模块命名空间**的，
> 所以必须 patch `ultralytics.data.dataset` 自己的属性，改 `ultralytics.utils.NUM_THREADS` 无效。

---

## 6. 已被套件揭示的核心问题

**过采样复制造成大面积内容重复**：`combined` 的 17,431 张图只有 **10,222 个不同 MD5**。
`prepare_data.py:104-131` 的 stove/pan 过采样（`TARGET=2000`）把同一张图复制成 `_d0`、`_d1`…
且 `for split in ["train","val"]` 让**验证集也被复制** —— 这是直接的评估泄漏。

**因此旧 `results.csv` 的 mAP50 0.792 不可作为泛化证据。** 本套件的分层结果正是为了量化这一点。

---

## 7. 冻结与回滚

- 套件一旦冻结，**不得**再做增删或重新生成；如需变更，**新建 `test_suite_v2`** 并保留 v1。
- 每次评测必须记录：套件名、`manifest.csv` 的哈希、权重 SHA-256、`imgsz`、conf、NMS IoU。
- `../src/build_test_suite.py` 已内置**拒绝覆盖**保护（目录已存在即报错退出）。

---

**再次声明**：本套件与本文由 **DeepSeek**（`deepseek-flash`，DeepSeek Harness）自动生成，**仅供参考**。
类别语义假设（§4.1）、域差异（§4.2）与许可（§4.3）**尚未经人工确认**，
任何据此得出的模型结论都应在人工复核数据后再引用。
