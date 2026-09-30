# 视觉数据集整理记录（2026-09-26，v2）

## 目标和边界

用户要求：正常燃气灶火能被识别，终端能显示“正在做饭”；短时爆炒/锅内窜火能与持续异常起火区分；异常起火触发报警。SHT31 温湿度和未来其他传感器只在状态融合阶段使用，不作为静态图片的真值。

视觉数据分两层：

1. **目标检测**：框出可见火焰，必要时框出灶具/锅具。无论火焰是否危险，都应被检测。不要把普通灶火标成“无火”。
2. **场景与时间序列**：`灶火正常`、`短时爆炒/窜火`、`异常持续/蔓延`、`无明火/不确定`。仅凭一帧的大火通常不能判定“爆炒”或报警；需要火焰位置、与锅/灶的关系、持续时间与变化，之后再结合传感器。

第二层的标签必须来自有前后过程的视频或可核验事件背景。截图、文件夹名和旧模型输出都不够。

## 本机原始数据审计

原始根目录：`C:\Users\ASUS\Documents\Codex\kitchen-fire-detection\datasets`。本轮仅读取它们，没有修改。审计脚本：`model_work/src/audit_vision_sources_v2.py`，证据：`model_work/data/vision_sources_v2_audit_20260926/local_sources.csv` 和 `summary.json`。

| 来源 | 图数 | 当前用途 |
| --- | ---: | --- |
| `fire-flame.v1i.yolov11` | 9,389 | 通用可见火焰候选；原许可标为 `Private`，需核实上游授权；并非已证实的厨房灶火集 |
| `stove-detection` | 512 | 灶具框候选，类别映射待核 |
| `pan-detection` | 410 | 锅具框候选；源类别 0/1，不能不检查就合并 |
| `kitchen-images` | 64 | 全部人工复核队列；旧伪标签暂停使用 |

总计 10,375 张。字节级 SHA-256 相同的图有 153 组、306 个文件；按 Roboflow 文件名前缀识别的同源图跨原 train/valid/test 有 1,129 组。这个数字是**可疑同源组**，不等于已证明 1,129 组像素完全相同；不能沿用原切分作为独立测评。

64 张厨房图中，`normal_cooking` 有 11 张带旧 fire 框，不能由目录名推断框必错；`big_flame` 有 16 张旧空标签，已经看见其中一张确有大幅锅上火焰；另有 1 条越界框。抽看一张 `kitchen_fire` 图是室外模拟救火场景，不能按文件夹名当成真实厨房失控起火。`iStock`/`VCG` 等库存图片先标记授权待查。原 64 张已进旧模型训练，不能作为旧模型独立测试。

## 本轮生成的整理索引

脚本 `model_work/src/build_vision_curation_v2.py` 产生：

- `model_work/data/vision_curation_v2_20260926/generic_fire_candidate_manifest.csv`：从原 `fire-flame` 以 SHA-256 去重后保留 9,237 张；用同源文件组和跨名完全重复图组成不可拆分组，再重新分配 train 7,276 / val 954 / test 1,007。脚本断言三组无相同哈希和相同组 ID。**这是候选索引，不是可直接开训的已核准数据集**。
- `model_work/data/vision_curation_v2_20260926/kitchen_visual_review_queue.csv`：64 张全部 `hold`，附旧标签问题、授权风险和待复核字段。没有把任何旧伪标签升级为真值。
- `model_work/data/vision_curation_v2_20260926/curation_summary.json`：上述数量与状态。

重新切分仍需补充感知哈希/视觉近重复审核；SHA-256 只能抓字节相同，文件名同源规则也不能抓所有视频相邻帧。正式验证必须按拍摄视频/地点/事件分组，不能让同一段视频的邻近帧同时进训练与测试。

## 网上找到的来源及处置

| 来源 | 与目标的关系 | 当前处置 |
| --- | --- | --- |
| [Roboflow gc_kitchen_annotation](https://universe.roboflow.com/sojib-ldk6u/gc_kitchen_annotation) | 1,086 原图，四类含燃气灶火、锅下火、火焰反光、无火；标示 CC BY 4.0，最贴近“正常灶火”缺口 | **用户已下载并解压 v2**，收件与机器审计见下文。尚需抽检更多图的类别语义与标注质量，不能直接开训。v2 的 1,848 是含增强后的版本数，不是 1,848 个独立场景。 |
| [Home-fire Dataset v1](https://github.com/PengBo0/Home-fire-dataset) | 6,500 张室内火灾/烟雾，YOLO 标签；CC BY-NC 4.0；不保证都是厨房 | **已下载并校验 val.zip** 作为抽检样本。抽检后仅择与厨房相关的异常火样本，不能把全部室内火混为厨房正例。上游图像还有二次来源，保留出处和署名。 |
| [CMU Kitchen Capture anomalies](https://kitchen.cs.cmu.edu/anomalous.php) | 三名被试、五道菜，三明治/煎蛋有突然起火和烟；带视频过程 | 适合作独立时间序列评测候选。还未取得并查看具体片段；不要先声称可训练。 |
| [DetectiumFire](https://github.com/ZixuanLiu4869/DetectiumFire) | 论文描述有受控烹饪火和失控厨房火 | Kaggle 包约 102.6 GB；许可元数据 CC BY-NC-ND 4.0 与论文文字有差异。**暂不整包下载/并入训练**，先核许可、文件清单及可否单取厨房部分。 |
| [Indoor Fire Smoke, Zenodo](https://zenodo.org/records/15826133) | 5,000 张室内火/烟，约 200.5 MB | 公开页面未给出明确许可证，暂不并入训练。 |
| [Hugging Face gas-burner collection](https://huggingface.co/datasets/baizhanquan/FireDetectionDataset-flame-indoor-gas-burner) | 名称相关 | **排除**：仓库明确写着目前为空，0 张接收样本。 |

## 下一阶段具体步骤

1. 抽看 Home-fire 的正常/异常/易误报场景；按真实厨房相关性筛选，不因名字入选。
2. 对已收到的 Roboflow v2 检查更多蓝色小火焰、锅下遮挡和反光难例；先明确 `no-flame` 类是显式标注的“无火炉位”框，不能机械当成整图空标签。制定类别映射后再训练。禁止把“有火但受控”标成负例。
3. 人工复核本机 64 张厨房图并重做火焰框；对于单张无法确定异常性质的图填 `不确定`，不硬判。
4. 建立独立测试集：按视频/场景组划分；正常蓝焰、锅下被遮挡蓝焰、爆炒短时窜火、失控持续火、蒸汽/反光/LED 难负例都要有覆盖。当前没有完整的爆炒视频真值，不能声称已能识别爆炒。
5. 固定类定义与清单后交 DeepSeek 训练；先训练通用 `flame` 框（可选 `stove`/`pan` 上下文），再单独开发时间状态规则。以板端同一摄像头视角评测漏检、误报、延迟；不能只报旧 val mAP。

目前**不需要用户拍火或改动开发板**。灶火数据集已由用户下载，后续核标与合并由开发侧处理。

## 本轮在线文件实收记录

- `model_work/data/online_sources/home_fire_v1_val.zip`，352,482,306 字节，SHA-256 `ffe0444340b3baab129e24b75f1a67ec489f072be85ea7b1c2cc4c5ce7f6ef63`。公开 GitHub Release v1.0.0 的 `val.zip`，下载脚本 `model_work/src/fetch_home_fire_val.py`，只创建新文件，下载大小不符会拒绝。
- `model_work/src/audit_home_fire_zip.py` 只读核验：ZIP 完整；1,300 张图与 1,300 份标签一一对应；其中 33 份空标签，源类别 ID 0 有 963 框、ID 1 有 617 框，无字段数/坐标范围错误。**尚未核实类别 ID 对应名称，也尚未人工抽看场景，因此 0 张批准并入训练**。
- Roboflow 优先集已解压在 **`D:\gc_kitchen_annotation.v2-version_two.yolov11`**。`data.yaml`：YOLOv11，v2，CC BY 4.0，类序 `0 flame-reflection / 1 flame-under-pot / 2 gas-stove-flame / 3 no-flame`。`README.roboflow.txt` 说明 640×640 拉伸及训练增强。本轮只读审计脚本 `model_work/src/audit_gc_kitchen.py`：train 1,524 / valid 215 / test 109，总 1,848；按 `.rf.` 前缀归并为 **1,086 个源文件名组**，其中 762 组各有两个版本；四类框数分别为 1,541 / 1,524 / 2,462 / 1,526。图和标签配对齐全、无空标签/越界、无字节级重复，未发现源文件名组跨 split。此检查**未发现问题**不等于全部人工标注正确，也尚未核验同场景连续帧是否跨 split。
- 抽看 `train/images/DSC_4780_JPG_JPG.rf.2a0127756d2a01f9f7642916bd2876d2.jpg`：是真实商用厨房，有明火、锅下火和未点燃炉位；对应标签文件同时出现类 1、2、3，语义初看合理。但只抽看了这一张，不把全数据标为通过。
- **2026-09-26 补审独立性：**train/valid/test 依源图名前缀有 762/215/109 组。test 类 0/1/2/3 分别覆盖 59/79/77/45 个源图组，同组多类可重叠；它们仍可能来自同一拍摄地点/连续照片，不等于独立厨房现场。训练报告必须另列拍摄场景数，不足的分项标“未验收”。
- **拟定映射的只读预演：**仅保留类 1、2 为目标 `fire` 后，train/valid/test 分别有 3,284/468/234 个 fire 框，空标签图均为 **0**。因此删去反光与 `no-flame` 框没有产生纯无火整图；此数据缺少独立全图负例。构建脚本必须输出映射前后的类别框、空标签图及源图组统计，另补经审核的纯无火厨房难例。
- **原数据备份已完成：**`C:\Users\ASUS\Documents\ChatGPT\SRTP\artifacts\dataset_backup_2026-09-26_gc_kitchen_v2`，共 3,699 个原文件、76,254,966 字节；`MANIFEST_SHA256.json` 逐文件记录 SHA-256，复制后逐个与 D 盘原件核对一致。源与备份都不得作为派生训练集的输出目录。备份脚本：`model_work/src/backup_gc_kitchen_v2.py`。

## 当天 Wi-Fi 记录索引

主板 Realtek 8188FU 的 `wlan0` 今天已扫描、连接热点、取得 DHCP 地址，并验证网关、外网 IP 与域名连通；详细命令、实测 IP、手册对照和边界已记在 `2026-09-23_RT-Thread_Smart_AI套件实机验证记录.md` 第 25 节。此次通过的是**主板 Wi-Fi**；副板 RW007 与 OLED 的 `WIFI --` 实时状态还未跑通，也没有验证断电自动联网。热点密码不写入文档。
