# 模型数据与标注口径推进｜2026-10-02

v12 已完成且拒绝替换 v3。本阶段停止追加来源采样上限，转向可见火焰框和烟雾标签语义。用户持续目标仍有效，当前无开发板连接。

## 运行前确定的范围

1. 保留 v3、v5–v12 与全部原数据；从已有 train 抽查中，复核六张标注含大块锅体的灶火图，依据实际可见火焰给出明确的新框口径，不能依模型预测生成标签。新修订另建目录和身份清单，不触碰旧 val/test 或把新标签得分混成旧 mAP。
2. 下载 [D-Fire 官方 README](https://github.com/gaia-solutions-on-demand/DFireDataset) 链接的 [Kaggle 镜像版本1](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo/versions/1)，先实际核验与抽查，不自动加入训练、不使用其 split 冒充独立厨房测试。原作者仓库 commit 固定 `4bf9c31b18fadcd44d5f0b6d66f82bc56fa5e328`；其 LICENSE 描述集合 CC0且不持有所收集图片版权，许可边界保留。

公开元数据实际返回：id=6556263、ref=sayedgamal99/smoke-fire-detection-yolo、version=1、lastUpdated=2025-01-27、totalBytes=3118334483、licenseName=CC0: Public Domain。镜像自己说明增加 validation split，因此不是未经变化的官方划分。

下载 GET 响应：application/zip、3049605157 字节、ETag=`9745236834d65441af996cb6da6a4fde`，重定向到公开 storage.googleapis.com 对象。HEAD 返回404，GET实际200，不把HEAD失败当下载不可用。初次元数据控制台输出因GBK无法编码表情失败，ASCII JSON重查成功，未进行重训/重下载。

目标目录 `D:/SRTP_Datasets/dfire_kaggle_v1_20261002` 必须新建，预计下载+解包约6.2GB，最低剩余空间12GB；不清理原文件。单次下载最长1800秒、HTTP读超时45秒，收到字节/ETag-MD5/本地SHA/ZIP CRC逐级核查；异常保留partial与身份记录，不自动重试。解包总大小上限8GB、成员上限50000，拒绝目录穿越、Windows特殊路径、大小写碰撞、符号链接和覆盖。

```powershell
python -B model_work/src/acquire_dfire_kaggle.py --out D:/SRTP_Datasets/dfire_kaggle_v1_20261002 --max-download-seconds 1800
```

工具使用匿名公开HTTP，不登录或读取账户令牌；只采集，绝不自动启动模型训练。原图、归档、元数据全文和逐图审核留本地；公开代码、脱敏汇总、失败原因与模型实验记录。

当前状态：下载、全量结构检查、近似关联检查和本阶段逐图抽查已完成；来源未自动加入训练。详见下方实际结果及 [公开身份汇总](../../docs/dfire_source_audit_20261002.json)。

下载前核对 D 盘可用 156006236160 字节，requests=2.33.1。首次路径测试在 Windows 创建 ZIP fixture 时反斜杠被 ZipInfo 自动规范成正斜杠，导致非法路径用例没有实际进入验证器；修正 fixture 恢复原输入，再检查拒绝行为。这是测试构造问题，未解包外部数据或改变原文件。

## 实际获取与全量结构检查

下载进程已正常结束，`acquisition.json` 为 complete；下载用时96.54秒。收到3049605157字节，MD5与预定ETag一致，归档 SHA256：`65d81b602991261b47ddc71281e99108362a30336a448e1f8d21352c825c6d83`。43055个成员CRC与安全解包通过，解包数据3118334483字节，原图仍只留D盘。

镜像未附 data.yaml。类ID采用原作者 [固定版本 dfire.yaml](https://raw.githubusercontent.com/pedbrgs/Fire-Detection/9d99b0e71fd4cc930da053a41b79de1c58aabdf2/data/dfire.yaml)：原始0=smoke、1=fire，配置SHA256 `407e8eb8aa5a449224d6d4f8e6dd32bda0dcd3a021b9b801ca8db16c49f1477e`。实际抽查支持这一映射，不能混用本项目单类fire=0，也不把其他仓库配置当成该镜像映射。

全量21527图均可解码、无孤立标签。逐图保存图像/标签SHA、像素身份、尺寸、合法框、异常和pHash；不自动裁剪异常框或改写标签。

| 镜像原划分 | 图片 | 解析为空 | 仅smoke | 仅fire | 两类 | smoke框 | fire框 | 有结构异常图 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 14122 | 6458 | 3836 | 770 | 3058 | 7788 | 9631 | 224 |
| val | 3099 | 1375 | 845 | 174 | 705 | 1755 | 2176 | 54 |
| test | 4306 | 2008 | 1183 | 225 | 890 | 2303 | 2878 | 62 |

340图有结构异常：314图框越界、26图存在非法类别/坐标/字段。类别计数按成功解析行计算，异常行不参与，因此不强行对齐发布者宣传计数；解析为空也不等于人工确认负样本。当前按整图暂缓，未将异常行删除后当作完整真值。

字节和解码像素均未发现重复组；与两份已知数据清单6481个独立字节身份交叉比对，字节命中0。字节不同不能证明事件或视角独立。

## 近似关联与实际目视复核

pHash采用OpenCV灰度32×32 DCT低频8×8、AC中位数阈值、DC位清零，Hamming半径4。新增64位BK树查询与暴力搜索对照测试；对21527图完成全量查询：186119对相似候选，其中跨原划分94015对，8958图有直接跨划分候选；785个候选连通分量，591个跨划分，最大分量1258图。它们是审核线索，传递合并可能把无关图片纳入同组，不能按这些数字宣称重复率或场景数。

已有两份数据清单及三个既有诊断/派生图目录共6553个字节身份，与新来源得到37图/196对相似线索。未知预训练和未记录曝光未排除。

原pair文件仅存最先10000对，受train优先扫描顺序影响；所有计数和分量完整。逐图审核没有从这一偏置文件选跨划分例子，而是在随机选中的完整分量内重建实际最近跨划分边。固定种子20261002，共36项、9张审核页：

- 12个随机选中的跨划分候选分量：12项均目视确认同一摄像视角或原作/视频来源关联，包括相隔5–10秒的帧，以及同视角但相隔约11分钟的帧。不能据12项推断全库重复比例，也不把同机位不同日期称为相同事件。
- 8张随机选中的已知数据相似候选：5项确认同摄像视角或原作/视频来源关联，3项为无关夜间火光、日落云等相似度误判。pHash不能自动赋予原作身份或标签真值。
- 原train四种类集合各4图：无目标样本主要为云/天气；正例支持0=smoke、1=fire的映射，但出现远处微小低对比羽流、饱和夜间亮区、框同时含大块羽流、白色羽流与蒸汽/云难区分等问题。另打开26/30/34/36号原图，仍有未决边界。16图不代表全库标签质量验收，未凭照片确认燃烧事件真值。

完整本地记录 `model_work/out/dfire_visual_review_20261002_v2/review_complete.json` SHA256 `49228cba395642d83970aa60473f02fed2e75f01ccc9f28904cbe2322e835e8f`，公开汇总只含编号、结论、计数和身份，不含原图或逐图私有路径。首次制图因给SHA函数传str而非Path报错，修正后另建v2目录，保留失败目录；未重下载或更改来源。

## 可见火焰框修订种子

从此前48图抽查中重新打开6张存在大块锅体框的原train图片，按每个燃烧器一个近似可见火焰包络框处理，排除上方大块锅体和反光；遮挡时矩形内仍可能含非火焰像素，不宣称像素分割或火灾事件真值。新框由实际目视给出，不依模型预测；实例数保留。

5张/6框批准为候选标签，1张弱紫色火焰图在看过放大图后仍暂缓，未转成负例。旧框总面积351356像素，新框94027像素，仅表示标签范围变化，不能表示模型提升。已实际查看3张旧框洋红/新框绿色对照页，见 [脱敏修订汇总](../../docs/fire_envelope_review_20261002.json)。本地完整候选绑定原图、原标签、新标签、图页和审核SHA；尚未加入任何训练版本。

这是1045张灶火train中的5张修订种子，不能声称全部口径已统一。旧train/val/test和v3/v5–v12均未修改。

## 后续实验门槛

1. 先扩大灶火框审核覆盖，建立独立修订版本；改变标签后的分数与旧标签回归分开记录。5张修订不足以说明解决整个锅体口径问题。
2. 从D-Fire准备小规模可见烟火开发候选，优先逐图审核清楚的素材；结构异常图、与已曝光来源有关联的候选和未决标签暂缓。来源框原样保留，新标签必须有审核身份。
3. 新烟雾实验采用明确的可见羽流/烟雾视觉定义，不能单图区分的蒸汽/油烟不得被写成燃烧烟雾验收真值。保留v3，分别比较其火焰回归与新增烟雾能力。
4. 不沿用镜像val/test作为独立厨房验收；不能靠pHash图分量重新随机切分就声称厨房/事件独立。独立测试≥20%、真实厨房≥30%、两类F1≥.90及长时误报警仍需另外证据。

本阶段没有启动新训练或晋升模型。用户持续目标保持active；这是为下一轮模型实验准备数据，未完成计划书整体模型验收。

## 本机验证与复现入口

82项本机测试全部通过（含3项ZIP路径/覆盖检查、2项pHash索引测试）；此前77项模型、推理、父子数据身份和融合测试仍通过。CI环境不装PyTorch/NumPy/OpenCV，相关可选测试会明确跳过，不把基础CI说成GPU或完整推理验证。

各输出目录必须尚不存在；再次复现用新目录，不能覆盖本阶段记录。Windows本机Python为 `C:/Users/ASUS/miniconda3/python.exe`。采集命令见上方，后续实际运行入口：

```powershell
python -B model_work/src/audit_dfire_inventory.py --acquisition D:/SRTP_Datasets/dfire_kaggle_v1_20261002 --known-data D:/SRTP_Datasets/kitchen_vision_v3_20260926 --known-data D:/SRTP_Datasets/kitchen_fire_binary_codex_20260927 --out model_work/out/dfire_inventory_20261002
python -B model_work/src/audit_dfire_perceptual.py --inventory-dir model_work/out/dfire_inventory_20261002 --known-data D:/SRTP_Datasets/kitchen_vision_v3_20260926 --known-data D:/SRTP_Datasets/kitchen_fire_binary_codex_20260927 --known-dir model_work/data/hardcase_v11_negative_curation_20260930/images/train --known-dir model_work/data/commons_blue_review_20260927 --known-dir model_work/data/manual_kitchen_labels_v2_20260927/commons_candidates --out model_work/out/dfire_perceptual_20261002
python -B model_work/src/prepare_dfire_similarity_review.py --inventory-dir model_work/out/dfire_inventory_20261002 --perceptual-dir model_work/out/dfire_perceptual_20261002 --out model_work/out/dfire_visual_review_20261002_v2
python -B model_work/src/prepare_fire_envelope_revision.py --source-review model_work/out/training_color_review_20260930_v2/review_complete.json --proposals model_work/out/fire_envelope_review_20261002/proposals.json --out model_work/out/fire_envelope_review_20261002/proposal
python -B -m unittest discover -s model_work/src -p 'test_*.py' -v
```

制图只生成pending记录，必须实际查看原图和图页再填写人工结论；不自动复现人类语义判断、批准训练或启动实验。本次填录脚本和逐图完整决策留在本地输出目录；公开汇总绑定其完成记录SHA，不能只复制脱敏编号作为训练许可。
