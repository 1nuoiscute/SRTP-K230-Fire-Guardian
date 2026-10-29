# 烹饪视频隔离收件与蓝焰初审｜2026-10-29

## 原件与隔离角色

来源 [Preparation-du-poulet-realisatrice-marcko-wiki-loves-africa-cote-divoire-fan-club.webm](https://commons.wikimedia.org/wiki/File:Preparation-du-poulet-realisatrice-marcko-wiki-loves-africa-cote-divoire-fan-club.webm)，作者 KoffiMARCKO，Wikimedia 页面标注 CC0。2026-10-29 Commons API 的原始文件大小173,694,134字节、SHA-1 `54fc49da22884fdaa3d4e2b6bdca13552899e763`；本机原件字节数、SHA-1 均核对一致，SHA-256 `5c7150580f8665eb7e20c6bd813f072b8a7d9f87a7c73ab09d6dbabd19073204`。原件保存在本机 `model_work/data/cooking_vlog_quarantine_20261029/original.webm`，不入 Git。

获取前已固定 `quarantine_plan.json`（SHA-256 `bbc694020c17be6f2c223837c1292b8cce655774dc28637992b77a843b9bed8e`）：整部视频先隔离，禁止训练和预测，不赋独立测试角色；真值、来源重叠和既有预测曝光待审。新增素材不是四张历史隔离预留源，后者未使用。

## 只看原画的初审

25 fps 元数据，顺序解码6597帧。先每250帧抽样，共27张画面，主控查看全部3页；然后对第4125–4400帧每25帧抽样12张，查看全部3页。第4225、4250、4275帧（约169–171秒）可见燃气灶蓝焰，最后一个样本有锅体遮挡。前后逐秒样本为字幕画面，视频是编辑后的多场景 vlog；不能将镜头前后推断成点火或关火，也不能用该片当完整固定机位事件。未批准定位真值框，也未逐帧观察全部视频。时间为帧序号/元数据 fps 估算，PTS 未核验。

本机证据：`model_work/out/cooking_vlog_intake_20261029/intake.json` SHA-256 `189e8c0562f6a05eb100477c7769df82ec8abed0950940cc67780b33c6953698`；`burner_source_review/manifest.json` SHA-256 `f08183550881b9042deedd8aac069352df6abe7155b5136faf47097804fc4390`；两组原画页均留本机，未包含模型框。

## 有限重叠核查

对三个可见蓝焰样本与 v24 使用的 v20 清单中1445张训练图做解码像素和 pHash 比较，逐图核对哈希；三个样本均无完全相同的解码像素，最小 pHash 距离18。仅限这三个样本和该清单，最近图未作目视复核；不能排除同厨房、裁剪压缩、通用预训练、其他历史训练链或预测曝光。`burner_source_review/v20_overlap.json` SHA-256 `7f782d2b02ec3959f8174720c4238814d3693d615f37ad6b7118815495197131`，实际运行脚本和结果都留在同目录；脚本也复制入 `model_work/src/audit_cooking_vlog_v20_20261029.py`。

目前这段可作为待核来源的蓝焰短镜头候选；它无法解决关火/无火事件缺口。后续需核查完整来源和曝光、明确厨房/事件分组并在看预测前冻结定位标注和用途。不能据此训练、选择模型、计算跨事件准确率或宣称 K230 验收。
