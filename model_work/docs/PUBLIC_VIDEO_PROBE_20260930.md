# 新来源公开短视频探查｜2026-09-30

在查看视频像素之前冻结 v3、v9 last、v11 best/last 四份实际权重 SHA 和 640/conf=.25/NMS=.6/目标5fps/匹配IoU=.5。本批用于小规模外部视频探查，不是跨厨房完整事件验收。原件已下载并核对 Commons/API/实际 SHA-1、字节数和许可，无视频预览、帧提取或模型推理；完整冻结计划与收件记录留在本机 model_work/data/public_video_probe_20260930。

| 原作 | 作者 | 许可 | 原件 |
|---|---|---|---|
| [Rice Cooking on a gas stove](https://commons.wikimedia.org/wiki/File:Rice_Cooking_on_a_gas_stove.webm) | Shraddhahistory | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | WebM，848×480，约7.1秒 |
| [Gimbaled stove](https://commons.wikimedia.org/wiki/File:Gimbaled_stove.WebM) | Fanny Schertzer | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) | WebM，748×420，约28秒 |

前者说明为燃气灶煮饭，后者为帆船上的万向支架炉灶；描述并不保证可见蓝焰或适合定量标注。尚不能宣称两段都是新厨房或蓝焰正例。公开 [冻结及收件汇总](../../docs/public_video_probe_preflight_20260930.json) 不包含本机权重路径。

接下来先查原作/场景是否和原训练链或旧诊断重叠，并对固定采样帧独立记录火焰真值；不看模型输出后再挑有利片段。明确不可判读区域，保留原片，逐来源报告定位、错误位置和连续性。四权重和阈值不依据这批视频再改；首次推理后本批转为开发曝光，未来调参不能再称盲测。

训练链的可获得范围有限，哈希/pHash相似性检查只能发现重复线索，不能证明场景完全独立；不足以用这两段短视频估计每小时误报或火警事件可靠性。不向代码仓库提交原视频或派生画面；如另行发布可视化，须保留作者、原作链接、许可及修改说明。
