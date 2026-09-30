# 蓝焰来源与食物难负例数据对照 v9｜2026-09-30

v8 冻结主干保持了旧图诊断，但蓝焰定位仍为 0/5、难例适配不足，拒绝替换。v9 从 v3 出发，回到 v7 的不冻结配置，仅改变开发数据组成，检验补充灶型、视角和火力的蓝焰及食物无火难例能否改善。

## 已审核的数据

父数据 v7 manifest SHA-256：0a40ce9345652e322c8af0cd2d8a5abb1831e94432065c4b546828d64ee47a3b。新的 manifest SHA-256：7a262442e6227a0da6277b03c3b9211e65effc3d8e3e68e71995922fc636ecc2；最终审核 SHA-256：22fba7807d007ce7e4d1cfd3bebe70439d8d7191a60a65a2efd4e120c40da76e。

- 2382 份旧训练图继续使用原件；46 个父数据派生图/标签复制后与原件身份及权重一致。
- 加入队列第二批 8 张近似框审核照片，共 11 框，每张采样权重 12；7 个暂定场景组不是 7 个已证明的独立厨房。
- 加入旧视频帧的 1 个食物无火裁剪，权重 8；它不是新的独立来源。
- 两张模糊/弥散照片 queue2_04/06 保持暂缓，没有空标签入集。
- 共 2437 个文件、2782 个采样条目、55 个非旧基集派生文件，全部 2437 个标签均绑定哈希。

build_reviewed_source_queue.py 校验父数据审批及实际文件，检查新图字节/原作重复、已有诊断原作曝光、训练链审计、源元数据与标签；负例还逐像素核对裁剪和原图，检查裁剪不交叠旧火焰 GT。人工可见火焰判断不能由“不交叠 GT”自动代替，最终新 9 图已由 Codex 目视复核，未另取用户标签确认。

source_reviews 保存取得记录、源标签审批、重叠审计、负例审批和父审批的快照；这些实际文件及新旧审核图在训练前再次核验。初始 review_approval_pending.json 保留未批准状态，最终 review_approval.json 与 preflight_verified.json 分开留证。

首个构建因 queue2_08 的 license_url 差异在创建输出目录前拒绝：原 API 元数据为空，既有审核已核实原作页的公有领域声明。改为只允许有依据的署名更正，以及同一原作页 #Licensing 的公有领域空链接补全；原始取得记录未改，raw/reviewed/basis 保存在 lineage。不得因此放宽图像、原作、尺寸或许可类型身份检查。

公开来源/数据身份见 [数据清单](../../docs/hardcase_v9_sources_data_identity.json)；复制审核图、原始画面和权重留本机。

## 实际配置与复现

起点仍为 v3，SHA-256 48f284f9094fe334e02c66647f95fed8bfb3396b09f4488753508f514ed38980。seed=20260930、640、batch=4、AdamW、lr0/warmup_bias_lr=0.00002、最多 12 轮、patience=8 及所有增强与 v7 一致，freeze=None。已核对实际 args.yaml：仅 data、name、save_dir 不同。实际早停轮数可能变化，需在结束后报告。

```powershell
python -B model_work/src/build_reviewed_source_queue.py --parent model_work/data/hardcase_v7_sources_20260930 --blue-queue model_work/data/blue_source_queue2_20260930 --blue-audit model_work/out/blue_source_queue2_overlap_20260930/summary.json --negative-queue model_work/data/hard_negative_queue_20260930 --out model_work/data/hardcase_v9_sources_20260930 --blue-weight 12 --negative-weight 8
python -B model_work/src/train_hardcase_v5.py --data model_work/data/hardcase_v9_sources_20260930/data.yaml --start model_work/runs/fire-binary-user-video-v3/weights/best.pt --expected-sha256 48f284f9094fe334e02c66647f95fed8bfb3396b09f4488753508f514ed38980 --out model_work/runs/fire-binary-hardcase-v9-sources-20260930 --epochs 12 --lr0 0.00002 --warmup-bias-lr 0.00002
```

构建后仍需目视核对最终渲染、记录审批并通过 preflight 才能训练；命令不是自动审批。脚本拒绝覆盖已有目录。训练身份记录新增加本地数据校验依赖脚本哈希，记录加强不改变梯度或训练配置。实际身份见 [实验 JSON](../../docs/hardcase_v9_experiment_identity.json)。

v9 已完成 9 轮早停及全部顺序评测。训练日志 model_work/out/hardcase_v9_sources_20260930/training.log；已核实 PID 的 runner 顺序比较 v3/v9 best/last 的全部 55 个派生图、五图蓝焰、旧 286 图、按原作分组 Commons 和九段开发视频，并在同一 55 图补测 v7 best/last。新增训练图上的表现不能当新来源泛化结果。

原训练集并非仅含红火，既有 Flame 类记录含蓝焰/橙焰，早期还发现过漏标蓝焰。现有颜色像素排名只用于抽样审查，不能当蓝焰真值数量。此轮检验的是开发来源与难负例组成，不声称已证实唯一原因；持续关注锅下遮挡、框质量和误检。独立厨房完整事件、同步传感器、量化和实机验收仍未完成。

## 完整结果与决定

本机训练退出码 0，对比 runner 退出码 0；results.csv 完成 9 轮，最后累计训练时间 1472.13 秒。实际参数与 v7 仅 data/name/save_dir 不同，两轮均实际 9 轮，但每轮样本数不同，不等于梯度更新步数相同。best SHA-256：938d70b4ff7d6b6c0751d7c60f71099e61101db187f9356de36d02f4a306f5f7；last：1ead79c990f0121dc779cee97a155f779deeea428a32d0cc85e56b66be5fe867。

conf=0.25、匹配 IoU>=0.5。以下都是开发诊断；TP/FP/FN 是框数量，不能作为完整事件准确率。

| 诊断项 | v3 | v9 best | v9 last |
|---|---:|---:|---:|
| 五张蓝焰主要区域定位 | 0/5 | 0/5 | 3/5 |
| 旧 286 图 mAP50 | 0.9732386 | 0.9502558 | 0.9004365 |
| 10 张队友正例 TP/FP/FN | 3/5/7 | 4/4/6 | 10/3/0 |
| 10 张贴纸/下部负裁剪误检框 | 3 | 2 | 3 |
| 10 张 Commons 训练照片（13 框）TP/FP/FN | 1/3/12 | 1/2/12 | 7/2/6 |
| 1 张食物无火裁剪误检框 | 0 | 0 | 0 |
| 旧视频 24 个训练帧 TP/FP/FN | 18/0/0 | 18/0/0 | 18/0/0 |

同一批 55 派生图上的 v7 last 参考：Commons TP/FP/FN=2/1/11，队友=10/2/0，负裁剪误检=2。v9 的来源扩充改善了部分蓝焰定位，但多出误检；不能把训练图改善当作新厨房泛化。

五图 v9 last 命中 blue_bogtar、blue_ka23、blue_oven，IoU 分别 0.84724/0.85944/0.86218；blue_cardoner、blue_stovetop 无框。目视复核新增命中 ka23/oven 的框确实包围可见蓝焰。v9 last 的 teammate_f0cb3eb_1608 同时框到锅下蓝焰和蓝色贴纸；teammate_82f51e3_666 此次未框锅内食物，但一个曝光帧变化不足以证明食物误检已解决。

来源分组的旧 Commons 13 图：v9 last 在两个本轮训练蓝焰原作上定位 1/2，在两个未入此训练清单但已曝光的蓝焰诊断原作上亦为 1/2；无火三图 conf=.25 有 1 图误检，.5 为 0。不能与五图重复相加，亦不能称盲测。

决定：拒绝替换 v3。五图 3/5 是目前开发对照中的方向性进步，不抵消旧图退化和贴纸误检，也没有独立厨房/完整事件验收。完整本机 comparison_complete.json 与公开 [结果身份](../../docs/hardcase_v9_comparison_results.json) 已保留。

与此同时，[原训练抽查](TRAINING_COLOR_QUALITY_REVIEW_20260930.md) 确认原厨房来源已有多张蓝焰，通用来源继承漏标/可疑烟火语义。下一轮先保持 v9 所有新增来源、标签、权重和参数，检验移除整个 generic_fire 来源的影响；这不是宣称该来源 800 图全错，也不等同修复三处漏标。旧框口径问题另行审核，不混入这个单因素实验。
