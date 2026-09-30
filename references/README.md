# 有用参考资料与使用范围

| 参考 | 用于本项目 | 版本/边界 |
|---|---|---|
| [Kendryte K230 SDK](https://github.com/kendryte/k230_sdk) | 大小核结构、工具链、MPP、模型运行 | 本机历史构建 SDK v1.6，不能假设上游最新版适配本套件固件 |
| [K230 官方文档](https://github.com/kendryte/k230_docs) | AI、视频、VO/VB 与部署接口 | 与本机 SDK、厂商固件核对 |
| [RT-Thread 源码](https://github.com/RT-Thread/rt-thread) | RT-Smart 系统、驱动与进程接口 | 本板定制版本的行为以实机记录为准 |
| [Ultralytics 训练](https://docs.ultralytics.com/modes/train/) | 训练参数、续训、增强 | 本机固定 8.4.70 |
| [检测数据格式](https://docs.ultralytics.com/datasets/detect/) | YOLO 类别和归一化标签 | 正常灶火仍标 fire；火警事件另标 |
| [验证](https://docs.ultralytics.com/modes/val/) | P/R、mAP、框级验证 | mAP 不能替代每小时误报或逐事件验收 |
| [导出](https://docs.ultralytics.com/modes/export/) | ONNX 导出入口 | nncase/KModel 另有版本与量化核查 |
| [gc_kitchen_annotation](https://universe.roboflow.com/sojib-ldk6u/gc_kitchen_annotation) | 灶火、锅下火、反光与未点燃炉位 | 原 v2 元数据 CC BY 4.0；增强数量不等于独立厨房数 |
| [Home-fire](https://github.com/PengBo0/Home-fire-dataset) | 室内火/烟候选和来源核查 | 上游 CC BY-NC 4.0；逐图背景与二次来源需审核 |
| [Wikimedia Commons](https://commons.wikimedia.org/) | 蓝焰、蒸汽、无火诊断照片 | 每图作者、来源、许可保留在本地 manifest，不以站点许可代替每图许可 |

官方 SDK 与训练/验证文档入口于 2026-09-30 核实；数据许可栏目来自本地已收件版本记录。本地手册、选题指南和原理图见 [资料清单](LOCAL_MATERIALS.md)。当前公开资料包含工程笔记、失败记录和可复现脚本；未确认再分发许可的原手册通过索引保留。

## 本项目最有用的经验

- KPU run 成功后仍需检查后处理、框位、摄像画面和退出；一次局部耗时不是端到端延迟。
- ONNX 与 KModel 必须使用相同输入预处理比较；校准图需按来源去重并覆盖场景。
- VO/VB 显示成功需现场看屏；串口无异常仍可能乱码。
- 相邻视频帧与重复拷贝只是训练权重，不能增加独立样本数。
- 正常烹饪蓝焰应该有火框，但火框本身不触发火警。

已收录一张许可核实的 [CC0 蓝焰参考图](illustrations/README.md)，其模型数据身份保持待审。

第二批蓝焰参考原作、署名、许可及标签审核见 [开发候选记录](../model_work/docs/BLUE_SOURCE_QUEUE2_REVIEW_20260930.md)。下载记录与训练数据分开，原作重复不能因缩略尺寸不同而算新来源。


新增两段许可核实的公开灶台短视频，原件保留本机；来源及冻结探查边界见 [短视频探查](../model_work/docs/PUBLIC_VIDEO_PROBE_20260930.md)。煮饭片段的冻结无可见火焰探查已完成，船上灶台仍待真值判读。
