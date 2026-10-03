# 厨房来源队列4：新增火焰与跨批次留出保护

2026-10-03完成数据审核，尚未运行本批任何模型预测或新训练。演示保留v16 best，定位开发保留v18 last；开发板未连接。此阶段直接补充不同来源的火焰与厨房干扰，不能当作识别率提高。

## 取得与实际查看

预先列出的13作品中取得11张JPEG，包含队列3中3次HTTP429的明确重试，三项均成功。请求最小间隔2秒，实际取得源码快照SHA `42056ae80452f97ed854f42e6a3e774238a9434207fa39b4c7fe0315f7736695` 留在本地。旧失败记录不修改。

另外两项未下载：Bakken的API许可名为CC BY-SA3.0，但URL是HTTP，被当前只收HTTPS许可URL的策略隔离，**不表示判定作品不可用**；Coker为Public domain，不符合当前自动许可白名单，留待单独人工核对。没有放宽策略或悄悄换图。

实际查看11原图、11最近参考页、5放大图、5真值叠加图、2真值放大图及1幅旧作者参考，共35文件，逐个记录SHA。原图保留全部画面。所有下载图没有待应用的旋转EXIF（04为orientation1，其余无标记）；03的下载像素已经竖向，不重复旋转。请求宽1600不代表实际尺寸，按下载像素标框。

## 用途与真值冻结

| 用途 | 图片 | 可见火焰图/区域 | 无可见火焰图 | 内容 |
|---|---:|---:|---:|---|
| 开发 | 7 | 3/3 | 4 | 铜锅下蓝焰、工业灶内小蓝焰、单炉头蓝焰/橙色火舌；食物、反光壶、挂置传统炊具、产品灶具 |
| 预先留出 | 2 | 1/1 | 1 | Cocinafuego开放蓝焰炉头和Primalpia厨房锅具/橙色挂布 |
| 隔离 | 2 | 1/4，未复制标签 | 0 | 露营摩卡壶火焰可见性未知；BogTar四炉头与旧开发诊断同作者 |

露营炉发红区域放大后仍不能区分灼热金属与可见火焰，不作正例也不生成空标签。BogTar只有同作者及相关炉头外观证据，不断言同一拍摄，但不能据此增加独立厨房数；其4框只保存在审核记录，没有进入角色复制。

04/05/10/11每个可见炉头用一个轴对齐火焰包围框；环形或弧形火焰的包围框内部可能含金属/少量锅底，不额外扩大到整锅。不同炉头分别标框。空标签只表示无可见火焰，不表示无燃烧、无烟雾或无危险事件。

本批仅01/04具有灶台及厨房区域背景；传统炊具陈列不算正在使用的厨房，商用产品照片不算真实烹饪，其他近景也不伪装成全厨房。7开发组/2留出组/2隔离组都是暂定来源组，不能证明所有厨房相互独立。

## 重叠与继承防护

扫描25,766路径/7,601字节唯一图，核对15,484原CSV图片身份，明确包括队列3十张原图及27个已查看的项目视频源画面。11张新图均无相同字节/像素；最近pHash距离16–20，仅作为人工查看线索。全部最近两图已看，未确认相同作品或场景。38个旧Commons原作API身份均已解析，另外按API作者字符串核对，只发现BogTar关系并隔离。

新工具`freeze_inherited_kitchen_review.py`在复制前绑定继承方案和队列3审核SHA，将两批group/author/original SHA1联合核查。旧留出组的相关新图不得转为开发，审核文件或继承方案改变会拒绝；文件前缀context4避免跨批次文件名冲突。旧队列3构建器字节和训练v19数据manifest保持不变。

两批合计14开发图/13暂定组，其中4火焰图/4区域、10张无可见火焰；4留出图/4组包含2火焰区域；3隔离图。仍不足以完成计划书独立测试≥20%或真实厨房≥30%的全项目要求。

## 核验与下一步

174项本机测试全部通过、无跳过，包含6项跨批次继承检查及1项安全前缀检查。35查看文件SHA与9份角色分离图像/标签逐项核对，4个正式火焰框YOLO往返误差小于1e-5像素；隔离图没有生成标签。完整小汇总见[结果JSON](../../docs/kitchen_context_queue4_results_20261003.json)。代码测试与身份检查不计作模型准确率。

下一步冻结使用两批**开发角色**的平衡数据扩展实验，保留全部留出和隔离身份，比较锅下蓝焰、厨具/食物/反光及已有视频代价。当前新火焰开发图仍只有4张，扩充价值应通过实际比较判断，不单靠加权或训练轮数宣布进步。烟雾、蒸汽与危险事件真值仍另行推进。

## 复现

原取得、重叠扫描和角色冻结均拒绝覆盖已有目录，复现请使用新输出路径。大原图/权重/本地清单留在电脑，Git只提交源码与小记录。

```powershell
python -B model_work/src/fetch_kitchen_context_queue.py --plan docs/kitchen_context_queue4_plan_20261003.json --out <new-quarantine-directory>
python -B model_work/src/audit_kitchen_context_queue.py --queue model_work/out/kitchen_context_queue4_quarantine_20261003 --plan docs/kitchen_context_queue4_overlap_plan_20261003.json --out <new-overlap-directory>
python -B model_work/src/freeze_inherited_kitchen_review.py --queue model_work/out/kitchen_context_queue4_quarantine_20261003 --review docs/kitchen_context_queue4_review_20261003.json --audit model_work/out/kitchen_context_queue4_overlap_20261003/summary.json --inheritance docs/kitchen_context_queue4_inheritance_20261003.json --out <new-role-separated-directory>
python -B -m unittest discover -s model_work/src -p 'test_*.py' -v
```

重新取得网络素材可能返回不同字节，不能套用本次审核身份；须重新审核。网络源码快照与当前实现均保留，逐图实际查看不能由命令自动宣布完成。
