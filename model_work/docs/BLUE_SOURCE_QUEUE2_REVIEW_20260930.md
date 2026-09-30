# 第二批蓝焰开发候选｜2026-09-30

为了继续增加火焰颜色、视角和灶型覆盖，从 Commons Gas flames 分类中列出 10 个候选原作。获取脚本先按原作页排除已知 18 个诊断作品，再下载到独立待审目录。本批尚未加入任何训练集，v7 正在使用原有两份诊断原作。

## 已完成审核

- 下载 10 张，逐项保存原作链接、原始 SHA-1、实际图像 SHA-256、下载尺寸、署名与许可。
- 8 张人工近似标签通过最终叠框复核，共 11 个火焰框；其中一张有四个灶头分别标注。
- 同一 Susan Slater 灶头低/高火力两张合成一个场景组；Arivumathi 的俯视/侧视两张也归同组。批准的 8 张共 7 个暂定场景组，不代表已经证明 7 个独立厨房/事件。
- queue2_04 强失焦与反射、queue2_06 弥散蓝色边界不确定，两张暂缓；不是无火负例。
- queue2_00 的 Artist 机器字段缺失，通过 [原作品页](https://commons.wikimedia.org/wiki/File:Cooking_with_gas.jpg)补齐署名 sfllaw；queue2_08 按[原作品页](https://commons.wikimedia.org/wiki/File:Natural_gas_burning_on_a_gas_stove.jpg)核对作者用户名 WrS.tm.pl 及其公有领域发布声明。

来源、标签、许可、场景组与审批状态见 docs/blue_source_queue2_review_20260930.json。原照片和叠框图仅保留本机，公开仓库发布审核记录与原作链接。

## 重叠检查

对可获得的两条训练链 5942 个去重图像逐字节重新核实 SHA-256，并计算 pHash；本批 10 张无相同字节匹配。另与旧 18 张诊断图做原作身份和 pHash对照：本批无同原作页，诊断最近距离为 20–28，训练最近距离为 12–18。

最接近的 queue2_05 与训练图 distance=12 已目视对比：前者是双环蓝色燃气灶火，后者是户外橙色火焰容器，明显不同。pHash不能据此证明没有其他连拍、裁剪或初始化链未知数据。范围与最近图身份见 docs/blue_source_queue2_overlap_20260930.json。

## 复现与下一步

```powershell
$env:PYTHONUTF8='1'
python -B model_work/src/fetch_commons_source_queue.py --titles-json docs/blue_source_queue2_titles_20260930.json --out model_work/data/your_new_source_queue
python -B model_work/src/audit_blue_candidate_overlap.py --candidates model_work/data/your_new_source_queue --base D:/SRTP_Datasets/kitchen_fire_binary_codex_20260927 --initializer-data D:/SRTP_Datasets/kitchen_vision_v3_20260926 --out model_work/out/your_new_overlap_audit
```

下载和审查目录都拒绝覆盖。一次临时作者表输出因 Windows GBK 无法编码 ł 而失败，文件已生成；用 UTF-8 重读核实，没有重抓或替换图片。上述 UTF-8 环境设置保留在复现命令中。

先等 v7 的按曝光范围分组对照完成，再决定是否增补这些来源及训练起点/日程。批准标签不代表候选已训练。后续任何划分必须按同场景组归边，重复权重不能增加独立样本数。这批已被查看并准备开发标签，不能以后冒充最终盲测。
