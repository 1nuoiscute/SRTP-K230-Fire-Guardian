# 编码：烟雾开发样本扩充与固定特征对照训练

请根据附带源码交付两个完整 Python 脚本，各用一个 python 代码块，并在代码块前写脚本文件名。主控负责实际查看图片、审批标签并运行；你不得生成任何人工批准。

1. prepare_smoke_extension_review.py：复用 prepare_dfire_pilot_review.py 的结构/关联隔离逻辑，argparse --inventory-dir --perceptual-dir --exclude-review --out，exclude-review 指先前 review_complete.json；先排除之前全部 images 的 sha256 与 candidate_group（包括 held），再按固定 seed 20261028 SHA 排序挑选 12 张 raw classes=[0] 的 smoke 候选与 12 张 raw classes=[] 的无烟候选。只取源 train/WEB/无结构错误/无跨split或known_similarity关联；每 component 最多一张。不要挑新val/test。渲染每页4图的原图/框对照共6页，写 review_pending.json（沿用原来的字段），候选一律manual_review=None。不要自动批准或下载。原文件只读。保留完整 boxes 和 source sha。

2. train_smoke_extension_v5.py：argparse --data --manifest --start --expected-start-sha256 --out。读取自定义 JSON manifest（字段 images 列表，每条 file/split/image_sha256/label_sha256/group；data.yaml 为单类 smoke，train images/train、val images/val）。遍历这些文件核对sha、group不跨split、split只train/val，保留原图。用 start 作为起点，训练配置精确沿用 train_independent_smoke_head_v4.py 的40轮/AdamW/seed/增强/batch/nbs/BN冻结/EMA同步与实际更新审计；禁止覆盖输出。用 FrozenBackboneAudit 和 OptimizerCadenceAudit，不用旧 verify_smoke_dataset，因为新manifest另有扩充来源。meta保存manifest哈希、起点哈希、完整config、实际step/batch、shared live/EMA审计哈希、两个checkpoint哈希和环境。本轮 start 将用 smoke v4 的原始迁移起点，新增样本只加入train，旧9张val成员完全保持。当前只评可见烟雾外观，不能将未知烹饪羽流赋负例，不能声称火灾事件真值。

优先最小必要代码，不加测试框架、重构或复杂抽象。不要改变附带原文件，交付新脚本。
