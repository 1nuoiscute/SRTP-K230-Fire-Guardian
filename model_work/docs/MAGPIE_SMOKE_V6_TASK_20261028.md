# 依据 v5 结果实施烟雾后段特征对照 v6

请输出完整新脚本 train_smoke_partial_features_v6.py，单python代码块。基于附带 v5 脚本最小改动，其数据验证函数可以直接 import train_smoke_extension_v5.verify_extension。保留命令行 --data --manifest --start --expected-start-sha256 --out。不要修改附带原脚本。

实际v5已完成40轮/432次更新，旧9图val为2TP/1FP/1FN（v4为2/0/1），16图新增训练反馈1TP/0FP/3FN，固定特征下仍漏3个新烟雾目标，旧val新增的是橙色烟柱上的重复局部框。现做同数据、同迁移起点、同40轮日程的部分特征对照，除了开放module10–22参数外尽可能一致：freeze=10；全部feature module0–22 的BN running_mean/var/counter固定，后段BN affine可学；head23普通训练。固定DFL。

用FrozenBackboneAudit(model.model,10,...)检查前缀；on_batch_start 调该audit后将feature module0–22 所有BatchNorm2d设eval；on_epoch_end先查前缀不变，再查feature全部BN缓冲与DFL不变，并确保至少一个后段feature weight和head weight实际改变。EMA中复制固定前缀+所有featureBN+DFL后逐张量确认。计实际optimizer.step。最终best/last保存后核对固定状态相等和后段/head真实改变，meta写实际变化计数。保持 v5 的40轮增强seed学习率全部配置。保存源码快照与依赖哈希。

此模型不再与原fire共享同一backbone，不得打包替换旧双头；它是独立烟雾候选。无独立测试/上板验收。请给可执行代码，不要增加大批检查或测试框架；上述均为本次对照必要身份与冻结验证。
