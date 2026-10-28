# 编码任务：v21 对照结果汇总

请输出一个完整 Python 3 标准库脚本，放在单独 python 代码块内，并附简短说明。脚本只读 experiment_meta.json 和 comparison_complete.json，用 argparse 接受 --training、--comparison、--out，拒绝覆盖已有 out 目录，写 summary.json 和 summary.md。不执行预测、不修改模型、不自动晋升。

comparison 的结构：顶层 models 是模型标签到对象的映射。每个对象有以下准确字段：
- weights_sha256
- blue.primary_localized_iou50 和 blue.images
- legacy.map50
- visible_flame.supplemental['0.5']，值为 {'tp':整数,'fp':整数,'fn':整数,'micro_f1':浮点}
- sources.nofire_real_indoor，值含 images、tp/fp/fn、images_with_fp
- new_training_sources.by_origin.reviewed_commons_queue，值含 images、tp/fp/fn、images_with_fp
- frames.by_origin 是来源到计数的映射，各值含 images、tp/fp/fn

training 的结构有 actual_optimizer_steps、actual_batches、config、checkpoints、serialized_frozen_state_equal、serialized_adaptation、environment。

summary.json 要保留以上所有原始子结构和模型身份（不能仅做数字表后丢掉原始结构），summary.md 只做紧凑对照表：标签、蓝焰定位/图数、紧火焰TP/FP/FN、室内无火FP、14图训练曝光TP/FP/FN、旧mAP50。缺少字段明确报错，绝不默认0。Markdown 标注“全部是已曝光开发反馈，未完成独立测试或上板验证”。training 字段完整保留。给出相对 v16_best 的计数差，不设晋升阈值。

这是具体编码交付；不需要建议新增测试框架或重构训练代码。
