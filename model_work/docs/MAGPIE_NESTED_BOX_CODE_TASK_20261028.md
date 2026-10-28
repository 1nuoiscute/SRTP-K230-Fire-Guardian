# 编码：锅体大框与火焰紧框的嵌套抑制开发对照

请写一个完整标准库Python脚本 eval_nested_fire_boxes.py，用已有 data_integrity.sha256/matches/iou；不导入YOLO、不重新预测或训练、不改原权重/标签。接受 --comparison（v21图片比较目录）、--videos（v21视频目录）、--blue-truth（5图标注JSON）、--out。拒绝已有out。

先固定实验：模型v16_best、v18_last、v21_best、v21_last；原始conf=.25/NMS=.6缓存。四种policy：original、nested_r3（主规则）、nested_r2和nested_r4（预先固定敏感性对照，不能事后挑最好一个作为独立结论）。规则只用预测框：同类，child面积小于outer，child面积至少1e-9，intersection/child_area>=.95，outer_area/child_area>=ratio时抑制outer；同时判定基于原始列表，不在循环中依次修改；不使用GT、文件名、已知目标位置或图像内容。重复相同面积框不互相删除，非有限/无效几何报错。保留原顺序、完整confidence，并记录被删除索引与导致删除的child索引。必须明确几何嵌套不等于外框语义错误，因此规则可能误删合法整体火焰框。

数据读取（示例结构附带）：
1) comparison/legacy_sources/local_inputs.json 是286图源记录list，每条file/image/sha256/label_sha256/gt。各模型 comparison/legacy_sources/{model}_local_predictions.json list，image(文件名)/provenance/predictions[{confidence,xyxy}]，与local_inputs按file匹配。用原gt算每provenance TP/FP/FN/images_with_fp。
2) comparison/visible_flame/{model}_details.json list：index/image/visible_flame_boxes/original_boxes/predictions，已批准近似物理火焰目标。每policy在.5 IoU计分，逐图原始/保留/删除完整保存并汇总TP/FP/FN。不得修改original指标来冒充新的真值。
3) comparison/comparison_complete.json中models[model].blue.per_image list，slug/predictions，GT来自--blue-truth list的slug/primary_flame_xyxy；只报主目标best_iou与命中数，不推断全部FP。
4) comparison/comparison_complete.json中models[model].new_training_sources.frames list，image/predictions。GT来自既有 data_manifest 所在目录不方便则从 comparison/{model}_new_sources/summary.json 同结构也无GT；本脚本无需评该组，不编造GT。
5) videos/per_frame.jsonl逐帧video_index/frame_index/sample_index/.../predictions[model]。对各policy记录原始/保留/删除数量与预测几何，按video汇总被删帧数和被删框数；框数不是识别准确率或误报率。可读videos/plan.json绑定输入身份。

输出 comparison_plan.json（脚本hash、所有读取JSON/JSONL hash、固定rules与参数、角色边界），summary.json，image_details.json与video_details.jsonl，summary.md紧凑比较表。先对original重计分并与旧summary来源分组与tight supplemental['0.5']、blue命中数核对，若不一致直接报错，不能在错位数据上比较。组内sum不能默认为0掩盖字段缺失。写out前完成必要输入读取与校验，输出过程不覆盖原缓存。所有结果是已曝光开发诊断，无新mAP、独立测试或上板结论。

本阶段只要准确可运行的缓存对照，不要写大测试框架、下载或修改默认推理入口。
