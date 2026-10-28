# 编码交付：烟雾 v5 固定对照评测

请输出一个完整Python脚本（单python代码块）eval_smoke_extension_v5.py，用项目已有data_integrity.sha256/matches与Ultralytics YOLO。接受 --data（烟雾扩充目录）、--models（模型清单JSON）、--cooking-review（旧review_complete.json）、--out。拒绝覆盖out。只预测指定数据，不读预留图。

data/build_manifest.json 的images每条字段为 file(图片绝对路径)、label(标签绝对路径)、split(train/val)、origin(v4_parent/extension)、image_sha256、label_sha256、group。每张标签YOLO单类0=smoke，空标签代表经过人工审核无可见烟雾。只评两组：split=val的原9张开发图；origin=extension的新增16训练曝光图。旧train不用预测。每个模型都在同一图同参数640/conf.25/NMS.6/FP32/device0/rectTrue预测，用matches(GT像素xyxy,预测xyxy)计TP/FP/FN，按组报告images/images_with_fp/TP/FP/FN，逐图结果完整保留。模型清单是 [{label,weights,expected_sha256}]，要求names={0:'smoke'}，保存权重sha。

cooking-review的reviews每条含image(绝对路径)、image_sha256字段名请根据附带示例确认；若是sha256就用sha256。这19图语义为烹饪羽流/蒸汽/油烟未知，只有预测统计images_with_predictions与box_count，不得打TP/FP/FN或F1，也不当负例。独立报告unknown_cooking_plume。给每个模型的9个val+16个新增+19个未知诊断图保存叠加预测图，每页4图的联系表（原框green/预测blue，未知图不画真值）。写summary.json、局部逐图结果与联系表文件名列表。

只用标准库、cv2、numpy、torch和Ultralytics以及现有data_integrity。torch.set_num_threads(2)。不需要复杂框架、不下载、不训练、不自动晋升。对照限于已曝光开发；新增数据增加训练更新量，不能声称因果或泛化。
