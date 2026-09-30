# 视觉结果转融合输入的修复｜2026-09-30

此前转换器分别取所有框中的最大置信度、最大灶区外比例和最大框面积，可能拼接出不存在的单框证据。例如使用 0.55 置信度门槛时，灶内 0.90 框和灶外 0.26 框会合成“高置信度灶外火焰”。原函数还把每段画面写死为 720×1280，无法正确处理其他尺寸。

## 当前规则

转换时读取回放配置。先筛选达到 fire_conf_min 的框，在合格框中选择灶区外比例最大的一个，置信度和面积也取自这个框；因此合格的灶外框不会被更高置信度的灶内框遮蔽。没有合格框时保留同一个最高置信度低阈值框，回放会按相同门槛忽略它；空检测列表输出零值。

画面宽高必须由预测 CSV 提供，与每段 ROI 的宽高一致；框和 ROI 需在实际画面内。框面积/灶外比例是包围框几何代理，不是分割后的真实火焰面积或严重程度。固定 ROI 仍要求固定机位或另行处理镜头移动。

示例 ROI JSON（尺寸和区域只是说明格式）：

```json
{"example.mp4": {"width": 1920, "height": 1080, "roi_xyxy": [500, 600, 1100, 950]}}
```

输入采用 eval_development_videos.py 的 video、time_seconds、width、height、predictions 字段。旧 predictions_025 表和无尺寸的 ROI 列表需明确转换为新结构，程序不猜测尺寸。

```powershell
python -B model_work/src/fusion_replay.py visual-to-fusion --input your_per_frame.csv --roi-json your_reviewed_rois.json --config model_work/data/fusion_demo_config_20260927.json --out model_work/out/your_new_fusion.csv
python -B model_work/src/fusion_replay.py replay --input model_work/out/your_new_fusion.csv --config model_work/data/fusion_demo_config_20260927.json --out model_work/out/your_new_replay
```

转换生成 CSV 和 .csv.meta.json，记录输入/ROI/配置/输出及程序哈希。回放要求使用相同配置并校验转换文件身份，避免转换和回放门槛不同。文件哈希只绑定这些离线文件，不证明摄像头源时间或完整视频/模型来源。

## 验证与限制

新增 7 项转换测试覆盖弱灶外框、高置信度灶内与合格灶外并存、横屏面积、低阈值框、空检测、非法坐标、转换到回放及配置/输出修改拒绝。加上 7 项数据输入校验与原 17 项，共 31 项通过。

初次集成测试失败的原因是测试借用了演示配置的 0.25 门槛，却按 0.55 预期判断；已改为测试自己的 0.55 配置，演示配置未改。

实际还用 v6 last 对原 3.mp4 的 34 个已曝光预测采样完成转换→回放。ROI 沿用既有人工火焰包络，属于固定开发代理区域；用未标定演示阈值，34 个样本全部 UNKNOWN，烟雾/温度值保持空、有效标志保持 0。缺少同步传感器时不会编造正常或报警结论。原 CSV、视频及模型未改，独立输出位于 model_work/out/visual_fusion_boundary_20260930；公开哈希和结果见 docs/visual_fusion_boundary_results_20260930.json。

没有真实异常事件、同步烟雾、标定 ROI 或板端运行。本次仅修复证据拼接与尺寸错误，不能作为融合可靠性或报警验证。
