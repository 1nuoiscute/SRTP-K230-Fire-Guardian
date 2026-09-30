# 真实同步数据接口｜待板端接入

电脑端融合回放已有实现。真实数据验收前，每条记录保留 UTC 时间、视频/厨房/机位/事件 ID、原始传感器值、通道有效标志与原采样时间。接收时间不能冒充采样时间；跨核只知道文件读取时间时，将 sensor_sample_time 标为 unknown。

| 输入 | 字段 | 当前缺口 |
|---|---|---|
| 图像 | video_sha256、frame_index、fps、width/height、capture_time、fire boxes/conf | 板端还没有统一事件日志输出 |
| SHT31 | temperature_c、humidity_pct、crc_ok、sample_time | 已有读数；需在源采样处记录时间 |
| BMP280 | temperature_c、pressure_hpa、valid、sample_time | 温度/气压不能冒充烟雾/燃气 |
| 烟雾/燃气 | sensor_model、raw_value、units、valid、sample_time | 尚无真实集成输入；保留缺失，不以 0 替代 |
| 事件真值 | event_id、start/end、normal cooking / abnormal / uncertain、label_basis | 需完整事件过程和人工审核 |

先记录原始值，再在开发集估计基线与阈值。测试按完整事件和厨房隔离，报告漏报/误报、触发时间与缺数行为。现有 tmpfs status v1 序号不是时间戳；本轮未修改已验收板端格式。

下一次板端接入依次验收：源时间戳 → 两核一致记录 → 视频对齐误差 → 通道失联 → 固定规则回放 → 自适应对照。未获得同步烟雾和异常事件真值时，完整融合准确性保持未验收。

电脑端视觉转换现已使用同一个框的置信度和几何，要求预测 CSV 的真实宽高与 ROI 一致，并绑定转换/回放配置。技术复核见 [输入边界修复](../model_work/docs/VISUAL_FUSION_BOUNDARY_FIX_20260930.md)。这不补齐源采样时间、烟雾或真实事件真值。
