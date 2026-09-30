# SRTP K230 Fire Guardian｜边缘视觉火情检测与多传感器融合

面向厨房场景的大学生 SRTP 研究原型。基于 **K230 + RT-Thread Smart / Linux**，开展火焰目标检测、环境采集、事件判断与模拟联动研究。

**当前阶段：板端视觉与环境显示 MVP，电脑端模型改进中。** 正常灶火与异常火情需要分开判断；当前模型尚未通过跨厨房可靠性验收。

## 已有成果

- 早期四类视觉模型已完成 KModel 转换与板端摄像头画框演示。
- SHT31 温湿度、BMP280 温度/气压、SSD1306 OLED 有实机验证。
- 副板 K1 控制视觉与实时面板，K2 控制传感器；底板矩阵 KEY1 控制面板显隐。
- 小核通过 tmpfs 状态文件向大核传递传感器读数；已有 20 分钟内存历史缓存。
- 电脑端具备数据来源审核、重复检查、训练、逐图与视频诊断脚本，以及可解释的融合状态机。

## 当前限制

- 2026-09-28 新厨房视频暴露锅下蓝焰漏检、蓝色贴纸误检；电脑端 v3 是开发基线，新权重尚未上板。
- 全屏数据页切换曾出现乱码，候选未作为正式启动版本。
- 烟雾/燃气输入、真实事件融合、自适应标定、执行反馈和 MQTT 闭环尚未验收。
- 旧图片集和已反复分析的视频均属于开发诊断；其帧数、mAP 或有框比例不能替代独立事件验收。

## 最新实验（2026-09-30）

v5 已完成训练与对照：训练曝光蓝焰帧定位改善、贴纸误报消除，但旧图明显退步，外部蓝焰未改善，拒绝替换 v3。v6 在同数据上测试较小学习率日程，仍在运行。详见 [v5 完整记录](model_work/docs/HARDCASE_V5_EXPERIMENT_20260930.md) 和 [v6 实验](model_work/docs/HARDCASE_V6_EXPERIMENT_20260930.md)。

## 从这里开始

| 内容 | 入口 |
|---|---|
| 最新状态及电脑端推进 | [项目状态](docs/STATUS.md)、[路线与实验门槛](docs/ROADMAP.md) |
| 板端实时面板 | [实机记录](model_work/docs/BOARD_UI_LIVE_INTEGRATION_20260928.md) |
| 数据页失败及恢复 | [数据页记录](model_work/docs/BOARD_DATA_PAGE_KEY2_20260928.md) |
| 新厨房视频失败 | [冻结模型测试](model_work/docs/TEAMMATE_NEW_VIDEO_TEST_20260928.md) |
| 训练与数据沿革 | [模型总报告](model_work/docs/MODEL_TEST_AND_TRAINING_REPORT_20260927.md)、[视频适配](model_work/docs/USER_VIDEO_ADAPTATION_RUN_20260927.md) |
| 电脑端运行方法 | [复现指南](docs/REPRODUCE_PC.md) |
| 有用参考资料 | [参考入口](references/README.md)、[本地资料清单](references/LOCAL_MATERIALS.md) |
| 源码与版本边界 | [板端版本身份](docs/BOARD_BASELINE.md)、[第三方说明](THIRD_PARTY_NOTICES.md) |

## 目录

`src/` 保存板端代码与实验候选；`patches/` 保存历史 SDK 适配补丁；`model_work/src/` 保存电脑端数据、训练、诊断与回放脚本；`model_work/docs/` 保存实验记录；`docs/` 是公开导航与复现说明；`references/` 是参考资料索引与使用笔记。

本地大文件、原视频、私有数据、权重、固件备份和个人申报材料保留在原路径，不加入普通 Git。数据与固件的身份由各实验清单和 SHA-256 记录。部分旧脚本含历史机器路径，请按复现指南配置；新实验脚本提供路径参数。

## 验证

```sh
python -B -m unittest discover -s model_work/src -p 'test_*.py' -v
python -B tools/check_public_repo.py
```

该仓库公开研究过程与失败记录。第三方代码、数据及资料遵守各自许可，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
