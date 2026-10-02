# 无板电脑端模型推进｜2026-10-02

用户授权旅游期间自主推进模型工作，本次没有连接板子。9 月 30 日的暂停决定保留为历史；本次重新开始电脑端实验。不会修改原数据、v3 权重、失败候选或已验收板端版本。

## 项目检查与本轮选择

开始时 Git main 与 origin/main 同步，工作区干净，HEAD e74f646。早期四类板端 MVP、KEY2 永久启动与单类电脑端 v3 是不同版本；模型的跨厨房泛化、真实同步融合、执行反馈和长期运行仍缺验收。电脑端 PyTorch 2.12.0+cu132、Ultralytics 8.4.70、RTX 5060 Laptop 8GB 可用；ONNX 与 ONNX Runtime 已安装。

v5–v11 已多次微调。v9 last 改善蓝焰但旧图 mAP50 和无火误检退化。此次只对 v3 与其后代 v9 last 做固定系数插值，检验是否存在更好的开发折中；不继续添加照片或调训练权重。方法依据 [Model soups 原论文](https://arxiv.org/abs/2203.05482)，该论文不保证本项目 YOLO 灶火检测会改善，实际结果决定结论。

## 运行前冻结的方案

- v3 SHA-256：48f284f9094fe334e02c66647f95fed8bfb3396b09f4488753508f514ed38980。
- v9 last SHA-256：1ead79c990f0121dc779cee97a155f779deeea428a32d0cc85e56b66be5fe867。
- 固定 alpha=0、0.25、0.5、0.75、1；其中 0/1 直接评测原始权重，其余生成独立候选。
- 浮点参数与 BN 浮点缓冲以 FP32 线性插值，整数缓冲沿用 v3；不重新标定 BN、不训练，不做预测集成。必须报告这一范围。
- 所有候选比较同一 v9 的 55 个派生开发图、五张蓝焰近似主要炉位、原 286 图 mAP 和按来源的固定阈值 TP/FP/FN。固定阈值 conf=.25、NMS IoU=.6、匹配 IoU=.5。
- 预先定义进一步检查门槛：旧图 mAP50 损失不超过 .01、蓝焰定位至少多 1 图、原无可见火焰来源 FP 框最多多 1。该门槛只是开发筛选，不是正式验收标准；候选仍需完整画面和视频复核，不能自动替换 v3。
- 每个子任务超时 30 分钟，失败保留目录和日志，不自动重训/重启；最多检查上述三份中间权重，不追着五张照片追加系数。

这些图片/视频均曾曝光。部分蓝焰原作已用于 v9 训练，原图 mAP 也不是独立厨房成绩。本次不新增独立测试声明。插值结果不能解释为训练收敛或事件报警能力。

```powershell
python -B model_work/src/run_checkpoint_blend.py --base model_work/runs/fire-binary-user-video-v3/weights/best.pt --adapted model_work/runs/fire-binary-hardcase-v9-sources-20260930/weights/last.pt --base-sha256 48f284f9094fe334e02c66647f95fed8bfb3396b09f4488753508f514ed38980 --adapted-sha256 1ead79c990f0121dc779cee97a155f779deeea428a32d0cc85e56b66be5fe867 --data model_work/data/hardcase_v9_sources_20260930 --out model_work/out/checkpoint_blend_20261002
```

本地详细结果、权重、原图和私人逐图身份不进入公开 Git。运行身份绑定两端权重、数据 preflight 与脚本哈希。结果和 ONNX 导出验证完成后追加本文件，并保存公开汇总。
