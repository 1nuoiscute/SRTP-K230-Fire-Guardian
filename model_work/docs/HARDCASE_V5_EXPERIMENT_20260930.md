# 蓝焰与贴纸难例实验 v5｜2026-09-30

状态：GPU 训练进行中，尚无改进结论；开发板未开机。

## 数据

- 旧 train 全部 2382 张保留，按原文件哈希核对；原 val/test 哈希与新增训练图不相交。
- 队友五段视频中已有人工查看的 10 张帧使用近似火焰框，每张在训练列表加权 12 次。
- 8 张帧下部裁剪及 2 张贴纸局部裁剪作为空标签难负例，每张加权 8 次。全部 44 张派生图已目视核查：正例框位覆盖可见蓝焰，负例不含可见火焰。
- 原四段视频的 24 张已审核帧保留，每张加权 3 次。
- 总计 2654 个训练条目、2426 个不同路径；新增派生图 44 张。重复加权不是独立场景增加。
- 构建清单、审核确认和叠框图在本地 model_work/data/hardcase_v5_20260930。

## 配置和身份

起点：电脑端 v3，SHA-256 48f284f9094fe334e02c66647f95fed8bfb3396b09f4488753508f514ed38980。

YOLO11s，640，batch 4，AdamW，lr0=0.00015，12 轮，seed=20260930。完整参数/环境在运行 preflight 和训练 args.yaml。运行目录：model_work/runs/fire-binary-hardcase-v5-20260930；日志：model_work/out/hardcase_v5_20260930。

```powershell
python -B model_work/src/build_hardcase_v5.py --base D:/SRTP_Datasets/kitchen_fire_binary_codex_20260927 --out model_work/data/hardcase_v5_20260930
# 查看 label_review_sheet.jpg；审核后记录 review_approval.json 和清单哈希，再训练。
python -B model_work/src/train_hardcase_v5.py --data model_work/data/hardcase_v5_20260930/data.yaml --start model_work/runs/fire-binary-user-video-v3/weights/best.pt --expected-sha256 48f284f9094fe334e02c66647f95fed8bfb3396b09f4488753508f514ed38980 --out model_work/runs/fire-binary-hardcase-v5-20260930 --epochs 12
```

## 待完成比较

v3 与 v5 best/last：人工审核帧一对一 IoU≥0.5 定位、贴纸负例误报、旧四段视频、队友五段视频、未训练的五张蓝焰照片、旧 286 图诊断。权重选择依据与退化都记录；这些材料均不作为最终盲测。

## 数据边界

九段视频已进入开发链；新视频脚本现在检查公开 exposure registry，拒绝同哈希素材，即使改名。哈希检查不证明不同厨房/事件，也不能替代初始化链审计。缺少全新事件/关火数据，因此独立验收、真实每小时误报与火情判断仍未完成。
