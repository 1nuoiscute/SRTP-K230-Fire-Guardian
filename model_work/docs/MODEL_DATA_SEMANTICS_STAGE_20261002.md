# 模型数据与标注口径推进｜2026-10-02

v12 已完成且拒绝替换 v3。本阶段停止追加来源采样上限，转向可见火焰框和烟雾标签语义。用户持续目标仍有效，当前无开发板连接。

## 运行前确定的范围

1. 保留 v3、v5–v12 与全部原数据；从已有 train 抽查中，复核六张标注含大块锅体的灶火图，依据实际可见火焰给出明确的新框口径，不能依模型预测生成标签。新修订另建目录和身份清单，不触碰旧 val/test 或把新标签得分混成旧 mAP。
2. 下载 [D-Fire 官方 README](https://github.com/gaia-solutions-on-demand/DFireDataset) 链接的 [Kaggle 镜像版本1](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo/versions/1)，先实际核验与抽查，不自动加入训练、不使用其 split 冒充独立厨房测试。原作者仓库 commit 固定 `4bf9c31b18fadcd44d5f0b6d66f82bc56fa5e328`；其 LICENSE 描述集合 CC0且不持有所收集图片版权，许可边界保留。

公开元数据实际返回：id=6556263、ref=sayedgamal99/smoke-fire-detection-yolo、version=1、lastUpdated=2025-01-27、totalBytes=3118334483、licenseName=CC0: Public Domain。镜像自己说明增加 validation split，因此不是未经变化的官方划分。

下载 GET 响应：application/zip、3049605157 字节、ETag=`9745236834d65441af996cb6da6a4fde`，重定向到公开 storage.googleapis.com 对象。HEAD 返回404，GET实际200，不把HEAD失败当下载不可用。初次元数据控制台输出因GBK无法编码表情失败，ASCII JSON重查成功，未进行重训/重下载。

目标目录 `D:/SRTP_Datasets/dfire_kaggle_v1_20261002` 必须新建，预计下载+解包约6.2GB，最低剩余空间12GB；不清理原文件。单次下载最长1800秒、HTTP读超时45秒，收到字节/ETag-MD5/本地SHA/ZIP CRC逐级核查；异常保留partial与身份记录，不自动重试。解包总大小上限8GB、成员上限50000，拒绝目录穿越、Windows特殊路径、大小写碰撞、符号链接和覆盖。

```powershell
python -B model_work/src/acquire_dfire_kaggle.py --out D:/SRTP_Datasets/dfire_kaggle_v1_20261002 --max-download-seconds 1800
```

工具使用匿名公开HTTP，不登录或读取账户令牌；只采集，绝不自动启动模型训练。原图、归档、元数据全文和逐图审核留本地；公开代码、脱敏汇总、失败原因与模型实验记录。

当前状态：方案在下载前固定；待路径/归档安全测试通过后启动。后续追加实际下载、标注复核和训练决定。

下载前核对 D 盘可用 156006236160 字节，requests=2.33.1。首次路径测试在 Windows 创建 ZIP fixture 时反斜杠被 ZipInfo 自动规范成正斜杠，导致非法路径用例没有实际进入验证器；修正 fixture 恢复原输入，再检查拒绝行为。这是测试构造问题，未解包外部数据或改变原文件。
