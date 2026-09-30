> 历史记录公开副本；机器路径已简化。状态按文内日期理解，最新入口为 docs/STATUS.md。

# RT-Thread Smart AI 套件实机验证记录

日期：2026 年 9 月 23 日  
项目：边缘 AI 火情检测与自适应联动装置  
硬件：RT-Thread Smart AI 套件，K230 双核平台

## 1. 今日结论

今天完成了开发板首次系统化实机检查，并从零打通了完整开发链路。开发板、显示屏、摄像头、Linux 小核、RT-Smart 大核、双串口以及 ADB 文件传输均可正常工作，大小核也能够共同访问 `/sharefs`。此外，已在外置 SSD 上建立独立 WSL2 环境，取得 K230 SDK v1.6，完成工具链准备，并成功自行编写、交叉编译、传输和运行新的 RT-Smart Hello World 程序。

目前只能确认硬件和基础系统链路正常，不能确认火灾识别功能已经完成。板内原有的 `kitchen_fire.elf` 和 `best.kmodel` 是此前赶工留下的实验文件，来源、版本、模型类别、准确率和代码完整性均未验证。实测表明 `kitchen_fire.elf` 的摄像头模式尚未实现，因此后续应建立新的干净工程，不能直接把这些旧文件当作项目基础或已有成果。

本次新程序在 RT-Smart 实机输出 `SRTP RT-Smart hello`，证明 Windows、WSL2、Docker、K230 SDK、RISC-V 交叉工具链、ADB、共享文件系统和 RT-Smart 执行环境已经贯通。该结果来自新的干净源码，与此前失败的 `kitchen_fire.elf` 无关。

## 2. 硬件状态

从实物检查和上电现象确认：

- 开发板能够正常上电和启动。
- MIPI 显示屏能够显示摄像头实时画面。
- CSI2 摄像头工作正常。
- 2.4 GHz 天线已经连接。
- SensorFusion Shield 已安装在主板扩展接口上。
- 启动拨码保持原有正确状态，使用 eMMC 启动。
- `DEBUG&5V` 接口用于供电和双串口调试。
- `OTG` 接口连接电脑后能够提供 ADB 通道。

安全边界：今天没有连接继电器、燃气阀或其他执行机构，也没有使用真实火焰进行测试。

## 3. Windows 串口识别与调试工具

Windows 设备管理器识别出以下串口：

| 端口 | 设备 | 对应系统 |
|---|---|---|
| COM5 | USB-Enhanced-SERIAL-A CH342 | Linux 小核 |
| COM6 | USB-Enhanced-SERIAL-B CH342 | RT-Smart 大核 |
| COM3、COM4 | 蓝牙标准串口 | 与本次开发板调试无关 |

电脑未安装 MobaXterm，因此使用已有的 VOFA+ 1.3.10 作为串口终端。可用配置如下：

- 数据引擎：`RawData`
- 数据接口：串口
- 波特率：`115200`
- 数据位：`8`
- 停止位：`1`
- 校验位：`None`
- 流控：`None`
- COM5 的 Linux 终端可接受 `\n`。
- COM6 的 RT-Smart 终端使用 `\r\n` 更可靠。
- 关闭 VOFA+ 的 `Hex` 显示后，可以直接阅读终端文本。

## 4. Linux 小核实测结果

通过 COM5 成功以 `root` 身份进入 Linux 小核终端，提示符为：

```text
[root@canaan ~ ]#
```

### 4.1 内核和系统

执行：

```sh
uname -a
cat /etc/os-release
```

得到：

```text
Linux canaan 5.10.4 #1 SMP Wed Apr 23 04:50:51 UTC 2025 riscv64 GNU/Linux
NAME=Buildroot
VERSION=-g87b812017-dirty
ID=buildroot
VERSION_ID=2021.02-git
PRETTY_NAME="Buildroot 2021.02-git"
```

已确认 Linux 小核为 RISC-V 64 位 Buildroot 系统。`free -h` 在当前 BusyBox 1.33.0 中不受支持，需要改用 `free -m` 或 `free -k`。

### 4.2 文件系统

执行 `df -h` 后确认：

```text
/dev/root       150.9M   73.0M   73.2M  50%  /
/dev/mmcblk0p4  252.0M  102.5M  149.6M  41%  /sharefs
```

`/sharefs` 是大小核共享目录。初次检查时包含：

```text
app/
best.kmodel        10819968 bytes
kitchen_fire.elf    9811640 bytes
test.jpg                  0 bytes
```

其中 `test.jpg` 是空文件，不能作为有效测试图片。

## 5. RT-Smart 大核实测结果

通过 COM6 成功连接 RT-Smart。开机摄像头程序运行时，发送小写 `q` 并使用 `\r\n` 换行后，程序正常停止，日志包括：

```text
kd_mpi_isp_stop_stream chn enable is 1
kd_mpi_isp_stop_stream chn enable is 0
release reserved vb 269402112
release reserved vb 0
```

这些信息表明摄像头 ISP 视频流和视频缓冲区被正常释放。退出后可以进入 `msh`：

```text
msh />
```

RT-Smart 能够看到与 Linux 小核相同的 `/sharefs` 内容，证明共享目录链路正常。

`/sharefs/app` 中存在多种官方或随镜像提供的示例程序，包括：

- `sample_vicap.elf`
- `sample_vicap_dump.elf`
- `sample_face_detect.elf`
- `sample_triple_camera_facedetect.elf`
- `sample_vo.elf`
- `sample_venc.elf`
- `sample_gpio.elf`
- `sample_pwm.elf`

这些文件证明当前镜像具备摄像头、显示、编码和常见外设示例，但只有 ELF 二进制文件不能直接作为新项目源码。后续需要取得与当前开发板匹配的 K230 SDK 和示例源代码。

## 6. 旧火灾识别文件的实际状态

在 RT-Smart 中执行旧程序但不提供参数时，得到：

```text
Usage: kitchen_fire.elf <kmodel> <score_thres> <nms_thres> <input_mode> <debug_mode>
  kmodel       kitchen_fire.kmodel path
  score_thres  detection score threshold
  nms_thres    NMS threshold
  input_mode   image path / None (for camera)
  debug_mode   0/1/2
```

使用以下命令尝试摄像头模式：

```sh
kitchen_fire.elf best.kmodel 0.5 0.45 None 0
```

程序返回：

```text
Camera mode not yet implemented
```

由此确认：

1. 旧程序具有参数解析和单图模式接口。
2. 帮助文本虽然写有 `None (for camera)`，但摄像头模式实际没有实现。
3. 开机时出现的摄像头画面来自另一个自启程序，不能据此证明旧火灾识别程序支持实时摄像头。
4. `best.kmodel` 尚未验证模型类型、输入尺寸、类别表、量化配置、精度或板端兼容性。
5. 旧 ELF 和 KModel 只能作为失败实验材料留存，不能写入项目材料作为已完成功能。

## 7. ADB 文件传输

Windows 最初没有安装 `adb`。随后下载并解压官方 Android SDK Platform-Tools，当前路径为：

```text
D:\platform-tools-latest-windows\platform-tools
```

开发板通过 OTG 连接后，执行：

```bat
adb.exe devices
```

成功识别：

```text
RTT_K230_ADB    device
```

为避免覆盖旧实验文件，在板端创建了新的干净目录：

```text
/sharefs/srtp_clean
```

通过 ADB 将一张开发板照片传入：

```text
/sharefs/srtp_clean/board_test.jpg
```

验证结果：

```text
-rwxr-xr-x  1 root root 218209 Sep 23 2026 board_test.jpg
```

这证明 PC 到 Linux 小核的 ADB 通道、`/sharefs` 写入以及后续大小核共享部署路径可用。旧文件没有被覆盖。

## 8. 外置 SSD 与独立 WSL 开发环境

由于电脑内置分区空间不足，接入了 F 盘外置 SSD。Windows 将其识别为通过 `ASMT 2115` USB 桥接的约 119.2 GB NTFS 磁盘，初始可用空间约 83.3 GB。为避免占用系统盘和干扰原有 WSL，没有迁移旧发行版，而是在 F 盘新建了项目专用发行版：

```powershell
mkdir F:\WSL\K230-Ubuntu
wsl --install -d Ubuntu-22.04 --name K230-Ubuntu --location F:\WSL\K230-Ubuntu --vhd-size 70GB --web-download
```

安装结果：

- WSL 发行版名称：`K230-Ubuntu`
- 默认用户：`k230`
- 系统：Ubuntu 22.04.5 LTS
- 架构：`x86_64`
- WSL 根文件系统容量约 69 GB
- 环境初始化后约有 64 GB 可用

以后从 Windows 启动开发环境：

```powershell
wsl -d K230-Ubuntu --cd ~
```

工程应放在 `/home/k230` 内，不要放在 `/mnt/c`、`/mnt/d` 或 `/mnt/f` 下直接编译。外置 SSD 必须保持连接；准备拔出 SSD 前应先在 Windows 执行：

```powershell
wsl --shutdown
```

## 9. 网络、Docker 与 K230 SDK v1.6

### 9.1 网络经验

WSL 启动时会提示“localhost 代理配置未镜像到 WSL，NAT 模式不支持 localhost 代理”。本次测试中：

- GitHub 的 Git/TLS 连接偶尔出现 `GnuTLS recv error (-110)`，不适合依赖其完成长时间源码下载。
- Docker Hub 拉取 `hello-world` 时发生连接重置，但这不代表 Docker 安装失败。
- `ghcr.io` 可正常访问，`curl -I https://ghcr.io/v2/` 返回 `405 Method Not Allowed` 也能证明服务器可达，因为这里只是不接受 HEAD 方法。
- K230 SDK 源码改用官方 Gitee 镜像获取，容器镜像从 GHCR 获取。

### 9.2 Docker 与官方构建镜像

在 `K230-Ubuntu` 中安装并启动 Docker 后，`sudo docker version` 能同时显示 Client 和 Server。成功拉取：

```sh
sudo docker pull ghcr.io/kendryte/k230_sdk:v1.6
```

镜像摘要：

```text
sha256:49be8e30c52407d657a41e2dddd79d4a4aa973c97ff55a35ed709ec13b898e95
```

### 9.3 SDK 源码与工具链

SDK 位于：

```text
/home/k230/k230_sdk_v1.6
```

当前源码提交：

```text
8074e1e1adde7587d36eca6745342c14f1dcd8fb
```

已确认本套件相关配置存在：

```text
configs/k230_canmv_dongshanpi_defconfig
```

执行以下命令准备源码和工具链：

```sh
cd ~/k230_sdk_v1.6
source tools/get_download_url.sh
make prepare_sourcecode
```

脚本提示 `URL is not accessible` 时，指的是首选站点 `https://ai.b-bug.org/k230/` 不可用；脚本随后会设置官方备用下载地址 `https://kendryte-download.canaan-creative.com/k230`。该地址跳转到 `download.kendryte.com` 后可正常下载，因此这条提示本身不是失败。最终 `make prepare_sourcecode` 返回码为 0。

准备完成后的实测占用：

- SDK 目录约 6.6 GB
- `toolchain` 约 3.1 GB
- WSL 根文件系统已用约 10 GB，剩余约 55 GB

### 9.4 进入构建容器

在 SDK 根目录执行：

```sh
sudo docker run --rm -u root -it \
  -v "$(pwd):$(pwd)" \
  -v "$(pwd)/toolchain:/opt/toolchain" \
  -w "$(pwd)" \
  ghcr.io/kendryte/k230_sdk:v1.6 \
  /bin/bash
```

容器内验证结果：Ubuntu 20.04.6、Python 3.8.10、SCons 3.1.2，且以下两个 RISC-V 工具链可见：

```text
/opt/toolchain/Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.0/bin/riscv64-unknown-linux-gnu-gcc
/opt/toolchain/riscv64-linux-musleabi_for_x86_64-pc-linux-gnu/bin/riscv64-unknown-linux-musl-gcc
```

## 10. 从零编译并运行 RT-Smart Hello World

这是本次最重要的可复现验证。所有新文件位于 `srtp_sandbox`，不会覆盖 SDK 示例或板内旧实验文件。

### 10.1 新建源码

在上述 Docker 容器内执行：

```sh
mkdir -p srtp_sandbox/hello
cd srtp_sandbox/hello
vi hello.c
```

`hello.c` 内容：

```c
#include <stdio.h>

int main(void)
{
    printf("SRTP RT-Smart hello\n");
    return 0;
}
```

Vim 保存退出方法：先按 `Esc` 确保离开插入模式，再输入 `:wq` 并回车。若把 `:wq` 错写进正文，可先按 `Esc`，将光标移到该行并按 `dd` 删除，再重新输入 `:wq`。

### 10.2 交叉编译

编译目标文件：

```sh
/opt/toolchain/riscv64-linux-musleabi_for_x86_64-pc-linux-gnu/bin/riscv64-unknown-linux-musl-gcc -o hello.o -c -mcmodel=medany -march=rv64imafdcv -mabi=lp64d hello.c
```

链接 RT-Smart ELF：

```sh
/opt/toolchain/riscv64-linux-musleabi_for_x86_64-pc-linux-gnu/bin/riscv64-unknown-linux-musl-gcc -o srtp_hello.elf -mcmodel=medany -march=rv64imafdcv -mabi=lp64d -T ../../src/big/mpp/userapps/sample/linker_scripts/riscv64/link.lds -L../../src/big/rt-smart/userapps/sdk/rt-thread/lib -Wl,--whole-archive -lrtthread -Wl,--no-whole-archive -n --static hello.o -L../../src/big/rt-smart/userapps/sdk/lib/risc-v/rv64 -L../../src/big/rt-smart/userapps/sdk/rt-thread/lib/risc-v/rv64 -Wl,--start-group -lrtthread -Wl,--end-group
```

检查产物：

```sh
file srtp_hello.elf
ls -lh hello.c hello.o srtp_hello.elf
```

### 10.3 从容器导出到 Windows

输入 `exit` 离开 Docker 容器，回到 `k230@LAPTOP...` 的 WSL 提示符，然后执行：

```sh
cp ~/k230_sdk_v1.6/srtp_sandbox/hello/srtp_hello.elf /mnt/d/platform-tools-latest-windows/platform-tools/
```

### 10.4 通过 ADB 传到板端

在 Windows PowerShell 执行：

```powershell
cd D:\platform-tools-latest-windows\platform-tools
.\adb.exe push .\srtp_hello.elf /sharefs/srtp_clean/srtp_hello.elf
.\adb.exe shell "ls -l /sharefs/srtp_clean/srtp_hello.elf"
```

本次首次传输时出现 `device offline`。该问题属于 ADB 会话掉线，不是编译失败。可依次尝试：

```powershell
.\adb.exe reconnect offline
.\adb.exe devices
.\adb.exe kill-server
.\adb.exe start-server
.\adb.exe devices
```

若仍为 `offline`，保持开发板供电不动，仅重新插拔 OTG 数据线，然后再次执行 `adb devices`。恢复后的正确状态为：

```text
RTT_K230_ADB    device
```

### 10.5 在 RT-Smart 真机运行

在 VOFA+ 的 COM6 终端进入 RT-Smart `msh`。如果开机摄像头程序仍在运行，先发送小写 `q` 和 `\r\n` 停止它。随后执行：

```text
cd /sharefs/srtp_clean
srtp_hello.elf
```

实机成功输出：

```text
SRTP RT-Smart hello
```

这证明以下链路已全部真实打通：

```text
自行编写 C 源码
→ K230 RISC-V 工具链交叉编译
→ 生成 RT-Smart ELF
→ 从 WSL 导出到 Windows
→ ADB 传入 /sharefs
→ RT-Smart 大核执行
```

## 11. 当前项目状态边界

### 已确认

- 板卡能够正常启动。
- 摄像头和显示屏能够实时工作。
- Linux 小核和 RT-Smart 大核均可使用。
- COM5、COM6 双串口可用。
- `/sharefs` 大小核共享正常。
- ADB 连接和文件传输正常。
- F 盘项目专用 WSL2、Docker、K230 SDK v1.6 和交叉工具链可用。
- 已从干净源码编译并在 RT-Smart 真机运行新的 Hello World。

### 未确认或尚未完成

- 没有从源码编译摄像头示例。
- `best.kmodel` 的来源和有效性未知。
- 旧 `kitchen_fire.elf` 的摄像头模式未实现。
- 尚未完成单图火灾识别的可信验证。
- 尚未完成实时摄像头火灾识别。
- 尚未开展误报率、漏报率、延迟、帧率和稳定运行测试。
- 尚未连接传感器融合、STM32 安全协处理器或执行机构。

## 12. 后续建议顺序

1. 保留当前 `srtp_sandbox/hello` 和 `/sharefs/srtp_clean/srtp_hello.elf`，作为工具链回归测试基线。
2. 对旧 `best.kmodel`、`kitchen_fire.elf` 和相关材料做只读归档，不覆盖、不当作完成成果。
3. 从当前 SDK v1.6 源码编译与本板匹配的官方摄像头采集或显示示例。
4. 在不刷写固件的前提下，通过 ADB 部署并验证摄像头取帧。
5. 使用来源明确、类别和输入参数明确的 KModel 完成单图推理。
6. 将摄像头采集、预处理、KPU 推理、后处理和屏幕叠加整合为新的实时程序。
7. 在普通图片、屏幕播放素材或安全模拟条件下测试，暂不使用真实火焰和高风险执行机构。
8. 取得可靠视觉基线后，再进入多传感器融合、状态机、告警和安全联动。

## 13. 本次经验总结

1. **先验证链路，再做复杂功能。** Hello World 虽然简单，但它把源码、编译器、链接脚本、RT-Smart 库、文件传输和真机运行一次性验证清楚，是后续排错的基线。
2. **区分 Linux 小核与 RT-Smart 大核。** COM5 对应 Linux，COM6 对应 RT-Smart；ADB 主要用于文件传输，程序最终在 COM6 的 `msh` 中运行。
3. **不要把开机摄像头画面误认为自研程序已经支持摄像头。** 旧 `kitchen_fire.elf` 已明确返回 `Camera mode not yet implemented`。
4. **旧二进制只能作为线索，不能代替源码和验证记录。** 来源、参数和构建环境不明的 ELF/KModel 不应被当作完成成果。
5. **命令必须标明执行环境。** PowerShell、WSL、Docker 容器、Linux 小核和 RT-Smart `msh` 的命令不可混用。
6. **网络错误需要区分服务问题与工具问题。** Docker Hub 或 GitHub TLS 失败不等于 Docker/WSL 本身损坏；可使用官方 Gitee 镜像、GHCR 和 Kendryte 下载站分别完成任务。
7. **ADB `offline` 不会破坏编译产物。** 优先重连 ADB 服务或重新插拔 OTG 数据线，无需重新编译，也不必立刻重启或刷写开发板。
8. **外置 SSD 上的 WSL 适合承载大型 SDK。** 工程仍应放在 WSL 自身的 `/home/k230`，拔盘前先执行 `wsl --shutdown`。

## 14. 本地参考材料

- `RT-Thread Smart AI 套件开发手册.docx`
- `k230-副板.pdf`
- `项目计划书_边缘AI火情检测与自适应联动装置_新版_20260913.docx`

其中 `k230-副板.pdf` 主要对应当前安装的 RW007/SensorFusion 类扩展板电路，不应当作 K230 主板完整引脚图使用。涉及引脚、电压和外设连接时，仍需以对应主板原理图、扩展板原理图和实物丝印交叉确认。

## 15. 2026-09-23 至 2026-09-24 摄像头/显示适配追加记录

> 本节为后续实机排错的最新记录；与前文较早的“尚未编译摄像头示例”等状态冲突时，以本节为准。

### 15.1 板内原厂基准已确认

- 板内开机脚本 `/bin/init.sh` 实际启动：

  ```text
  /bin/fastboot_app.elf /bin/test.kmodel
  ```

- 原厂程序可稳定显示 GC2093 摄像头画面，证明摄像头、排线、NT35516 屏幕和板内固件本身正常。
- 原厂 ELF 已只读备份到：
  - `artifacts/board_reference/stock_fastboot_app.elf`
  - `artifacts/board_reference/stock_fastboot_test.kmodel`
- 原厂 ELF 大小为 `9,707,664` 字节，SHA-256：

  ```text
  025e795058fe7678dfc18cdf6bfb0780d403060d42065d9dbe4ab2525e15deec
  ```

- 对原厂 ELF 的反汇编证明 `sample_connector_init()` 使用连接器枚举值 `5`；对应 NT35516。
- 原厂连接器信息结构体复制长度为 `112` 字节。
- 原厂运行日志确认摄像头为 `gc2093_csi2`，运行时传感器类型编号为 `52`。

### 15.2 手册关于显示适配的真实含义

手册的 AI 部署章节要求：

1. 使用 `k230_canmv_defconfig` 准备环境；
2. 修改 CANMV 分支的显示参数；
3. 使用 NT35516 屏；
4. 执行 `make mpp` 后再编译应用；
5. 横屏显示时旋转画面和叠加层。

手册文字写作 `NT35516_MIPI_2LAN_540X960_30FPS` 和约 `540×960`；板内原厂 ELF及公开 `rtos-v0.6` 接口实际使用 `NT35516_MIPI_2LAN_536X960_30FPS`。因此 `540` 不应被直接当作所有缓冲区的有效像素宽度。

公开嘉楠问答还明确说明：通用 SDK 并不原生支持 RT-Thread Smart AI 套件的 NT35516，需向开发板来源方取得适配。由此不能假定任意公开 K230 SDK/MPP 与板内定制固件二进制兼容。

### 15.3 已完成的 AI 与模型验证

- 新的 YOLOv8 火焰检测应用源码位于 `src/kitchen_fire_yolov8/`。
- `best.kmodel` 可被 KPU 正常加载并完成单图推理。
- 已观察到火焰测试图被识别为 `fire`，但也出现高亮/雾状区域误报，说明旧模型质量不足，不能据此宣称火灾识别已经可靠完成。
- 曾有版本完成连续摄像头推理，单帧约 `174–177 ms`，但显示链路不可靠，因此不能作为可交付版本。

### 15.4 明确失败且不得重复的路线

以下实验文件均只作故障证据，不得替换板内 `/bin/fastboot_app.elf`：

- `artifacts/kitchen_fire_demo/kitchen_fire_ili9806.elf`
- `artifacts/kitchen_fire_demo/kitchen_fire_nt35516.elf`
- `artifacts/kitchen_fire_demo/kitchen_fire_mpp_v06.elf`
- `artifacts/board_reference/rebuilt_stock_fastboot_v06.elf`

失败路线及证据：

1. **旧东山 MPP + 新 NT35516 connector 源码混用**
   - 现象：整屏乱码、连续 `dump_frame failed`、VB 地址校验错误。
   - 原因：只替换连接器接口，`libvo/libvicap/libvb/libsys` 仍属另一版本，ABI 不一致。

2. **公开 `rtos-v0.6` 完整用户态 MPP + 旧 AI 示例 `vi_vo.h`**
   - 现象：整屏乱码；`sensor type not supported`；约每秒一次 `kd_mpi_vicap_dump_frame failed`。
   - 原因：摄像头枚举按启用宏重新编号，同时旧 AI 示例的 VO/VB 初始化不等同于板内 fastboot。

3. **公开 `rtos-v0.6 fastboot_app` 源码原样重建**
   - 现象：`can't probe sensor on 2, output 1920x1080@30`，随后仍出现乱码。
   - 结论：公开 `rtos-v0.6` 源码/库仍不是板内原厂定制构建的精确版本，不能称为本板“原版”。

4. **失败程序退出后立刻重启原厂 ELF**
   - 可能出现：`VB is initialized`、`vb_set_config failed`。
   - 原因：失败程序未完整释放 VB/VO/ISP 状态。
   - 恢复方式：不要继续反复启动 ELF；按硬件 RESET 或执行整板重启，由 `/bin/init.sh` 自动恢复原厂程序。

### 15.5 暂停时的板卡状态

- 用户已按下硬件 RESET。
- 暂停后不再发送串口命令、不再运行实验 ELF。
- 正常预期是重启后由 `/bin/init.sh` 自动运行板内原厂 `fastboot_app.elf` 并恢复摄像头画面。
- 未刷写固件、未覆盖 `/bin/init.sh`、未覆盖板内原厂 ELF/KModel。

### 15.6 下次恢复工作的唯一安全起点

1. 先确认 RESET 后原厂摄像头画面正常。
2. 将板内原厂 `/bin/fastboot_app.elf` 视为唯一已验证显示基准。
3. 首选从 RT-Thread/开发板资料方取得与该固件精确对应的定制 SDK 源码或编译包；在取得前，不再把公开通用 MPP 当作精确替代品。
4. 获得同版源码后，完整照搬原厂 fastboot 的摄像头、VB、VO、NT35516 和退出清理流程，只替换 AI 预处理、KPU 推理、YOLOv8 后处理与画框逻辑。
5. 每次新 ELF 测试前记录哈希；先短测初始化与稳定取帧，再观察屏幕；异常立即停止，若资源释放不完整则 RESET。
6. 视觉链路稳定后再评估/重训火焰模型；环境传感器 I²C（温湿度、烟雾/气体等）另开独立验证，不与显示故障同时调试。

## 16. 2026-09-24 厂商目标检测程序与火焰模型适配全过程

> 本节记录第 15 节之后的新增工作和实测结果。前文第 11、12、15.5、15.6 节是当时的状态快照；涉及当前进度、板端文件与下一步判断时，以本节为准。**本节结束时，尚无可用的板端火焰实时识别 demo，也尚未把板端实时视频传到电脑。**

### 16.1 为什么改用厂商可执行程序

此前尝试自行编译 MPP 相关程序，出现显示乱码、`dump_frame failed`、传感器枚举与板内固件不匹配等问题。后续实验进一步确认：

- 用较新 MPP 编译的火焰程序，AI 通道像素格式日志为 `5`，而厂商程序使用 `16`，取帧失败。
- 用公开 v1.6 MPP 编译的火焰程序，板上 `GC2093` 的传感器类型 `52` 被报告为 `invalid sensor type`。
- 两份失败的电脑端 ELF 保留在 `artifacts/vendor_fire_demo/kitchen_fire_vendor.elf` 和 `artifacts/vendor_fire_demo/kitchen_fire_vendor_v16abi.elf`，只作为故障证据；本节后续没有继续编译 MPP。

从资料包 `D:\111\06_AI应用开发学习资料\02_目标检测相关\04_可执行文件\` 找到与本套件配套的 `ob_det.elf`、`yolov8n_320.kmodel`、测试图及启动脚本。厂商原始 `ob_det.elf` 大小 `12,194,784` 字节，SHA-256 为：

```text
3BDF55AE5B6723BE5887167D1EE6319686CFAF7B787D65160ABC275597E24334
```

厂商原版 ELF 配厂商模型、`bus.jpg` 的单图推理正常结束并生成目标检测结果；先前也完成过摄像头模式的连续推理，日志约 `82–85 ms/帧`，摄像头为 `gc2093_csi2`。这说明**这份厂商 ELF 与板内摄像头链路可以工作**，但不能据此推断它与我们的火焰 KModel 完全兼容。

### 16.2 原模型与数据的实际位置

- 原始权重：`<USER_HOME>\Documents\Codex\kitchen-fire-detection\runs\kitchen-fire\weights\best.pt`。
- 原始 ONNX：同目录下的 `best.onnx`，`37,900,559` 字节；实测输入 `1×3×576×576`，输出 `1×8×6804`，ONNX opset `12`。
- 数据集类别顺序来自 `<USER_HOME>\Documents\Codex\kitchen-fire-detection\datasets\combined\data.yaml`：`fire, smoke, stove, pan`。
- 板上旧 `best.kmodel` 的电脑端备份为 `artifacts/vendor_fire_demo/best.kmodel`。此前用它做单图推理曾检出火焰，也把高亮/烟雾状区域误报为火焰；不能把该结果当作可靠识别效果。

### 16.3 本次生成的兼容文件和方法

为了使用厂商固定的 80 类 YOLOv8 检测程序，新增三个可复现脚本：

| 文件 | 作用 |
| --- | --- |
| `src/kitchen_fire_yolov8/make_vendor80_onnx.py` | 检查原模型输入输出形状，在输出端保留 4 个框通道，把 4 类放入 COCO 类别槽位，其余 76 类填零，生成 `1×84×6804` ONNX 副本。 |
| `src/kitchen_fire_yolov8/compile_vendor80_kmodel.py` | 用 nncase `2.9.0`、nncase-kpu `2.9.0` 和 50 张现有训练图片进行 PTQ，生成 K230 KModel。 |
| `src/kitchen_fire_yolov8/patch_vendor_obdet_labels.py` | 检查四个标签字节串各只出现一次，再在厂商 ELF **副本**中把显示文字改为 `fire/smoke/stove/pan`。 |

四类映射到厂商 COCO 槽位：`fire → 3 (motorcycle)`、`smoke → 9 (traffic light)`、`stove → 10 (fire hydrant)`、`pan → 12 (parking meter)`。这里修改的是输出通道摆放与显示文字；原始 `best.onnx`、`best.pt` 和厂商原版 ELF 均未被覆盖，也没有重新训练权重。

转换环境：`K230-Ubuntu` 中使用已有的 `ghcr.io/kendryte/k230_sdk:v1.9` 容器及 `/home/k230/srtp_nncase29/` 隔离目录；nncase `2.9.0` 和 nncase-kpu `2.9.0` 的 wheel 从嘉楠下载站取得。容器原有 Python 为 `3.8`。默认 pip 镜像超时，随后改用嘉楠提供的 wheel；未修改系统 Python。第一次 ONNX 转换因旧 protobuf 不支持重复字段的 `.clear()` 而失败，脚本改为 `del model.graph.output[:]` 后成功。此错误发生在保存前，没有产生不完整的正式 ONNX 文件。

本次产物均在 `artifacts/vendor_fire_demo/`：

| 文件 | 字节数 | SHA-256 |
| --- | ---: | --- |
| `best_vendor80.onnx` | 39,970,554 | `FE6B49CDA99AF97B6223CFC8C2E1EB116F63DA7BDF75FB298B0DB58CD153A891` |
| `best_vendor80.kmodel` | 12,887,568 | `27059F28D082B7DCF997F05FBB83B922931325D7A09B1CF27EFDE2DF698A9AE5` |
| `ob_det_fire.elf` | 12,194,784 | `11A518CB062C52DB16B65BABE8DA083ABF06C97251141DCC6DC17715708E8A52` |

`ob_det_fire.elf` 与厂商原版等长。定点改动的文件偏移为 `0x983B80`、`0x983BA0`、`0x983BB0`、`0x983BD0`，分别对应上述四个长标签。短词如 `person/car` 在二进制中不是可安全全局替换的独立字符串，因此没有对其做盲目替换。**标签副本尚未通过完整推理验证，不能作为可交付程序。**

### 16.4 板端部署、空间不足和处理

电脑使用 `D:\platform-tools-latest-windows\platform-tools\adb.exe`；板端测试目录为 `/sharefs/srtp_clean/vendor_ob_det/`。当时 ADB 显示 `RTT_K230_ADB device`。

第一次 `adb push` 两个新文件时，板端报告 `No space left on device`；只读检查显示 `/sharefs` 总容量 `252.0M`，已用 `246.7M`，仅剩 `5.3M`，目标文件未完整存在。确认电脑工作区中两个失败 ELF 的备份仍在后，仅删除板端测试目录下的：

```text
/sharefs/srtp_clean/vendor_ob_det/kitchen_fire_vendor.elf
/sharefs/srtp_clean/vendor_ob_det/kitchen_fire_vendor_v16abi.elf
```

删除后 `/sharefs` 剩余 `28.7M`；重新传输 `ob_det_fire.elf` 与 `best_vendor80.kmodel` 成功，板端 `ls -l` 显示的字节数分别与电脑端一致。没有删除厂商原版 `ob_det.elf`、原版模型、开机程序或其他用户文件。上述两份失败 ELF 的**板端副本**已删除，电脑端证据仍可恢复。

### 16.5 真正的单图推理对照测试

最初通过 `adb shell` 发送一次运行命令，没有得到 RT-Smart 日志，旧 `object_det.jpg` 的时间仍为 `1980`；这不构成成功推理证据。之后直接连接 RT-Smart 大核串口 `COM6/115200`，发送 `q` 结束自启程序，进入 `msh /sharefs/srtp_clean/vendor_ob_det>`，在该环境执行以下测试：

1. **新标签 ELF + 新 KModel + `fire_test.jpg`，阈值 `0.50`：**
   - `set_input`、`set_output`、`pre_process image`、`run`、`get_output` 均执行；`OBDet run` 约 `118.6 ms`，随后打印 `OBDet post_process took 16.8 ms`。
   - 紧接着 RT-Smart 报 `Exception 13: Load Page Fault`，`stval=0x0000000101097018`，`sepc=0x0000000200895014`，进程被杀并返回 `msh`。没有得到可信的新结果图。
2. **未改动的厂商原版 `ob_det.elf` + 同一新 KModel + 同一图片，阈值 `0.50`：**
   - 仍在 `post_process` 计时输出后出现**相同** `Load Page Fault`。因此不能把故障归因于四处标签文字修改。
3. **厂商原版 ELF + 厂商 `yolov8n_320.kmodel` + `bus.jpg`：**
   - `OBDet run took 34.2 ms`，`post_process took 5.3 ms`，正常返回 `msh`，无异常。证明原版静态图片路径在配套模型下正常。
4. **厂商原版 ELF + 新 KModel + `fire_test.jpg`，阈值提高到 `0.99`：**
   - 依然出现相同地址的 `Load Page Fault`。简单提高检测阈值不能绕过故障。

目前能严格确认的是：**新 KModel 被成功加载，并完成一次 KPU `run` 和输出获取；厂商程序的完整图片检测流程未通过。**异常在 `post_process` 计时文字之后出现，具体是后处理之后的哪次访问、绘图/保存还是清理过程，尚未定位；“输出形状已经改成 84 类”也不等于所有布局、数值和厂商程序的隐含要求都匹配。不要把 `118 ms` 记作可用 demo 的端到端延迟。

### 16.6 电脑显示视频的可行性与实际状态

用户提出暂时不用板载屏幕，改为在电脑上看视频。手册的目标检测示例只明确描述了板载屏幕实时显示；资料中有 `sample_venc.elf` 及 MJPEG 编码示例，说明存在可研究的视频编码能力，但**当前未验证 UVC、RTSP 或现成网络推流接口，也没有实现 ADB 连续取帧**。因此电脑目前没有收到板端实时视频；这仍是一项待完成工作，不可表述为已经能看实时识别效果。

### 16.7 暂停时的真实状态与恢复依据

- 本次最后一次 COM6 静态测试异常后，串口返回 `msh /sharefs/srtp_clean/vendor_ob_det>`；没有在运行新摄像头推理程序。
- 本节只验证了图片模式，未对新火焰 KModel 启动摄像头模式，未重新编译 MPP，未刷固件，未修改板内 `/bin/fastboot_app.elf`、`/bin/test.kmodel` 或 `/bin/init.sh`。
- 板载屏幕此刻显示什么没有重新拍照确认；不能把先前的正常摄像头画面或乱码状态当作本节结束时的实测状态。
- 若以后恢复工作，先依据本节第 16.5 项的**原版正常、新 KModel 异常**对照定位模型输出和厂商程序的真实接口；静态图片完整跑通后，再验证无屏取帧/电脑显示。不要再次运行第 16.1 项中已证实不匹配的 MPP 实验 ELF。

### 16.8 电脑端产物的复现入口

以下命令记录本次产物的生成方式，执行环境均为 **K230-Ubuntu 内的 shell**；目录中的 `SRTP` 路径映射到 Windows 工作区。nncase wheel 已安装在 `/home/k230/srtp_nncase29/site`，原始模型目录只读挂载。没有必要为查看本记录而重新运行这些命令。

```sh
docker run --rm -u root \
  -v /mnt/c/Users/ASUS/Documents/ChatGPT/SRTP:/work \
  -v /mnt/c/Users/ASUS/Documents/Codex/kitchen-fire-detection/runs/kitchen-fire/weights:/models:ro \
  ghcr.io/kendryte/k230_sdk:v1.9 \
  python3 /work/src/kitchen_fire_yolov8/make_vendor80_onnx.py \
  /models/best.onnx /work/artifacts/vendor_fire_demo/best_vendor80.onnx
```

```sh
docker run --rm -u root \
  -e PYTHONPATH=/opt/nncase/site \
  -e NNCASE_PLUGIN_PATH=/opt/nncase/site \
  -v /home/k230/srtp_nncase29:/opt/nncase \
  -v /mnt/c/Users/ASUS/Documents/ChatGPT/SRTP:/work \
  -v /mnt/c/Users/ASUS/Documents/Codex/kitchen-fire-detection:/source:ro \
  ghcr.io/kendryte/k230_sdk:v1.9 \
  python3 /work/src/kitchen_fire_yolov8/compile_vendor80_kmodel.py \
  /work/artifacts/vendor_fire_demo/best_vendor80.onnx \
  /work/artifacts/vendor_fire_demo/best_vendor80.kmodel \
  /source/datasets/combined/images/train --samples 50
```

```sh
python3 /mnt/c/Users/ASUS/Documents/ChatGPT/SRTP/src/kitchen_fire_yolov8/patch_vendor_obdet_labels.py \
  '/mnt/d/111/06_AI应用开发学习资料/02_目标检测相关/04_可执行文件/ob_det.elf' \
  /mnt/c/Users/ASUS/Documents/ChatGPT/SRTP/artifacts/vendor_fire_demo/ob_det_fire.elf
```

三个命令只产生电脑端副本。板端第 16.5 项的新 KModel 组合**已实测崩溃**，应先定位原因，再决定是否重测；不能把以上“能生成文件”理解为“部署验证通过”。

## 17. 2026-09-24 板端火焰检测 MVP 验证

> 本节更新第 16 节结束时的状态：**板端单图检测和摄像头连续推理现已跑通**。尚未由人眼确认板载屏幕上的实时画面，也尚未实现电脑实时视频；检测准确率没有完成评估。

### 17.1 资料和故障定位

- 对照了本目录及 `D:\111\01_学习手册\` 中的《RT-Thread Smart AI 套件开发手册》，以及 `D:\111\06_AI应用开发学习资料\02_目标检测相关\04_可执行文件\` 的原厂 ELF、模型、脚本。手册的目标检测例程是 YOLOv8n、320 输入、nncase 2.9.0、板载摄像头和屏幕路径。
- 实际 `best.pt` 是 **YOLO11s、四类 `fire/smoke/stove/pan`**。训练与验证标签中没有 `smoke`（类别 1）的正例，烟雾能力不能据此宣称有效。
- 原厂 ELF 加原厂 320 KModel 与同一火焰测试图正常，原厂 ELF 加新 576/640 KModel 在 `post_process` 后发生 RT-Smart `Load Page Fault`；因此曾经的崩溃不能简单归咎于 `.pt` 编译失败。
- 结合 SDK `object_detect_yolov8n/ob_det.cc` 和该 ELF 的 RISC-V 反汇编，定位到 `post_process` 中对**已递增的输出指针**调用 `delete[]`。只在原厂 ELF **副本**中把这一个删除调用替换为两个 NOP，保留析构释放，连续帧不再因悬空指针崩溃。补丁只适用于脚本中锁定 SHA-256 和原始指令字节的这份原厂 ELF；这是 MVP 临时修复，后续应从匹配 SDK 源码正式构建。

### 17.2 模型重编译及结果

- 单独从原始 `best.pt` 导出 320 和 640 ONNX；320 版本在火焰图的原始 ONNX 上 `fire` 最大分数仅 0.013，故选 640。没有覆盖原始 `.pt` 或 `.onnx`。
- 640 模型按原厂程序接口扩充为 `1×84×8400` 输出。PTQ 使用去重、类别均衡的 100 张校准图，涵盖现有 `fire/stove/pan` 与背景；没有可用的 `smoke` 正例。使用 nncase 2.9.0 编译为 `artifacts/vendor_fire_demo/board_mvp_640/best_vendor80_640.kmodel`，SHA-256 `ec475725794417e83b58c661769a915c7a0e7a71ed61a6e2d267ddc7e49096a7`。
- 同一火焰图的 `fire` 最大分数：原始 640 ONNX 为 0.817，nncase KModel 仿真为 0.779；KModel 输出形状和数值均正常。这说明编译后仍保留了该图的检测信号，不代表总体准确率。
- 最终板端 ELF 为 `artifacts/vendor_fire_demo/board_mvp_640/ob_det_fire_mvp_spaces.elf`，SHA-256 `ec2caafd85e56851c44c838b4c9a2a15477cae8b24ac915f1c571f1a0f9a2ac1`。它还把四个 COCO 标签槽位显示为 `fire/smoke/stove/pan`，用空格填满原字符串宽度，避免屏幕文字出现问号。

### 17.3 板端实测和备份

- 在 `/sharefs/srtp_clean/vendor_ob_det/` 中，用最终 ELF、640 KModel 和 `fire_test.jpg` 完成单图推理，正常回到 `msh`。重新生成并拉回的**板端结果图**是 `artifacts/vendor_fire_demo/board_mvp_640/result_fire_labeled_clean.jpg`，图中实际火焰区域显示 `fire` 框，也有一处高亮烟雾状区域误报为 `fire`。该图 SHA-256 为 `6fdbcb5be083342423a5e4a47215c89ce5c5cfd1256b260d49100c822f17abb4`。
- 最终 ELF 和 640 KModel 的摄像头模式在 COM6 连续运行约 12 秒，持续出现采集、推理、`osd draw`、`osd copy` 和 `total time took` 日志，单帧约 238–245 ms（约 4 帧/秒），未见异常；发送 `q` 后停止 VICAP、释放 VB 并返回 `msh`。日志表明绘制流程运行，但没有独立拍摄屏幕确认视觉结果。
- 板端空间不足时，先将旧实验文件 `ob_det_fire.elf`、`best_vendor80.kmodel` 拉回 `artifacts/board_backup_2026-09-24_vendor_ob_det/` 并核对 SHA-256，然后只删除这两个旧板端副本。原厂 `ob_det.elf` 和原有结果图也各有电脑端备份。板内 `/bin` 和启动脚本未改。
- 当前板端测试目录保留原厂 `ob_det.elf`、原厂 `yolov8n_320.kmodel`、最终 ELF、最终 640 KModel、测试图片和结果图。`/sharefs` 剩余约 4.1 MB；需要存放更多模型时，先移走已备份的实验文件，不要删除原厂 `/bin` 文件。

### 17.4 复现板端 MVP

连接 COM6（115200）到 RT-Smart `msh`，在板端执行：

```sh
cd /sharefs/srtp_clean/vendor_ob_det
ob_det_fire_mvp_spaces.elf best_vendor80_640.kmodel 0.5 0.6 None 1
```

`None` 是摄像头输入，最后一个参数 `1` 是调试模式；按 `q` 加回车退出。静态单图验证可执行：

```sh
ob_det_fire_mvp_spaces.elf best_vendor80_640.kmodel 0.5 0.6 fire_test.jpg 0
```

静态结果写为本目录的 `object_det.jpg`。目前实测是无崩溃的板端检测链路和一张真实板端结果图；下一步应现场确认板载屏幕是否看到实时画面及 `fire` 框，再针对误报和缺失的烟雾训练样本改进模型。

记录结束前已再次启动上述摄像头命令，关闭电脑端串口后重新读取 COM6 仍连续看到推理和 OSD 绘制日志，未返回 `msh`；**程序留在板上运行，供现场看屏幕**。若要退出，在 COM6 输入 `q` 并回车；重启板子也会结束本次运行。

### 17.5 用户现场屏幕确认

用户随后提供实机照片，已归档为 `artifacts/vendor_fire_demo/board_mvp_640/board_live_fire_2026-09-24.jpg`（SHA-256 `33a68abb41cdff67e091f82870fbe5e5e00e342fb0218666ef8f099f5c8ccdff`，与原临时图片逐字节一致）。照片显示板载屏幕的实时摄像头画面，以及打火机火苗区域附近的蓝色检测框，屏幕文字为 `fire 0.50`。这补齐了**板载屏幕实时显示火焰框**的现场证据，可将当前成果表述为板端视觉检测 MVP。它是一张现场照片，不是持续准确率或误报率评估；第 17.3 节的静态图误报仍需处理。

## 18. 阶段总结与下一轮训练交接

板端视觉检测 MVP 已由串口日志、板端生成的单图结果和用户现场屏幕照片共同证实。当前重点转为数据质量、火焰误报与烟雾正例缺失。为避免把旧验证集和旧训练脚本的问题带入下一轮，已另写两份可供 DSH 直接接手的文档：

- `<PROJECT_ROOT>\2026-09-24_SRTP阶段总结与DSH交接.md`：当前事实、全部关键绝对路径、板端状态、备份与接手顺序。
- `<PROJECT_ROOT>\2026-09-24_火焰烟雾模型训练计划.md`：数据审计、标注、版本化切分、训练命令、KModel 转换及板端对照验收。

两份文件是**规划与交接，不代表新模型已训练**。下一轮先锁定独立测试集并记录基线，再建 `combined_v2` 和新训练 run；不要原样运行会删除重建旧 `combined` 的 `prepare_data.py`，也不要覆盖已跑通的 `best.pt`、KModel 或 ELF。

环境传感器状态另须明确：SensorFusion Shield 物理安装不等于温度/烟雾通信已通。**本轮未做 I²C 总线扫描、设备地址应答、寄存器读取或传感器数值变化测试，温度与烟雾/气体传感器连通性未知。** GC2093 摄像头的运行日志不属于这项证据。DSH 应先依主板/副板资料确认具体器件和接口，再独立验证并留存日志。

> 上段只记录第 18 节写入时的状态。随后已完成两颗传感器验证，见第 19 节。

## 19. 2026-09-24 至 25 副板传感器与 OLED 开发过程

1. **核对实物和手册。** 底板 Arduino SCL/SDA 对应 K230 GPIO44/45；副板有 SHT31、BMP280 和 128×64 OLED。AHT20 芯片已脱落，不参与本轮测试。资料为 `D:\111\01_学习手册\K230-底板.pdf`、`k230-副板.pdf` 和用户提供的 BMP280 手册。
2. **定位 I²C 失败。** BMP280 约有 3.3 V，但最初读 ID 失败。GPIO 探针发现拉低 SCL 或 SDA 会让另一线同时变低；取下副板后，底板两线可以独立变化，问题集中在副板跳线。
3. **修正跳帽。** 两只跳帽原先都接 `4.7`，副板图纸 P2 的两个 `4.7` 脚实际相连。用户断电取帽后测到 SCL/SDA 不再导通；最终改为**上排 `2.2–SDA`、下排 `SCL–4.7`**。
4. **验证 BMP280。** `0x76` 地址读到 `0xD0=0x58`，`0x77` 无应答；小核 GPIO 模拟 I²C 和大核硬件 I²C3 都读到正确 ID。读取原始值与校准参数后，算出约 26–27°C、952–953 hPa；临时引脚复用已恢复。
5. **验证 SHT31。** `0x44` 读到一次测量原始字节 `68 A8 A6 6E 8F ED`，温湿度两个 CRC 均通过，换算为 **26.54°C、43.19%RH**；后续连续采样仍为 `CRC OK`。
6. **完成 OLED 显示。** 把曾出现大字和乱码的方案改为普通字号五行：`SHT T`、`SHT H`、`BMP T`、`BMP P`、`WIFI --`。用户实机照片 `<LOCAL_TEMP_IMAGE>` 显示五行清楚；Wi-Fi 只是预留，RW007 尚未测试。
7. **整理代码并联测。** 源码 `<PROJECT_ROOT>\src\sensor_mvp\` 按 GPIO I²C、SHT31、BMP280、SSD1306 和主循环拆分。板端 `/sharefs/srtp_clean/sensor_mvp/sensor_mvp_oled` 曾连续输出 SHT31 `26.26°C / 45.66%RH (CRC OK)`、BMP280 `26.73°C / 952.83 hPa`；大核火焰视觉程序同时持续推理和绘制。详细寄存器、日志与启动方式见 `2026-09-24_SHT31_BMP280_实机调试记录.md`。

**已经跑通：SHT31 温湿度、BMP280 温度与气压、OLED 实时显示，以及与视觉程序同时运行。** AHT20 掉件、Wi-Fi 未测试，与上述成功结果分开记录。

## 20. 2026-09-25 K1/K2 按键开发过程及当前暂停点

1. **按手册查引脚。** 副板 J1 的 `K1–D9`、`K2–D10` 是可选跳线；底板 CN2 将 D9 接 GPIO43、D10 接 GPIO14。两键支路有 10 kΩ 上拉，按下为低。图纸还有 RW007 复位/配置键的同名编号，不能仅凭 `K1/K2` 名字判断走线。
2. **接跳帽并实测。** 用户断电后分别接 `K1–D9` 和 `K2–D10`，原 SDA/SCL 跳帽不动。按 K1 时 GPIO43、按 K2 时 GPIO14，均测得 `0x8000018F → 0x0000018F → 0x8000018F`。**两键物理输入已验证。**
3. **编写控制程序。** `src/key_controls/` 中 `key_input.c/.h` 只读按键并消抖；`vision_key.c` 拟用 K1 启停大核视觉 ELF，关闭时向子进程输入 `q`；`sensor_key.c` 拟用 K2 启停小核传感器程序。新版副板程序收到 SIGTERM 后清 OLED 并释放 GPIO。两个控制器都已交叉编译，**板端启停尚未验证**。
4. **处理存储不足。** `/sharefs` 只有 252 MB，一度占满。核对板端旧 `ob_det.elf` 与本机 `artifacts/board_backup_2026-09-24_vendor_ob_det/ob_det.elf` 的 SHA-256 一致后，才删去板端这份已备份的旧 ELF 及一次失败复制留下的不完整文件；当时空余约 12 MB。现用视觉 ELF、640 KModel 和旧副板程序均保留。
5. **推送卡住并暂停。** 推送新 `sensor_key` 约 3.9 MB 时 ADB 长时间无返回，新的 ADB shell 和大核读取该共享目录也未返回；原因未确定，板端目标文件可能不完整，不能执行。视觉程序已用 `q` 正常退出；用户决定关板暂停。下次先只读检查 ADB、`df -h /sharefs`、文件大小，再决定清理不完整文件与继续推送。用户准备增加内存卡；须先核实新卡挂载点，插卡不会自动扩大现有 `/sharefs` 分区。详细恢复步骤在 `2026-09-25_K1_K2按键控制实机记录.md`。

## 21. 2026-09-25 重启后继续按键控制开发

1. **先检查旧故障。** 板子重新上电后 ADB、COM6 恢复；`/sharefs` 仍是 `/dev/mmcblk0p4` 的 252 MB vfat 分区，可用约 10.7 MB。上次推送的 `sensor_key` 只有 1,376,256 字节，确认是不完整文件后仅删除这一份，空余恢复约 12 MB。识别到的整卡 `mmcblk0` 约 15 GB，但 `/sharefs` 分区未自动扩容。
2. **减少板端写入。** 使用板端现有的 `/lib/ld-linux-riscv64xthead-lp64d.so.1`，另编译动态链接小核程序：`sensor_key_dynamic` 为 19,928 字节，`sensor_mvp_oled_dynamic` 为 28,752 字节。先在 `/tmp` 运行 K2 只读 `probe`，正常打印 `K2 GPIO14 input ready`；旧静态产物继续保留作备份。
3. **逐个部署并核对。** 动态小核程序和 300,568 字节的 `vision_key.elf` 已传到板端 `/sharefs/srtp_clean/key_controls/`，三份板端 SHA-256 均与电脑端一致；当前 `/sharefs` 可用约 11.7 MB。没有覆盖现有火焰视觉 ELF、KModel 或旧版传感器程序。
4. **启动只读探测和控制器。** 大核执行 `vision_key.elf probe` 打印 `K1 GPIO43 input ready`；Linux 小核的 `sensor_key` 与大核的 `vision_key.elf` 已分别启动，当前初始状态均为 OFF，等待用户实体按 K2/K1。**此时只验证控制器可启动；按键触发子程序启停尚未验收。**
5. **自检 K2 子程序流程。** 新增 `selftest` 模式让小核控制器直接执行两轮“启动副板程序→采样→SIGTERM 停止”。两轮均返回成功且无残留子进程；日志显示 SSD1306 `0x3C`、SHT31 `24.99°C / 52.32%RH (CRC OK)`、BMP280 `25.46°C / 952.31 hPa` 等连续数值。它验证了进程与传感器流程，不代替实体 K2 触发验收。
6. **修正并自检 K1 子程序流程。** 首轮自检中视觉 ELF 已响应 `q`、停流并释放 VB，但控制器始终等不到 `waitpid`，第二轮没有启动。检查本机 RT-Smart SDK 的 `sys_waitpid` 实现，发现传入 `NULL` 状态指针会被拒绝；改为传有效的本地 `int status` 后，板端两轮“启动视觉 ELF→连续推理→输入 `q`→停流释放 VB”均成功，串口分别打印 `K1: vision stopped cleanly`。现已重新启动两个控制器等待实体按键验收。
7. **实体按键首轮验收。** 用户短按 K2 后确认小屏能开启数据、再次按能关闭；板端控制器日志分别出现 `K2: sensor started` 和 `K2: sensor stopped`，新一轮传感器日志仍有 SHT31 CRC OK 与 BMP280 温度、气压。K1 第一次按下能启动大屏视觉，第二次却关不掉。由此可确认 K2 实体开关通过，K1 只通过启动测试，不能写成双向控制完成。
8. **定位 K1 第二次按键失效。** 对照 `/home/k230/k230_sdk_v1.6/src/big/rt-smart/kernel/rt-thread/components/lwp/lwp_pid.c`：板端 `waitpid` 实现忽略 `WNOHANG`，视觉子进程启动后，控制器主循环里的 `waitpid(child, &status, WNOHANG)` 一直阻塞，按键轮询不再执行。这也解释了为什么两轮定时自检能通过、实体按键却不能关闭。已从主轮询移除该等待；停止视觉时仍在发送 `q` 后等待子进程回收。重编译的 `vision_key.elf` 以**新文件名** `/sharefs/srtp_clean/key_controls/vision_key_v2.elf` 推送，保留原文件。待重启板子结束旧控制器，再做 K1 实体开/关验收；目前不能声称 K1 已修好。
9. **核对修复版部署与重启边界。** 电脑端修复版和板端 `vision_key_v2.elf` 的 SHA-256 均为 `d50aa814db7df2ea4cd0cc19b70b82be2716f9ff43997d081effd618b2cf04e7`，文件 300,608 字节；`/sharefs` 剩余约 11.4 MB。执行 `adb reboot` 后小核 ADB 恢复，但 COM6 仍未给出 RT-Smart `msh` 提示，不能据此认为大核旧控制器已退出。因此请用户断开两根 USB 再重新上电；在确认 `msh` 可用以前，不运行或宣称 v2 实体验收通过。

## 22. 2026-09-25 32GB microSD 扩容 `/sharefs`：备份、迁移、开机验收

1. **先对照手册与实物。** `D:\111\01_学习手册\RT-Thread Smart AI 套件开发手册.docx` 说明 eMMC 启动拨码为 BOOT0 ON、BOOT1 OFF，SD 启动为两者 OFF；本次保持现有 eMMC 启动拨码不动。资料目录 `D:\111\04_开发板配套固件\readme.txt` 明确称 `sysimage-sdcard.img` 不能用于 SD 启动，`sysimage-sdcard_v1.img` 可启动但未严格测试；本次没有写入这两份镜像。
2. **识别新卡问题。** 32GB microSD 插电脑后是磁盘 2（USB `Mass Storage Device`，31,266,439,168 字节），Windows 能看到约 256MB 的 `G: K230_APP`，其中已有 49 个左右的原厂样例 ELF 和 `test.kmodel`；卡上原有四个约 20/50/160/256MB 分区。插板并保持 eMMC 启动后，Linux 将 14.6GiB eMMC 识别为 `/dev/mmcblk0`，将新卡识别为 `/dev/mmcblk1`（29.1GiB，`device/type=SD`）。原卡 GPT 仍指向约 512MB 镜像末尾，板端提示 `GPT array CRC is invalid`，没有生成 `mmcblk1p1` 等分区，因此不能直接挂载。
3. **先备份再改卡。** 将原卡前 513MiB（覆盖原 512MiB 系统镜像及分区）只读备份到 `<PROJECT_ROOT>\artifacts\card_backup_2026-09-25\microSD_first_513MiB.img`，大小 **537,919,488 字节**；本机与板端对同一 513MiB 计算的 SHA-256 均为 `a6be8b18739b71ecc31555a645f2fe767f5d96015320a2ec02231c3daaeead2a`。旧 ADB 的 `exec-out` 不支持、原始 PTY 会改写非零二进制字节，因此改用逐块压缩加 Base64 传输，并用板端全段哈希复核。另将原小核 `/etc/init.d/rcS` 拉回为同目录 `rcS_emmc_original`，SHA-256 `8a18cb124665ff4c5cb2b116323db41bc32351b8cd24552d88e83cc58d273d72`。
4. **建立并实测数据分区。** 只对确认是 SD 的 `/dev/mmcblk1` 执行 `parted -s /dev/mmcblk1 mklabel msdos mkpart primary fat32 1MiB 100%`，再执行 `mkfs.vfat -F 32 -n SRTP_DATA /dev/mmcblk1p1`；新分区 UUID 为 `3AE9-B11C`。先挂载到 `/mnt/srtp_sd`，写入一份 `healthcheck.txt`，卸载再重挂后读取一致，`df -h` 显示 29.1GiB 可用。原卡的预装系统分区已被改为数据分区；可用第 3 步的原始备份恢复原状。
5. **解决真正的共享区瓶颈。** 仅挂到 `/mnt/srtp_sd` 时，大核仍受 eMMC `/sharefs` 的 252MB 限制。检查原厂 `/etc/init.d/rcS` 确认它在 eMMC 启动时执行 `mount /dev/mmcblk0p4 /sharefs`，然后启动 `./sharefs` 服务。把原 `/sharefs` 的 **85 个文件**用 `cp -r /sharefs/. /mnt/srtp_sd/` 复制到新卡，并用 `cmp` 逐文件检查 **85/85 一致、0 个不匹配**；eMMC 上的原件未删除。
6. **最小开机修改与回退。** 工作区候选脚本是 `<PROJECT_ROOT>\src\sd_data\rcS_sd_sharefs`，与原 `rcS` 的差别仅在 eMMC 启动分支：若发现 UUID 为 `3AE9-B11C` 的 `/dev/mmcblk1p1`，优先把它挂到 `/sharefs`；若卡缺失或挂载失败，仍挂原 `/dev/mmcblk0p4`。板端另留 `/etc/init.d/rcS.before_srtp_sd`（哈希与原文件相同）。新 `rcS` 通过 `sh -n` 语法检查，安装后 SHA-256 为 `ecf3e0e4e0f49337965f0849faac0e779ac6ab7dbf5d83c36d71cd976b31a571`。
7. **重启验收。** ADB 重启后 `/dev/mmcblk1p1 on /sharefs type vfat`；`df -h /sharefs` 显示 **29.1GiB 总量、241.3MiB 已用、28.9GiB 可用**。原火焰 ELF、640 KModel 和 `sensor_key` 仍在原路径，小核 K2 控制进程自动启动。COM6 向原厂视觉程序输入 `q` 回到 `msh` 后，执行 `ls /sharefs/srtp_clean/vendor_ob_det`，大核确实看到 `best_vendor80_640.kmodel`、`ob_det_fire_mvp_spaces.elf` 等文件，证明大核共享区也已扩容。K1 的断电自启仍是独立待办，不能由扩容结果推断其已修好。
8. **使用和边界。** 今后把模型、ELF、数据放入 `/sharefs/srtp_clean/` 即落在 32GB SD 卡，可由小核 ADB 和大核 RT-Smart 共同访问。现在拔卡会按脚本回退到 eMMC 上迁移时的旧快照，之后写到 SD 卡的新文件不会自动同步回 eMMC；运行中不要拔卡。启动日志曾有 `mmc1: Problem switching card into high-speed mode!`，但本次格式化、约 240MB 复制、85 个文件逐字节核对和重启读目录均成功，长期稳定性尚未单独测量。

## 23. 2026-09-25 K1 大核自启：原厂镜像定点替换与冷启动复查

1. **从失败的整包重建恢复。** 先前用本机 SDK 重建完整大核 p1 后，大核未正常启动，且新镜像缺少原厂 GC2093 配置。已把刷写前备份 `artifacts/board_backup_2026-09-25_autostart/rtt_partition_p1.img` 写回 `/dev/mmcblk0p1`，板端读取 SHA-256 与原备份 `83f8286ae132a917c72c15cff004f695d35de1a9194024cf93f012b1532f9cf6` 一致；重启后 COM6 恢复 `RT-SMART Hello RISC-V` 和 GC2093 初始化日志。详情见 `2026-09-25_K1_K2按键控制实机记录.md`。
2. **只替换原厂 ROMFS 内的启动 ELF。** 原厂 `/bin/init.sh` 会运行 `/bin/fastboot_app.elf /bin/test.kmodel`。从原始 p1 提取的 `fastboot_app.elf` 为 9,707,664 字节，SHA-256 `025e795058fe7678dfc18cdf6bfb0780d403060d42065d9dbe4ab2525e15deec`。已验证的 K1 v2 控制器 `vision_key_v2.elf` 为 300,608 字节，SHA-256 `d50aa814db7df2ea4cd0cc19b70b82be2716f9ff43997d081effd618b2cf04e7`。`src/key_controls/patch_stock_romfs.py` 严格核验输入哈希，把 K1 ELF 补零至原厂文件长度，仅替换解压 ROMFS 中该文件的字节槽位，保留其他原厂字节；原始镜像重新封装可逐字节复现。板端先运行补零 ELF 的 `probe` 和正常参数，均可启动。
3. **刷写和热重启验收。** 定点补丁产物为 `artifacts/board_backup_2026-09-25_autostart/rtt_partition_p1_k1_stock_patch.img`，20,971,520 字节，SHA-256 `66d0a7088531ddbe96ce3cf2d877d0b9dda5bdab93c3d209ee74f3da5818f97b`。电脑、板端临时文件和刷入后的 `/dev/mmcblk0p1` 哈希一致。`adb reboot` 后 COM6 输出 `OpenSBI v0.9`、`RT-SMART Hello RISC-V`、`msh />K1 controls vision; initial state OFF; q exits controller`，说明这条启动路径执行到了 K1 程序。用户连续短按副板 K1 三次，现场反馈“三次都正常”；串口还记录了视觉正常停流、再次启动和 GC2093 连续帧。**这只证明软件重启后的 K1 开/关/再开。**
4. **ON/OFF 开关复查暴露差异。** 用户当时保持 USB 插着，只拨板上的 ON/OFF 开关，反馈“K1 还是打不开”；此前将此操作称为“拔掉 USB 断电”是错误记录。那次 ADB 正常，`/dev/mmcblk1p1` 仍挂到 `/sharefs`，小核 `sensor_key` 在运行，p1 哈希仍为 `66d0…f97b`；接入 COM6 后没有新的异步输出。再由 ADB 触发软件重启并监听 COM6，重新看到 K1 程序自启。仅凭接入串口后没有输出，不能断定大核没启动。
5. **核对手册适用性并改为抓实机日志。** `D:\111\01_学习手册\RT-Thread Smart AI 套件开发手册.docx` 的“连接显示屏”部分提到显示屏背板 K1 长按 3 秒。用户明确指出当前实物没有找到这个键，且先前成功显示时从未按过；因此不能把该手册条目直接套用于当前屏幕，也不能把黑屏归因于它。下一步在两根 USB 均断开后预先监听 COM6，再按手册的 OTG 先接、debug&5V 后上电顺序抓完整冷启动串口，确认大核与 K1 程序实际启动状态；实体冷启动验收仍未通过。
6. **纠正一次误判。** 用户曾在长按底板矩阵 KEY1 后，再按副板 K1 看到视觉；但此前我已经执行过一次 `adb reboot`。所以那次画面仍属于**软件重启后的成功**，不能证明矩阵 KEY1 能唤醒屏幕。用户随后保持 USB 插着再拨 ON/OFF 开关，副板 K1 仍无法打开视觉。用户明确要求保留已经验证过的 K1 开关功能，只解决上电时程序无法使用的问题；未修改 K2 程序、按键接法或增加“长按重启”功能。
7. **受控实体冷启动通过。** 在 COM6 预先持续监听的情况下，用户拔掉两根 USB，等待约 10 秒，按照手册“先 OTG、后 debug&5V”的顺序重新接线；这段过程**没有执行 `adb reboot`**。串口依次记录 `[COM6 disconnected]`、重新连接、`OpenSBI v0.9`、`RT-SMART Hello RISC-V`、`msh />K1 controls vision; initial state OFF`。用户短按副板 K1 后，串口出现 `K1: vision started, pid=2`、GC2093 初始化和持续约 220 ms/帧的推理日志，用户确认大屏出现。再次短按，串口出现 `kd_mpi_vicap_stop_stream`、VB 释放、`K1: vision stopped cleanly`；再次按又出现 `K1: vision started, pid=3`，用户确认“能关能开”。这一次满足**断电上电后 K1 程序自启及实体开/关/重开**验收。
8. **原因边界与现用方法。** 之前 K1 无反应发生在 USB 一直插着、只拨板上 ON/OFF 开关的操作后；本轮成功是**两根 USB 都拔掉再接回**。两者并非同一断电条件，后者更接近完整重新上电；但开关具体控制范围及差异根因仍待核对，不能断言先接 OTG 的顺序是唯一原因。当前可复现的启动方法是断开两根 USB 约 10 秒，再先接 OTG、后接 debug&5V，等系统启动后按副板 K1。若还需仅靠 ON/OFF 开关启动，先确认其丝印与电路作用，再抓该路径的 COM6 日志。不要改 K2 和已可用的 K1 控制代码。底板矩阵 KEY1 与副板 K1 不能混为同一个键。

## 24. 2026-09-25 右上角 ON/OFF 开关启动的 K1 故障与定点修复

1. 用户澄清之前失败时两根 USB **一直插着**，只是拨板子右上角无丝印的 ON/OFF 开关；它不在 BOOT0/BOOT1 或 RESET/PWR-ON 旁。开关拨 OFF 后，电脑端 ADB 设备及 COM5/COM6 都消失，说明它至少切断了这些板上接口/供电。不能把这种操作和“两根 USB 拔掉”混为同一次实验。
2. 预先监听 COM6，用户保持 USB 插着拨 ON。串口显示 `OpenSBI v0.9`、`RT-SMART Hello RISC-V`、`K1 controls vision; initial state OFF`，说明大核和 K1 控制器**确实自启**。但随后在小核共享区服务的 `<ipcm>` 输出之前，先出现 `K1: vision started, pid=2`，却没有视觉子程序应有的 `case ...`/GC2093 初始化；再按 K1 时出现 `K1: vision did not exit within 10 seconds, pid=2`，用户确认画面仍打不开。结合 `vision_key.c` 的启动顺序，推断 GPIO43 在上电时出现按键事件，视觉子进程在 `/sharefs` 未就绪时失败退出，控制器仍把旧 PID 当作运行中；这个推断由时间顺序和日志支持，尚未直接抓到子进程的 `chdir` 失败行。
3. 只改 `src/key_controls/vision_key.c`：常规模式先轮询确认 `/sharefs/srtp_clean/vendor_ob_det/ob_det_fire_mvp_spaces.elf` 和 `best_vendor80_640.kmodel` 可访问，之后才初始化 K1 并接收按键；`probe`/`selftest` 保持可直接运行。子进程 `chdir`/`execv` 失败时输出错误；`stop_vision()` 遇到 `waitpid` 返回负数时清掉已退出的旧子进程状态，避免 K1 永久卡在“运行中”。K1 原有短按开/关和向视觉程序发送 `q` 的流程保留。**未改 K2 代码、K2 启动项或任何跳线。**
4. 新版 `artifacts/sensor_validation/key_controls/vision_key_v3.elf` 300,688 字节，SHA-256 `51cd2188b94764639acc776e7a0cdfec8c2b7385de579972b359d47915804607`。板端 `/sharefs/srtp_clean/key_controls/vision_key_v3.elf` 哈希一致；COM6 `probe` 打印 `K1 GPIO43 input ready`，`selftest` 两轮均显示视觉启动、摄像头运行和 `K1: vision stopped cleanly`。手动板端测试没有替代开机验收。
5. 用 `patch_stock_romfs.py` 从**已核对哈希的原始大核 p1 备份**生成候选 `artifacts/board_backup_2026-09-25_autostart/rtt_partition_p1_k1_v3_ready.img`（20,971,520 字节，SHA-256 `fc010c3b6e9de506bff25e26ce021a24f2fa8bb7637713360660256d9a0ab1e8`）。脚本仅替换原厂 ROMFS 的 `fastboot_app.elf` 字节槽位，其他解压后字节不变并做镜像往返校验。推送 `/tmp` 前后哈希相同；刷写前 `/dev/mmcblk0p1` 的哈希 `66d0a7088531ddbe96ce3cf2d877d0b9dda5bdab93c3d209ee74f3da5818f97b` 与本机 v2 镜像一致；刷入 v3 后板端读回哈希为 `fc010c3b6e9de506bff25e26ce021a24f2fa8bb7637713360660256d9a0ab1e8`。原厂备份和 v2 镜像都保留。
6. **首次 ON/OFF 实体复查：启动通过。** 保持两根 USB 插着，用户拨右上角开关 OFF 约 10 秒，再拨 ON；此间没有 `adb reboot`。COM6 顺序记录 `OpenSBI v0.9`、`RT-SMART Hello RISC-V`、`K1: waiting for vision files on /sharefs`、`<ipcm> phys ...`、`K1 controls vision; initial state OFF`，证明 v3 在共享区就绪前没有抢跑。板端 `/dev/mmcblk1p1` 仍挂到 `/sharefs`，`sensor_key` 仍自启，p1 哈希保持 `fc010c3b…ab1e8`。随后用户短按副板 K1，串口出现 `K1: vision started, pid=2`、真正视觉 ELF 的 `case /sharefs/...`、GC2093 初始化和持续约 220 ms/帧的推理；用户现场确认“现在可以打开了”。因此**日常 ON/OFF 启动后的 K1 开启已通过**。
7. **今晚收尾反馈。** 用户随后自行再次拨 ON/OFF 复测，反馈“应该可以了”，并要求记录后结束今晚调试。这个反馈支持本轮问题基本缓解，但未逐项说明这次的“关闭→重开”画面及第二次开关循环日志；因此仍把可核对的验收写为第 6 项的“普通 ON/OFF 启动后 K1 自启并打开视觉”，把用户复测结论保留为其原话。今晚不再改板、不再刷写、不再要求继续测试。
8. **2026-09-26 用户独立复测。** 用户再次按日常方式通过右上角 ON/OFF 开关开机，自行测试后明确反馈：**“K1 2都能用”**。这补充了隔日、无需我通过 `adb reboot` 干预的现场验收：该开机方式下，副板 K1 视觉控制和 K2 传感器/OLED 控制均可用。本次没有新的串口抓取或逐次按键日志，不额外推断稳定运行时长、检测准确率或其他未测功能。

## 25. 2026-09-26 Wi-Fi 联网实测与模型基线复核

1. **区分两条 Wi-Fi 路径。** 对照 `D:\111\01_学习手册\RT-Thread Smart AI 套件开发手册.docx`、`D:\111\01_学习手册\k230-副板.pdf` 和实机照片：主板自带 Realtek 8188FU USB 网卡，当前 Linux 中为 `wlan0`；副板另有 RW007 模块，经 SPI/中断/复位线与底板相连。副板上丝印 `WIFI` 的小按钮在原理图中是 RW007 的 `FLASH` 按钮，正常联网不需要按。手册的 `Programmers/12345678` 是示例热点，不是板子的默认密码。
2. **核对当前驱动。** ADB 上 `lsmod`/`dmesg` 显示 `rtl8188fu` 已加载并创建 `wlan0`；当前 Linux 未列出 SPI 设备节点，也没有发现 RW007 网络接口。配套固件说明 `D:\111\04_开发板配套固件\readme.txt` 单列带 RW007 和不带 RW007 的 CanMV 镜像。今天没有更换固件、刷写 eMMC 或改动现有 K1/K2 启动配置，因此以下联网成功仅归于主板 USB 网卡，**不能当作 RW007 已跑通**。
3. **扫描与连接。** 在当前固件上执行 `ifconfig wlan0 up`、`iwlist wlan0 scan`，能扫描到附近热点。使用用户提供的热点凭据生成 `/tmp/srtp_wifi.conf`，权限设为 `600`；经 `wpa_supplicant -D nl80211 -i wlan0 -c /tmp/srtp_wifi.conf -B` 和 `udhcpc -i wlan0 -n -q` 完成连接。本文和项目源码均不记录热点密码。
4. **连通性结果。** `wlan0` 获得 `10.93.81.121/24`，默认网关 `10.93.81.154`；网关、外网 IP `1.1.1.1`、`www.baidu.com` 各 ping 2/2 成功，DNS 解析正常。稍后复查仍关联热点并保有 IP。`/tmp` 配置和本次手工启动是临时状态，**尚未验证断电后自动联网**。副板 OLED 的 `WIFI --` 仍为固定占位，没有接入实时状态。
5. **独立复核旧训练数据。** 只读检查 `<USER_HOME>\Documents\Codex\kitchen-fire-detection\scripts\prepare_data.py`：原脚本把原始 `test` 映射进 `combined/val`，并会删除重建 `combined`，不可原样重跑。逐张 SHA-256 比较，`kitchen-images` 64 张有标签图片与 `combined/images/train/manual_*` 对应副本 **64/64 完全相同**，不能作为独立测试。直接统计 `combined/labels/train` 与 `val`：`smoke`（类别 1）正例框均为 **0**；这份四类模型不能声称已经训练好烟雾检测。
6. **复核 DeepSeek 的干净集边界。** `model_work/data/test_suite_v1` 的 239 张 `cctv_clean` 图来自 **48 个场景组**（47 组各 5 张、1 组 4 张变体），并非 239 个独立现场场景。标签分布为仅火 98、火与烟同时存在 21、仅烟 120；本次人眼抽查 `cctv_clean__fire_detected_var10_img1.png` 与 `cctv_clean__smoke_detected_var23_img5.png`，类别 0/1 在这两张样本上的语义与 `fire/smoke` 一致，但这不等于完成全部标签质检。该集是异于厨房视角的 CCTV 场景；它和本项目厨房板端画面的准确率不能直接等同。
7. **独立复跑旧 `.pt` 的图像级指标。** 校验基线 `best.pt` SHA-256 为 `39d84feaef74d0c8ed9263320c7e60a5fc8e201b89a2c07f2b0feb6428dfc3ee`。使用 YOLO 推理 `imgsz=576`、`conf=0.5`、NMS `iou=0.6`、CUDA，对上述 239 张逐图统计 `fire` 类：有火 119 张检出 86、漏检 33；无火但有烟 120 张误报 30、未误报 90。结果与 DeepSeek 报告中的 **86/119** 和 **30/120** 一致。按文件名场景组计，24 个纯烟组中 **16 组至少有一张出现 fire 误报**。这里的 `30/120` 是**跨域纯烟图上的 fire 误报率**，不是普通无火厨房的综合误报率；五张变体同组相关，不能按 120 个独立场景计算置信区间。
8. **下一步模型工作。** 保留已上板的 `.pt`、640 KModel 和 ELF 作为不可覆盖基线。先采集同一板载摄像头视角的正常烹饪、高光/LED、蒸汽/油烟、真火与真烟短片，按原始视频或场景分组切分；先人工核对标注、补烟雾正例，再建版本化 `combined_v2`，避免旧数据跨 train/val/test 泄漏。用同一冻结测试集比较旧/新 `.pt` 的漏检和每场景误报，改善后再转换 KModel 并做板端对照。**本节没有训练新模型，也没有声称 RW007 已联网。**

## 26. 2026-09-26 灶火数据集收件与训练交接

1. **Wi-Fi 今日结果已记在第 25 节。** 主板 `wlan0` 扫描、连接、DHCP、外网和 DNS 测试通过；副板 RW007、OLED 实时 Wi-Fi 状态和断电自动联网未验收。此处只做索引，不重复保存热点凭据。
2. **收到针对灶火的数据。** 用户下载并解压 Roboflow `gc_kitchen_annotation` v2，实际路径为 `D:\gc_kitchen_annotation.v2-version_two.yolov11`。其 YAML 四类为 `flame-reflection / flame-under-pot / gas-stove-flame / no-flame`，比旧通用火焰集更直接覆盖正常做饭的小火和难负例。只读检查得 train 1,524、valid 215、test 109，共 1,848 张增强后文件；按源文件名前缀为 1,086 组，标签配对齐且无格式错误。已抽看一张真实商用厨房图及其四个框，未将单张抽检当成全量验收。
3. **训练尚未开始。** 今日新建 `model_work/docs/VISION_DATA_CURATION_20260926.md`（原始来源、审计和收件记录）与 `2026-09-26_视觉数据整理与DeepSeek训练执行单.md`（统一标签、构建、分组防泄漏、训练、独立评估、板端门槛）。旧 9,389 张通用火焰仍作候选；不能把 1,848 个增强文件当 1,848 个独立厨房场景。原始数据、旧权重与板端程序未覆盖。
4. **审查意见落地。** 将 5 份“仅供参考”的衍生审查稿移入 `reviews/`，保留原始实机记录和训练执行单的主目录入口。GC 数据 D 盘原件 3,699 文件、76,254,966 字节已逐文件 SHA-256 校验复制到 C 盘 `artifacts/dataset_backup_2026-09-26_gc_kitchen_v2/`，清单为 `MANIFEST_SHA256.json`。已知刷入后大核不能正常启动的 `rtt_partition_p1_k1_autostart.img` 旁加 `DO_NOT_FLASH.txt` 警示，不移动或改名历史镜像。
5. **缩小首轮结论。** 灶火集按源图组是 train/valid/test 762/215/109 组；test 锅下火 79 组、燃气灶火 77 组，但拍摄场景独立性未验。按拟定类 1、2→fire 映射预演，各 split 空标签整图都是 **0**，说明还缺纯无火厨房负例。执行单已改为“首轮仅验收灶火检测”，爆炒/异常持续起火列为视频数据待补；要求 DeepSeek 输出映射前后框数、空标签数、源图组数及独立拍摄场景数，缺少的项目标“未验收”。

## 27. 2026-09-27 模型训练、手工补标与外部照片对照经过

1. **先核对上一版 v3 的低分。** DeepSeek 构建的四类 `kitchen-vision-v3-640` 训练 55 轮，最佳第 35 轮验证集 P=0.307、R=0.369、mAP50=0.330、mAP50-95=0.159；用户据此要求 Codex 亲自训练。该数字来自 `<USER_HOME>\Documents\Codex\kitchen-fire-detection\runs\kitchen-vision-v3-640\results.csv`，四类指标不能直接与后续单类模型横比。
2. **训练单类 v1 并冻结测试。** Codex 用 v3 最佳权重为起点，另建只含 `fire=0` 的 `D:\SRTP_Datasets\kitchen_fire_binary_codex_20260927`：train 2382、val 243、test 286。YOLO11s/640/batch 4/AdamW，最多 60 轮，patience 15；实际 32 轮早停，最佳第 17 轮验证集 P=0.632、R=0.792、mAP50=0.570、mAP50-95=0.189。完整 286 张 test 为 P=0.715、R=0.649、mAP50=0.639、mAP50-95=0.182；其中 146 张厨房子集在 conf=0.25 时有火检出 70/71、定位 IoU≥0.5 为 54/71、无火误报 0/75。细节见 `model_work/docs/FIRE_BINARY_CODEX_TRAIN_20260927.md`。
3. **复查早期手标图并补漏。** `<USER_HOME>\Documents\Codex\kitchen-fire-detection\datasets\kitchen-images` 共 64 张，历史上 64/64 曾被复制进旧 train，不能作独立测试。逐图检查发现 25 张空标签实际可见火，其中正常烹饪小火/蓝焰 9 张；另建 `model_work/data/manual_kitchen_labels_v2_20260927/labels/`，为 25 张补 36 个 fire 框。原图及原标签均未改，逐图 SHA256 检查通过。框已看过叠图，边界仍是需再修的初稿；旧 39 张原有标签未逐框审定，部分有图库水印/网页截图，没有直接并入新的干净训练集。
4. **寻找外部真实照片。** 从核对过作者和许可的 Wikimedia Commons 作品页下载 13 张照片，逐张判为 10 张有火、3 张无可见火并写 YOLO 标签；与两代数据及旧 64 张共 9456 图做 SHA256/dHash 检查，精确重合 0。13 张留在训练之外作小规模外部诊断，不能代表独立厨房总体准确率。来源、许可证、哈希、框和检测结果在 `model_work/data/manual_kitchen_labels_v2_20260927/REAL_PHOTO_CANDIDATES.md`。
5. **只增补 25 张训练 v2。** 从 v1 最佳权重继续，训练集 2407 张，原 val/test 和 Commons 13 张不变；初始学习率 0.0003、最多 40 轮、patience 10。第 21 轮早停，最佳第 11 轮验证集 P=0.541、R=0.729、mAP50=0.515、mAP50-95=0.174，均低于 v1。权重在 `model_work/runs/fire-binary-codex-v2-manual25/weights/best.pt`，SHA256 `8cc003cb050065c9a47988959a236e27ae98d2d3d7e44468e4fcf01d69ec4d63`。
6. **分层对照，避免被高测试分数误导。** v2 在原 286 张 test 上 P=0.783、R=0.818、mAP50=0.806、mAP50-95=0.194；146 张厨房子集在 conf=0.25 时有火检出 71/71、定位 60/71、无火误报 0/75，较 v1 的定位有改善。但在独立来源 13 张上，v1/v2 在 conf=0.25 的有火检出分别为 7/10、6/10，定位分别为 5/10、4/10，无火误报均为 1/3；v2 **四张蓝焰全漏**，两版均把只有蒸汽的炒锅图误报为火。原 test 含同一厨房的不同拍摄事件，因此不能把 mAP50 0.806 写成跨厨房可靠性。逐图表及数值、训练脚本和所有路径集中在 `model_work/docs/MODEL_TEST_AND_TRAINING_REPORT_20260927.md`。
7. **收尾决定。** 电脑端暂保留单类 v1 作为候选，v2 封存为实验；**没有编译新 KModel、没有替换板端模型，也未实测 v1/v2 的 K230 推理**。下一步先补不同厨房/机位的锅下小蓝焰、蒸汽与反光无火难例，按厨房/事件隔离训练与评测，再决定是否重训。用户本轮要求写完文档即停，故这里不再启动新训练或部署。
