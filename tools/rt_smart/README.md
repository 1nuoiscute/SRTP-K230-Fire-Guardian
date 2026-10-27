# 大核 RT-Smart 程序构建工具（Zig + LLD）

在**没有 K230 SDK 官方交叉工具链**的 Windows 机器上，编译并链接大核 RT-Smart 可执行文件。

2026-10-26 之前这件事在本机做不到 —— 仓库里不含 RT-Smart 工具链，本地也没有任何编译器
能生成 `-mcmodel=medany` + 8 GB 基址的 riscv64 静态 ELF。

---

## 为什么不能直接用 zig cc

`zig cc` 能生成 riscv64 代码，但**链接阶段不行**：

| 尝试 | 结果 | 原因 |
|---|---|---|
| `zig cc -target riscv64-linux-musl` 直接编 | ❌ | Zig 注入自己的 medlow CRT 和 compiler_rt，`-nostartfiles` / `-rtlib` 被当成「未使用参数」忽略 → `R_RISCV_HI20 out of range` |
| 手写链接脚本放 0x10000（低地址，medlow 可用） | ❌ 链接成功但板子拒绝 | `[E/LWP] elf entry is invalidate, entry:0x10000!` — RT-Smart 的 LWP 加载器要求入口在用户空间（0x200000000） |
| **SDK musl sysroot + `zig ld.lld`（本方案）** | ✅ | LLD 是独立链接器，完全受链接脚本控制 |

关键点：**`lui` 的 20 位立即数编不出 0x200000000**（需要 22 位），所以放在 8 GB 基址的
目标文件必须用 `-mcmodel=medany` 编译。SDK 的 musl sysroot 正是 medany 编译的，直接拿来用。

---

## 需要的输入

1. **Zig 0.13.0**（自带 LLVM 18 和 LLD 18.1.6）
2. **K230 SDK 的 riscv64 musl sysroot**
   来源：SDK 的 `riscv64-linux-musleabi_for_x86_64-pc-linux-gnu.tar.bz2`
   里面有 `riscv64-unknown-linux-musl/{include,lib}`，`libc.a` / `crt1.o` / `crti.o` /
   `crtn.o` / `libgcc.a` 都是 medany 的
3. **SDK 的 MPP 预编译库**（`userapps/lib/lib*.a`，共 41 个）和头文件（`include/`、`userapps/api/`）
4. **本目录的 `link_rt.lds`** 和 `fix_crt_section_flags.py` 生成的 `crt1_fixed.o`

---

## 两步构建

```
第一步  zig cc -c   每个 .c → .o
        -target riscv64-linux-musl
        -mcpu=generic_rv64+a+c+d+f+m      ← 加原子指令，否则 musl 的 lr.w.aqrl 报错
        -mcmodel=medany
        --sysroot=<SDK musl sysroot>

第二步  zig ld.lld  链接
        -static --no-dynamic-linker
        -T link_rt.lds
        <crt1_fixed.o> <crti.o> <所有 .o>
        --start-group -l<MPP 库...> -lc -lgcc --end-group -lpthread
        <crtn.o>
```

`build_rt_two_stage.py` 封装了这两步。用法：

```sh
python tools/rt_smart/build_rt_two_stage.py <输出.elf> <源文件1.c> [源文件2.c ...]
```

路径配置在脚本顶部的常量里（`ZIG`、`SYSROOT`、`SDK_INCLUDE_DIRS`、`MPP_LIB_DIR`），
换机器时改那几行即可。

---

## 必须处理的四个坑

### ① `crt1.o` 的 `.start` 段 sh_flags = 0

SDK 的 `crt1.o` 里 `.start` 段的段头标志是 `0`（既不是 ALLOC 也不是 EXEC），
LLD 因此不给它分配地址，导致 `PCREL` 重定位越界。

修法：把该段的 `sh_flags` 从 `0x0` 改成 `0x6`（`SHF_ALLOC | SHF_EXECINSTR`）。

```sh
python tools/rt_smart/fix_crt_section_flags.py <原 crt1.o> <输出 crt1_fixed.o>
```

（`zig objcopy` 在本机会因 `UnableToSpawnSelf` 失败，所以直接改 ELF 段头字节。）

### ② 链接脚本里的 `*(*.plt)` 通配

SDK 官方链接脚本用了 `.plt : { *(.plt) }` 这种写法，通配符会把 `.rela.plt` 也吸进去，
LLD 报 `section type mismatch`。

修法：`link_rt.lds` 里显式列出 `.plt / .plt.got / .plt.sec / .iplt`。

### ③ `_DYNAMIC` 未定义

RT-Smart 的 LWP 加载器拒绝带 `PT_DYNAMIC` 的 ELF：

```
[E/LWP] ERROR: Can't open elf file /lib/ld.so!
```

修法两件事：
- 链接时加 `-static --no-dynamic-linker`
- `link_rt.lds` 里放一个 `.dynamic_stub`（两个 `LONG(0)`），并
  `PROVIDE(_DYNAMIC = __dynamic_stub)`，让 musl 读到 `DT_NULL` 立即停止

### ④ 链接脚本不能有 BOM

PowerShell 的 `Set-Content -Encoding UTF8` 会写 BOM，ld 报：

```
unknown directive: \ufeff
```

修法：用 Python 以 `newline='\n'` 和 `utf-8`（无 BOM）写入。

---

## 验证产物

```sh
# 入口地址必须是 0x200000000
readelf -h <输出.elf> | grep Entry

# 不能有 DYNAMIC / INTERP 段
readelf -d <输出.elf> | grep -E 'DYNAMIC|NEEDED'   # 应该无输出
```

**2026-10-26 实测产物：**

| 程序 | 大小 | 板端结果 |
|---|---|---|
| `hello_rt.elf` | 397,000 B | 打印 `HELLO FROM ZIG-BUILT RT-SMART ELF` |
| `data_page_gas.elf` | 749,048 B | 连续刷新 `DATA: seq=... mq2=... flame=...` |

---

## 已知限制

- **只能静态链接**（RT-Smart 没有动态加载器）
- **产物比官方 gcc 编的大**（Zig 静态链完整 musl，约 700 KB vs 原 20~330 KB），
  内存充裕时无影响；需要时加 `-Os` 和 strip
- **`vg_lite` / `vg_lite_util` 两个库要排除**（`pngtest.c` 里有重复的 `main`）
- 环境变量 `ZIG_GLOBAL_CACHE_DIR` / `ZIG_LOCAL_CACHE_DIR` 需要指向可写目录，
  否则 Zig 在受限沙箱里无法创建默认缓存
