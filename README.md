# 端侧大模型推理系统

**在 ALINX AX7020（AMD Xilinx Zynq-7020）上完整运行 Qwen3.5-0.8B 大语言模型，并实现可交互的多轮对话。**

整块开发板只有 1 GB DDR3 和两颗 666 MHz 的 Cortex-A9。本作品通过块浮点量化、手写 NEON
算子、按带宽预算重排访存，把 0.8B 参数模型装进这块板子，并跑到 **1.491 token/s**——相比
纯 Python/numpy 基线加速 **17.7 倍**。

---

## 目录

- [核心成果](#核心成果)
- [硬件平台](#硬件平台)
- [系统架构](#系统架构)
- [量化方案：W8A16 块浮点](#量化方案w8a16-块浮点)
- [性能剖析](#性能剖析)
- [硬件边界实测](#硬件边界实测)
- [PL 加速器](#pl-加速器)
- [仓库结构](#仓库结构)
- [快速开始](#快速开始)
- [从零构建](#从零构建)
- [实测证据索引](#实测证据索引)
- [技术栈](#技术栈)
- [关于权重文件](#关于权重文件)

---

## 核心成果

| 指标 | 实测值 | 说明 |
|---|---|---|
| 贪心解码延迟 | **670.7 ms/token** | 12 次连续前向稳定复现 |
| 生成吞吐 | **1.491 token/s** | 同上 |
| 默认采样延迟 | 850.0 ms/token | top-k 40 / top-p 0.9 / T=0.7 |
| 相对基线加速 | **17.7×** | 纯 Python/numpy 为 11,889 ms/token |
| 量化后权重 | **643 MB** | 8.125 bit/元素，可完整驻留 1 GB 内存 |
| 每 token 访存 | 681.1 MB | 已逼近实测带宽墙的 93% |
| 模型规模 | 24 层 / 1024 hidden / 3584 FFN | 18 层 Gated DeltaNet + 6 层全注意力 |
| 词表 | 65,536 | 从 151,936 切分，含 ChatML 特殊符号 |

---

## 硬件平台

| 项目 | 规格 |
|---|---|
| 开发板 | ALINX AX7020 |
| SoC | XC7Z020-2CLG400I（Zynq-7000） |
| PS | 双核 ARM Cortex-A9 @ 666.67 MHz |
| PL 资源 | 53,200 LUT / 106,400 FF / 140 BRAM36 / 220 DSP48E1 |
| 内存 | 1 GB DDR3，32-bit @ 1066.667 MT/s（理论 4.267 GB/s） |
| 操作系统 | Debian 8 (jessie)，Linux 4.9.0-xilinx armv7l |
| 工具链 | gcc 4.9.2、Python 3.4.2、numpy 1.8.2 |
| 交互 | 串口 COM4，115200 8N1 |

---

## 系统架构

```
┌──────────────────────────────────────────────────────────┐
│  应用层   交互控制台 chat_v3.py                            │
│           多轮上下文 · /clear · /think 思考模式 · 采样/贪心  │
├──────────────────────────────────────────────────────────┤
│  引擎层   qwen_real_engine.py                             │
│           24 层前向 · 分页 KV Cache · RoPE · ChatML 模板    │
├──────────────────────────────────────────────────────────┤
│  算子层   NEON C 内核（libqwen_*.so）                      │
│           GEMV · 全注意力 · GDN 递归 · 采样 · 分词器        │
├──────────────────────────────────────────────────────────┤
│  量化层   BFP8 block-64 权重 · int16 分块激活 · int32 累加  │
├──────────────────────────────────────────────────────────┤
│  平台层   Zynq-7020 PS + PL · 1 GB DDR3 · PL 加速器 RTL    │
└──────────────────────────────────────────────────────────┘
```

数据流：权重以 `mmap` 映射进地址空间，首次访问由内核按页调入，之后常驻 page cache。
每个 token 只做一次完整遍历，所有矩阵向量乘共享同一份激活。

---

## 量化方案：W8A16 块浮点

模型 FP16 权重约 1.6 GB，装不进 1 GB 内存。方案是 **BFP8 分块量化**：

- 权重每 **64 个元素**为一组，共享 **1 个 int8 指数**，每个元素存 **1 个 int8 尾数**
  → 平均 **8.125 bit/元素**，643 MB
- 激活按 64 元素块量化为 **int16**，块内共享一个 float32 缩放因子
- 点积用 **int32 累加**（每块 64 个 int8×int16 乘积，最大 |积| 远小于 2³¹）
- 输出还原为 float32，KV Cache 保持 float32

指数以 2 的幂次还原：`scale = 2^(e - 134)`。这个"整数算完再一次性缩放"的结构，使得
GEMV 全程不碰浮点，是 NEON 内核能跑满带宽的前提。

量化导出与逐张量校验脚本见 [`05-主机侧源码/`](05-主机侧源码/)，320 个张量的逐层参数见
[`06-模型与权重/qwen35_weights_64k_manifest.json`](06-模型与权重/qwen35_weights_64k_manifest.json)。

---

## 性能剖析

单 token 668.3–670.7 ms 的构成（贪心路径）：

| 阶段 | 耗时 | 占比 |
|---|---|---|
| `bfp8_gemv`（24 层全部线性层） | 427.1 ms | 63.8% |
| `lm_head`（64K 词表投影） | 113.0 ms | 16.9% |
| 注意力投影 | 43.7 ms | 6.5% |
| GDN 递归状态更新 | 38.0 ms | 5.7% |
| 其他（归一化、RoPE、采样等） | 31.6 ms | 4.7% |
| **合计** | **~670 ms** | 100% |

优化过程（每一档都是实测）：

| 版本 | ms/token | tok/s | 关键动作 |
|---|---|---|---|
| 纯 Python/numpy 基线 | 11,889 | 0.084 | — |
| + 权重 BFP8 量化 | 3,180 | 0.314 | 访存降 2× |
| + NEON GEMV 内核 | 1,240 | 0.806 | 算力不再受限 |
| + 注意力 C 内核 | 817 | 1.224 | 消除 np.repeat 广播拷贝 |
| + GDN 递归转置 | 736 | 1.359 | 消除跨步访问 |
| + lm_head / 采样融合 | **670.7** | **1.491** | 最终版本 |

---

## 硬件边界实测

所有上限都是量出来的，不是估的：

| 边界 | 实测值 | 测量方式 |
|---|---|---|
| DDR3 顺序读墙 | **1270.8 MB/s** | 大块顺序读，多轮取最好值 |
| NEON 指令发射上限 | 1220 MB/s | 纯 NEON load 循环 |
| PL 侧 AXI 吞吐 | 1178 MB/s | AXI DMA 实测，达 128-bit@100 MHz 峰值的 **74%** |
| PL 读延迟 | 180.3 ns @100 MHz | 单次读往返 |
| 纯读理论下限 | 536 ms/token | 681.1 MB ÷ 1270.8 MB/s |

关键结论：**CPU 路径的 670.7 ms 已经是实测读墙的 93%**。继续提速不能再靠"少算"，
只能靠减少访存或做计算/访存重叠——这也是 PL 加速器的设计出发点。

> 注意：`/mnt/qwen/membw` 若未加 `-DPF` 编译，会因缺少预取而低报约 36%（807–818 MB/s）。
> 本仓库引用的 1270.8 MB/s 来自带预取的正确编译版本。

---

## PL 加速器

PL 侧完成了完整的 RTL 设计、独立 testbench 验证、综合实现与时序收敛：

| 模块 | 功能 | 规模 |
|---|---|---|
| `gemv_core.v` | BFP8 GEMV 加速核：16 路并行 DSP、11 级流水、64 位饱和累加、AXI-Stream | 676 行 |
| `gemv_core_pipe.v` | 流水线变体 | 434 行 |
| `gdn_core.v` | Gated DeltaNet 递归状态更新 | 818 行 |
| `gdn_core_ref.v` | GDN 行为级参考模型 | 545 行 |
| `conv1d_silu.v` | 因果卷积 + SiLU 融合 | 199 行 |
| `conv_core.v` | 因果卷积核 | — |
| `fwht64.v` | 64 点快速 Walsh-Hadamard 变换 | 163 行 |
| `bfp8_pack16.v` | BFP8 分组打包 | — |

- **时序**：Block Design 在 200 MHz 约束下 **WNS +0.018 ns，0 条失败路径**
- **资源**（单核 BD）：LUT 11,768/53,200 = 22.12%，FF 14,770 = 13.88%，
  DSP 20/220 = 9.09%，BRAM 2.5/140 = 1.79%
- **板上实际时钟**：100 MHz（由 `BOOT.BIN` 里的 `ps7_init` 设定，非 200 MHz）
- **testbench**：6 套独立验证，全部通过

`gemv_core` 的累加器设计值得单独说明：内部是带 40 位小数的 **signed 64 位饱和**累加器
（`acc = y·2⁴⁰`），输出取 `clamp(acc, ±2³¹<<20)[51:20]` 即 `floor(y·2²⁰)` 的 int32。
用 int32 累加会在 8 个 block 后溢出，而最短的一行也有 16 个 block——所以 64 位是必须的。

---

## 仓库结构

```
.
├── README.md                    本文件
├── 00-评委会说明.md              选题变更与赛道选择说明
├── START_DEMO.cmd               零安装演示入口（双击运行）
├── demo_console.ps1             串口控制台（Windows）
│
├── 01-系统概述/                  9 个文件 · 主报告与专题分析
│   ├── FINAL_REPORT.md          ★ 主报告（架构 / 量化 / 算子 / 性能 / 结论）
│   ├── ACCEPTANCE.md            验收摘要：功能、性能、硬件逐条判定
│   ├── HARDWARE_LIMITS.md       硬件上限分析：带宽、发射、仲裁、PL 通路
│   ├── CPU_PROFILE_RESULTS.md   逐 token 性能剖析与瓶颈定位
│   ├── FINAL_BW_ALIGNMENT.md    带宽标定：实测 vs 理论，多路交叉验证
│   ├── DEMO_GUIDE.md            演示指南
│   ├── ATTN_C_KERNEL.md         注意力 C 内核设计说明
│   └── GDN_REC_OPT_RESULTS.md   GDN 递归状态优化结果
│
├── 02-RTL与仿真/                 37 个文件 · Verilog 与验证环境
│   ├── gemv_core.v              ★ BFP8 GEMV 加速核（核心）
│   ├── gdn_core.v / _ref.v      GDN 递归核 + 参考模型
│   ├── conv1d_silu.v / conv_core.v / fwht64.v / bfp8_pack16.v
│   ├── tb_*.v                   6 套独立 testbench
│   ├── build_bd.tcl             Block Design 构建脚本
│   ├── ps7_params.tcl           PS7 参数
│   ├── run_*.bat / *.tcl        仿真、综合、OOC 运行脚本
│   ├── util_*.rpt               各算子资源报告
│   ├── REPORT_gemv_core.md      GEMV 核设计报告
│   └── CORE_SIM_RESULTS.md      核心仿真结果
│
├── 03-实测证据/                  12 个文件 · 原始测量记录
│   ├── DEMO_LIVE_RUN.md         真实对话运行记录
│   ├── repl_run_*.log           冷/热启动实际对话日志
│   ├── REBOOT_TEST.md           掉电重启自恢复验证
│   ├── REPL_COMMANDS_VERIFIED.md 交互命令逐条验证
│   ├── acceptance_log.txt       验收脚本输出
│   ├── sim_pass.txt             仿真通过记录
│   ├── plclk_measurement.txt    PL 时钟实测
│   ├── timing_bd_200mhz.rpt     ★ BD 时序报告（WNS +0.018 ns）
│   └── util_bd_200mhz.rpt       ★ BD 资源报告
│
├── 04-板载脚本与工具/             21 个文件 · 板载运行的全部代码
│   ├── real_chat_v3             ★ 启动入口
│   ├── chat_v3.py               ★ 交互控制台
│   ├── qwen_real_engine.py      ★ 推理引擎（24 层前向 / 分页 KV）
│   ├── qwen_tokenizer.py        分词器封装
│   ├── qwen_kernels_opt.c       ★ NEON 算子库源码
│   ├── qwen_attn.c              注意力 C 内核源码
│   ├── qwen_sampling.c          采样内核源码
│   ├── libqwen_kernels.so       算子库（编译产物）
│   ├── libqwen_attn.so          注意力库
│   ├── libqwen_sampling.so      采样库
│   ├── libtokenizer.so          分词器库
│   ├── accept.sh / accept_prompts.sh   验收脚本
│   ├── startup.sh / install_startup.sh 开机自启配置
│   └── *_test.py / *_verify.py  功能与正确性验证
│
├── 05-主机侧源码/                 8 个文件 · 主机侧工具链
│   ├── prune_vocab_64k.py       词表切分至 64K
│   ├── slice_embeddings_64k.py  embedding 切分
│   ├── export_qwen35_weights_bfp8_64k.py  ★ 权重 BFP8 量化导出
│   ├── bfp8_packer.py           BFP8 分组打包
│   ├── verify_exported_weights.py 逐张量浮点校验
│   ├── membw.c                  内存带宽标定工具
│   └── profile_kernels.py       算子级剖析
│
├── 06-模型与权重/                 10 个文件 · 模型元数据（权重见下节）
│   ├── config.json              模型配置
│   ├── chat_template.jinja      ChatML 模板
│   ├── model.safetensors.index.json
│   ├── qwen35_weights_64k_manifest.json  320 个张量逐层参数
│   ├── new_to_old_64k.json      词表 ID 映射
│   ├── tokenizer_64k.bin        ★ 64K 词表二进制
│   └── tokenizer.c / .h / tb_   分词器 C 实现与测试
│
└── 07-板子系统与比特流/           4 个文件 · 系统镜像
    ├── BOOT.BIN                 启动镜像（含 ps7_init）
    ├── image.ub                 内核 + 设备树
    └── gemv_sys.bit.bin         PL 比特流（Linux 加载格式）
```

---

## 快速开始

### 方式一：零安装演示（推荐）

板子插好 USB 串口线，双击根目录的 **`START_DEMO.cmd`** 即可。它会自动找到 COM 口、
连上板子、启动推理服务，然后在 `you>` 提示符下等你提问。

不需要安装任何串口工具、Python 或依赖。

### 方式二：手动运行

```bash
# 板子上（串口或 ssh 均可）
cd /mnt/qwen
./real_chat_v3
```

`real_chat_v3` 会自动挂载分区、设置环境变量并拉起 `chat_v3.py`。

### 交互命令

| 命令 | 作用 |
|---|---|
| 直接输入文字 | 对话 |
| `/clear` `/new` `/reset` | 清空上下文与 KV 缓存 |
| `/think on` / `/think off` | 开启 / 关闭思考模式 |
| `/think show` / `/think hide` | 显示 / 隐藏思考内容 |
| `/stats` | 查看轮次、KV 占用、思考状态 |
| `/help` / `/quit` | 帮助 / 退出 |

启动参数：`--greedy`（贪心，最快）、`--sample`（采样，默认）、`--max N`（生成长度）、
`--ctx 4096`（上下文长度）、`--no-think`（强制关闭思考）。

---

## 从零构建

### 1. 权重导出（主机侧）

```bash
# 依赖：python3 + numpy + safetensors
python3 prune_vocab_64k.py            # 词表切分，产出 64K 映射
python3 slice_embeddings_64k.py       # 切分 embedding
python3 export_qwen35_weights_bfp8_64k.py   # BFP8 量化导出
python3 verify_exported_weights.py    # 逐张量浮点校验
```

> 导出脚本内部路径是硬编码的，运行前需按本地目录修改文件头部的路径常量。

### 2. 算子库（板上）

```bash
gcc -std=c99 -O3 -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard -DUSE_NEON \
    -shared -fPIC -o libqwen_kernels.so qwen_kernels_opt.c -lm -fopenmp
gcc -std=c99 -O3 -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard -DUSE_NEON \
    -shared -fPIC -o libqwen_attn.so qwen_attn.c -lm -fopenmp
gcc -std=c99 -O3 -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard \
    -shared -fPIC -o libqwen_sampling.so qwen_sampling.c -lm -fopenmp
gcc -std=c99 -O3 -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard \
    -shared -fPIC -o libtokenizer.so tokenizer.c -lm
```

### 3. PL 比特流（Vivado）

```bash
cd 02-RTL与仿真
vivado -mode batch -source build_bd.tcl      # 综合 + 实现 + 生成比特流
```

生成 `gemv_sys.bit`，转成 Linux 加载格式：

```bash
echo "all" > /sys/class/fpga_manager/fpga0/flags   # 或直接：
cat gemv_sys.bit.bin > /dev/xdevcfg
```

### 4. 仿真验证

```bash
vivado -mode batch -source run_sim.tcl       # 6 套 testbench
vivado -mode batch -source check_all.tcl     # RTL lint
```

---

## 实测证据索引

| 想核查什么 | 看哪里 |
|---|---|
| 系统确实在板上跑通 | [`03-实测证据/DEMO_LIVE_RUN.md`](03-实测证据/DEMO_LIVE_RUN.md)、`repl_run_*.log` |
| 性能数字不是编的 | [`01-系统概述/CPU_PROFILE_RESULTS.md`](01-系统概述/CPU_PROFILE_RESULTS.md)、`03-实测证据/acceptance_log.txt` |
| 带宽墙是真的 | [`01-系统概述/FINAL_BW_ALIGNMENT.md`](01-系统概述/FINAL_BW_ALIGNMENT.md)、`05-主机侧源码/membw.c` |
| 时序收敛 | [`03-实测证据/timing_bd_200mhz.rpt`](03-实测证据/timing_bd_200mhz.rpt) |
| 资源占用 | [`03-实测证据/util_bd_200mhz.rpt`](03-实测证据/util_bd_200mhz.rpt) |
| RTL 正确性 | [`02-RTL与仿真/CORE_SIM_RESULTS.md`](02-RTL与仿真/CORE_SIM_RESULTS.md)、`tb_*.v` |
| 掉电重启能恢复 | [`03-实测证据/REBOOT_TEST.md`](03-实测证据/REBOOT_TEST.md) |
| 交互命令都可用 | [`03-实测证据/REPL_COMMANDS_VERIFIED.md`](03-实测证据/REPL_COMMANDS_VERIFIED.md) |

---

## 技术栈

`Qwen3.5-0.8B` · `BFP8 块浮点量化` · `ARM NEON` · `Cortex-A9` · `Zynq-7020` ·
`Vivado` · `Verilog RTL` · `AXI4-Stream` · `AXI DMA` · `Debian 8` · `Python 3.4.2`

---

## 关于权重文件

量化后的权重 `qwen35_weights_64k.bin`（643 MB）与 `embedding_64k.bin`（134 MB）超过
GitHub 仓库容量限制，**以 Release 附件形式单独提供**。

若需完整重建，`05-主机侧源码/` 里的导出脚本配合 `06-模型与权重/qwen35_weights_64k_manifest.json`
（320 个张量的逐层形状与参数）可以从原始 safetensors 完全复现。

---

## 许可与致谢

模型权重与结构版权归 Qwen 团队所有，本项目仅作学术与竞赛用途。
推理引擎、NEON 算子、RTL 与全部工具链为本作品自主实现。
