> ⚠️ **性能数字已更新（2026-11-04 09:58）** —— 本文写于更早的阶段，正文保留为当时的历史记录。
>
> | 阶段 | ms/token | tok/s |
> |---|---|---|
> | 本文记录的 M2a 第一轮 / 第二轮 | 918 / 803.0 | 1.089 / 1.2454 |
> | + P1 注意力 C+NEON 内核 | 704.9 | 1.419 |
> | + `attn_post_f32`（numpy 胶水搬进 C） | 689.8 | 1.450 |
> | + GDN 递推状态 4×4 分块转置 | 673.0 | 1.486 |
> | **+ `lm_head` 预取对齐（当前）** | **670.7** | **1.491** |
>
> **另：本文里出现的「589 ms / 1.70 tok/s」和「618 ms / 1.62 tok/s」两个下限都已作废。**
> 每 token 实际流量是 **681.1 MB**（不是 777.6 MB），纯读墙是 **1270.8 MB/s**，
> 各算子按自身上限加总的复合下限是 **592 ms / 1.69 tok/s**；
> 「每阶段都取历史最好值」的天花板是 **653.3 ms / 1.53 tok/s**。
> 推导见 `HARDWARE_LIMITS.md` §1/§2 与 `MEASURED_CORRECTIONS.md` §7。
>
> **最新权威数字见 `HARDWARE_LIMITS.md` §2。**
# Qwen3.5-0.8B 在 ALINX AX7020 上的推理加速：最终报告

**日期**：2026-11-03 · **平台**：ALINX AX7020（Xilinx Zynq-7020，XC7Z020-2CLG400I），1 GB DDR3
**模型**：Qwen3.5-0.8B，24 层 = 18 层 Gated DeltaNet + 6 层全注意力
**权重**：BFP8 块浮点（每 64 元素 1 个共享指数 + int8 尾数），613.6 MiB

**一句话结论**：PS 侧两轮软件优化把吞吐从 0.0841 tok/s 提到 **1.2454 tok/s（14.8 倍，板载实测，
两条独立路径确认）**，其中 **77% 的收益来自 Python/ctypes/numpy 热路径，而不是内核**；
PL 侧自建 GEMV 核功能完全正确（与 NEON 内核逐位相同）、流带宽也更高（1178 vs 784 MB/s），
**但端到端比 CPU 慢 1.49 倍**（M2b 的对照是第二轮优化前的 930 ms 引擎）。目标 2.0 tok/s 未达成，
瓶颈不在算力，在 29.6 MB/token 的非一致回读。

## 1. 课题与目标

课题：在 Zynq-7020 开发板上加速 Qwen3.5-0.8B 的单 token 生成，并用板载实测数据回答
"这块板子到底能跑多快、卡在哪里"。所有数字均为板上实测或由实测值直接折算，没有跑分软件、
没有仿真外推（唯一例外在 §5.3 中明确标注）。

### 1.1 硬件与模型

| 项 | 值 |
|---|---|
| 器件 | XC7Z020-2CLG400I（2× Cortex-A9 + Artix-7 级 PL） |
| PS | 2× Cortex-A9 @ 666.67 MHz，NEON 2×128-bit |
| 内存 | 1 GB DDR3，32-bit，1066.667 MT/s，理论峰值 **4.27 GB/s**；可用 989 MB，无 swap |
| PL 时钟 | **166.67 MHz**，由 PS 写 `FPGA0_CLK_CTRL`（`0xF8000170 = 0x00100600`）设定 |
| PL 资源 | 53 200 LUT / 106 400 FF / 140 BRAM / 220 DSP48E1 / 4 个 AXI-HP 口 |
| 系统 | Debian 8 (jessie)，Linux 4.9.0-xilinx，Python 3.4.2，gcc 4.9.2 |
| 模型 | 24 层 = 18 层 GDN + 6 层全注意力；hidden 1024，FFN 3584；16 head × 128 |
| GDN / 词表 | 因果卷积核宽 4，状态矩阵 64×64 / head；词表裁到 65536 |
| 权重 | BFP8（block 64），613.6 MiB（643 362 048 B），320 个张量 |
| embedding / lm_head | 128.0 MiB（134 217 728 B，bf16） |
| **每 token 访存** | **777.6 MB**（613.6 + 128.0） |

### 1.2 里程碑

| 里程碑 | 目标 | 结果 | 状态 |
|---|---|---|---|
| M1 基线 | 测出来 | 11 889 ms/token，0.0841 tok/s | 达成 |
| M2a PS 侧内核 | ≥1.2 tok/s | **1.2454 tok/s（14.8×）** | **达成**（1.20 线以上 4%，两轮优化） |
| M2b PL 接管 GEMV | ≥2.0 tok/s | 0.483 tok/s（端到端）/ 0.724（每前向） | **未达，且比 CPU 慢** |
| MTP 投机解码 | 实测接受率 | 板上无草稿模型，改为给盈亏平衡阈值 | 完成（否定结论） |
| 板载对话验收 | 端到端可对话 | 5 个 prompt 连贯续写，输出与基线逐字一致 | 部分完成（无多轮脚本） |

## 2. 总体架构

### 2.1 PS/PL 分工

分界线不是"哪部分更复杂"，而是**哪部分被带宽卡住**。PS 承担 Python 控制流、rmsnorm、注意力、
GDN 递推、lm_head argmax（128 MB embedding）以及 BFP8 的指数缩放与跨块累加——合计约 325 ms/token，
都是"每 token 只做一次"的小张量工作；PL 承担 int8×int16 的 GEMV，它是唯一被 613.6 MiB 权重流支配的算子，
在 A9 上只能跑到 1075 MB/s，而 fabric 在同样时钟下能到 1178 MB/s。

数据通路是 AXI DMA（`0x40400000`）：权重从 DDR 经 `S_AXI_HP` 送进 PL，块和再经 S2MM 回读。
`S_AXI_HP` 是 **64-bit、AXI3、非相干**的，这一点决定了 M2b 的结局（§4、§5）。

```
PS (2x Cortex-A9 @ 666.67 MHz)      PL (FCLK0 = 166.67 MHz)
 control flow, rmsnorm,             gemv_core.v
 attention, GDN (NEON),    <DMA>    16 MAC/cycle, 6-stage
 lm_head argmax (128 MB),           64 elems -> 1 int32 block sum
 BFP8 exponent scale + add          no AXI-Lite (header in stream)
        |  mmap 613.6 MiB weights -> DDR3, 4.27 GB/s theoretical
```

### 2.2 为什么两个自定义算子走手写 RTL 而不是 HLS

队友的方案要求两个 PL 算子（GDN 状态核、Conv+FWHT+BFP8 打包核），原计划用 Vitis HLS 实现。
实测本机 HLS **无法指向这块器件**：

```
ERROR: [HLS 200-1023] Part 'xc7z020clg400-2' is not installed.
```

而 Vivado 2025.2 能综合 `xc7z020clg400-2`（我们一直在用），`data/parts/installed_devices.txt`
里也有 `xc7z020`——所以是 **Vitis HLS 这个安装缺 7 系列器件数据**，不是板子或工程的问题。
可选的只有两条：补装器件支持，或者改手写 RTL。选了后者，因为 `gemv_core` 已经证明
"手写 RTL + `create_bd_cell -type module -reference`"这条路能走通、能上板，而 HLS 路线要先解决安装问题。
另外队友那两个 HLS IP 的产物目录（`build_dual_core_system.tcl` 里写死的
`D:/fpga2026/work/hls/.../impl/ip`）本来就是空占位符，从未生成过 IP。
代价是：手写 RTL 的数值正确性必须自己用自检 TB 建立——这就是 §6 的内容。

## 3. M2a：PS 侧优化（14.8×，本项目的主力成果）

### 3.1 总账：两轮优化，920.7 → 803.0 ms

| 指标 | 基线（纯 Python/numpy） | 第一轮 | 第二轮（final） |
|---|---|---|---|
| 每前向 | 11 889 ms | 920.7 ms（917.8 / 924.1 / 920.7） | **803.0 ms（803.0 / 805.3 / 800.6）** |
| 吞吐 | 0.0841 tok/s | 1.0861 tok/s | **1.2454 tok/s** |
| 相对基线 | 1× | 12.9× | **14.8×** |
| lm_head（128 MiB，bf16） | 908 ms @ 141 MB/s | 114.4 ms | 114.2 ms @ ~1 121 MiB/s（未动） |
| `forward_layer(0)`（线性注意力层） | — | 34.1 ms | 29.0 ms |

表中均为 3 次运行的中位数。**独立复核**：另一个脚本 `verify_m2a.py`（8 token，单独进程）测得
802.2 / 799.9 / 801.8 / 801.5 / 801.3 / 802.3 / 802.7 / 803.1 ms → **avg 801.9 ms、1.2471 tok/s**。
两条独立路径相差 0.14%，所以 803 ms 不是某一次运行的运气。目标 ≥1.2 tok/s（≤833 ms）**达成且有余量**
（1.2454 vs 1.20）。

基准提醒：M2b（§4）的 CPU 对照是**第二轮优化前**的引擎（930 ms / 1.075 tok/s），与本节 803 ms 不是同一版。

### 3.2 117.7 ms 从哪来：77% 不在内核里

这是本轮最反直觉的发现——一个"内核优化"项目，收益的大头却不在内核。按 ctypes 函数对象插桩的逐项分解
（ms/token，来源 `M2A_PUSH_RESULTS.md` §2）：

| 组件 | 调用/次 | 第一轮 | 第二轮 | Δ |
|---|---|---|---|---|
| `bfp8_gemv` | 150 | 571.6 | 562.0 | −9.6 |
| `lm_head_argmax` | 1 | 114.4 | 114.2 | −0.2 |
| `gdn_recurrent_step` | 18 | 77.5 | 66.9 | −10.6 |
| `causal_conv1d_update_silu` | 18 | 17.6 | 9.1 | −8.5 |
| `swiglu` | 24 | 12.0 | 4.7 | −7.3 |
| `rmsnorm_gated` / `rmsnorm` / `qk_l2norm` | 85 | 10.3 | 9.8 | −0.5 |
| `gdn_gates`（新增） | 18 | — | 5.7 | +5.7 |
| **python + ctypes + numpy** | — | **124.5** | **34.2** | **−90.3** |
| **合计（带插桩）** | | **927.9** | **806.6** | **−121.3** |

插桩版比 bench 分别多 7.2 / 3.6 ms（wrapper 开销），所以构成看 121.3、总账看 117.7。内核与框架侧合计只省了
**31.0 ms**，**90.3 ms（117.7 的 77%）在 Python/ctypes/numpy 热路径上**：

- `ndarray.ctypes.data` 在这块板子上要 **34.3 us/次**（numpy 1.8.2 每次访问都重建 `__array_interface__`），
  而每 token 的循环调了它 **1127 次** = 38.7 ms/token。现在所有指针在 `_build_fast_path()` 里解析一次、
  缓存成普通 int，热循环里一次 `ctypes.data` 都不剩。
- 持久激活缓冲（`np.empty`，只分配一次）替代每 token 约 80 次分配；去掉 3 处 q/k/v 拷贝——它们只是
  conv 输出缓冲的字节偏移（`mixed+0` / `mixed+8192` / `mixed+16384`），而 `qk_l2norm` 本来就原地归一化；
  三个门标量合成一个 C 内核 `gdn_gates`（两次 16×1024 matvec + `sigmoid` + `softplus` 一次调用，
  18 层共 5.7 ms，替掉约 15 ms 的 numpy 工作和每层 8 次 ctypes）。

内核侧的三笔是 −10.6 / −8.5 / −7.3：`gdn_recurrent_step` 用精确恒等式
`out = a·(qᵀS_old) + (q·k)·delta` 把状态扫掠从 4 趟（6 MB/token）压到 2 趟（3 MB），数学上等价、只改浮点
舍入顺序；`causal_conv1d_update_silu` 把 `conv_state` 从 `[channels][3]` 改成 `[3][channels]`（每个抽头连续），
用 `vld4q_f32` 解交织权重、自写 `exp_neon`；`swiglu` 同样换成 NEON `exp`/`div`，4 浮点/迭代。

### 3.3 三个被测量推翻的判断

1. **LM head 不是 63 ms 的机会。** lm_head 表是 **bf16、65536 × 1024 = 128 MiB**，不是 BFP8 / 68 MB。
   114.2 ms 对应约 **1 121 MiB/s，即读墙的 ~89%**；它的地板约 102–108 ms，所以最多只有 **6–12 ms** 可挖，
   不是 63 ms。**保持原样未动。**
2. **`gdn_recurrent_step` 从来不是每 token 144 次 ctypes 调用。** 它是 18 次（每层一次），16 个 head 的
   循环在 C 里。真实成本是每层对 1 MB 状态扫掠 4 趟——这才是 3.2 里那条恒等式起作用的原因。
3. **`bfp8_gemv` 是发射受限（issue-bound），不是带宽受限。** 它跑 855 MiB/s，是这块板子实测只读天花板
   1 185–1 195 MiB/s 的 **72%**。所以剩下的 GEMV 空间在"每条权重的 NEON 指令数"，不在内存——这也解释了
   为什么只有 int8 激活那条路能推动它（§3.5），而那条路必须拿精度换。

### 3.4 正确性证据

- `generate('The capital of France is', 8)` → ids `[11698, 13, 198, 759, 6486, 313, 9298, 368]`，文本与改动前
  逐字相同（`ref8_match = True`、`txt8_match = True`）。
- 真正的证据：**三个不同 prompt**（`The capital of France is` / `def fibonacci(n):` / `Once upon a time`）
  分别在两个独立进程里跑旧引擎+旧内核与新引擎+新内核，输出 diff → **IDS_IDENTICAL**（空 diff，各 8 token）。
  这条才排除了对单一 prompt 的过拟合。
- 激活漂移：12 次前向（4 prefill + 8 生成）的最终 RMSNorm 输出对比改动前，max abs diff **4.23e-03**、
  absmax 40.11 → **1.05e-04 相对（0.01%）**，均值 6.0e-04，逐 token 峰值在 1.8e-03..4.2e-03 之间**不增长**。
  参照尺度：BFP8 权重量化本身约 **4.5e-03 相对**，所以这点漂移比模型自带的量化噪声低约 **43 倍**
  （4.5e-03 ÷ 1.05e-04）。

### 3.5 被否决的方案（"什么没帮上"）

**int8 激活路径（`-DX8`）确实更快：862.3 ms / 1.1597 tok/s，`forward_layer(0)` 31.6 ms，8-token 与 32-token
的 id 检查都过。但它仍然被否决**——激活漂移 **maxabsdiff 6.98e-01、相对 1.74e-02**，比 int16 路径差 **165 倍**，
这是截止日期压力下正确的取舍：**快 7% 换掉一个数量级的数值裕度，不值。**

另外三项测了没用：预取距离调优（d=0 678 / d=96 855.5 / d=192 833 / d=512 764 MiB/s，`PFD=96` 本来就最优）；
4 行分块 gemv（849.7 vs 855.5 MiB/s，无收益）；加载位流以避开 SIGBUS（`/dev/xdevcfg` 返回 0，但
`0x43C00000` 的 mmap 仍 SIGBUS，PL 不在本轮范围）。

### 3.6 顺带测出的带宽墙（M2a 最重要的副产品）

`membw.c`（纯 NEON 读流，613 MB）：单线程 **426 MB/s**；双线程 **818 MB/s**；双线程 + 软件预取 **1 259 MB/s**；
128 MB 双线程 813 MB/s。**PS 侧读墙是 1 259 MB/s，不是原方案假设的 2.75 GB/s**（2.75 GB/s 是 PL 经 AXI-HP
读 DDR 的数，A9 双核到不了）。777.6 MB ÷ 1 259 MB/s = **618 ms ⇒ 纯 PS 理论上限 1.62 tok/s**。
当前 803 ms / 1.245 tok/s 已用掉这堵墙的 **73%**，余量只剩 **1.36 倍**（803 / 589）。

## 4. M2b：PL 侧 GEMV 接管（功能全对，端到端更慢）

### 4.1 建了什么

`gemv_core.v`（**736 行 / 32 876 B**；2026-11-04 打了 burst 截断累加器泄漏补丁 +13 行，补丁前的 723 行 / 31 945 B 那一版保留为 `gemv_core_asrun_20261104.v`，因为板上比特流是它综合的；227 行那份是早期的 `gemv_core_v1.v`，别拿错）是一对 AXI4-Stream 进出的块点积引擎，**没有 AXI-Lite**：权重 int8（BFP8 尾数）
16 个/beat，x 为 int16（每 64 元素块独立缩放）8 个/beat，16 MAC/cycle、**12 级流水（S1…S12）**，每 64 元素块输出一个 int32 原始点积和。

三个关键决定：**(a)** 输出"块和"而不是"整行和"——BFP8 每块有自己的指数，整行不能共用一个比例，
所以核按块输出、PS 侧每行做 K/64 次 `fmaf`（K=1024 时 16 次），表达式与顺序和 `qwen_kernels.c`
完全一致，这是逐位相同的原因。**(b)** 去掉 AXI-Lite——AXI4-Lite 从设备必须响应读，第一版把 `rvalid`
恒接 0，互联一旦发起读就整板挂死；改成配置用流的第一个 beat 传（header：`x_beats` /
`beats_per_row-1`），BD 里因此不需要任何互联到核，PS 只要编三次 DMA 传输。**(c)** 输出级用深度 8 的
环形 FIFO——只有一个寄存器时 `m_axis_tready` 短暂为低会让下一个块和覆盖旧和，实测 K=1024/M=8
偶发错 18/128，换 FIFO 后归零。

### 4.2 实测结果（同一块板、同一轮启动、同一份权重）

| 路径 | GEMV ms/token | 权重 MB/s | 其它 ms/token | 合计 ms/token | tok/s（每前向） |
|---|---|---|---|---|---|
| CPU C/NEON（`--cpu`） | 605 | 784 | 325 | **930** | **1.075** |
| PL fabric GEMV | 402 DMA + 572 回读 = 974 | 1178（DMA 侧） | 407 | **1381** | **0.724** |

`generate('The capital of France is', max_new_tokens=8)`（12 次前向 = 4 prefill + 8 decode）：
CPU 11.16 s → **0.717** 新 token/s；fabric 16.57 s → **0.483**。
**1.49×** 就是 1381 / 930（新 token 速率口径 0.717 / 0.483 = 1.48，同一件事）。

### 4.3 正确性：完全通过

| 检查 | 结果 |
|---|---|
| `layers.0.mlp.down_proj`（M=1024, K=3584）fabric vs NEON | **max\|abs\| = 0.000e+00，逐位相同** |
| 同上 vs numpy 浮点参考 | 5.047e-05（max rel 9.490e-03），仅 int16 x 量化差 |
| 端到端 8 token | id 完全相同 `[11698, 13, 198, 759, 6486, 313, 9298, 368]`，文本相同 |
| 真实权重 vs Python BFP8 参考 | **0/4096 块和错误** |
| 合成权重块和（全 0x01，x 全 0x1111，应为常数 279616） | **57 344/57 344 全对** |
| 仿真（`tb_gemv_core`，K=128/M=3 与 K=1024/M=8） | 2/2 PASS，0 mismatch |

也就是说：**fabric 没有精度问题，它只是更慢**。这一条必须先立住，否则后面的负面结果会被误读成"没做对"。

### 4.4 负面结果：赢了带宽，输了端到端

fabric 在流式传输上赢了 **203 ms**（1178 vs 784 MB/s），但核内吐出的是 `M*K/16` 字节的 int32 块和
= **29.6 MB/token**，而 `S_AXI_HP` 与 CPU cache **不一致**，这个回读只能走非缓存 `/dev/mem` 视图，
实测约 **52 MB/s**，花掉 **572 ms**——是省下时间的 **2.8 倍**，单位字节比 DMA 慢 **22 倍**。

| 回读方式 | 速度 | 正确性 |
|---|---|---|
| 非缓存 `/dev/mem` 视图（默认） | ~52 MB/s，572 ms/token | 始终正确 |
| arena 的普通缓存映射 | 快约 100 倍 | **错** |
| 缓存映射 + 8 槽环形缓冲 | 大块可用 | 碎片化后失败（`max\|abs\| = 1.76`，输出乱码） |

缓存读错的原因值得记下来：ARM 的 `cacheflush` 系统调用（`__ARM_NR_cacheflush`，0x0f0002）只维护
**L1**，Zynq 的 L2（PL310，512 KB）对 PL 写过的页仍保留旧行。环形缓冲靠 L2 自然换出，大块时碰巧能过，
arena 碎片化后块变小就失效——正确性检查当场抓到。该路径只保留在 `PL_READBACK=cached` 下用于测上限，
**不能用来出结果**。冷启动（一次性）：位流 + 时钟约 1 s，arena 分配 + pagemap + mlock 4.7–5.3 s，
stage 474 MB 用 26.5 s（18.8 MB/s），cacheflush 1.4 s，**首个 token 前约 33 s**。

## 5. 为什么 PL 没有赢，以及要怎样才能赢

### 5.1 数字账

权重流省下 474 MB × (1/784 − 1/1178) ≈ **203 ms**；块和回读花掉 29.6 MB ÷ 52 MB/s ≈ **572 ms**；
净亏 **+369 ms**，对应 930 → 1381 ms。结论很硬：**在这块板子上，fabric 的 1.5 倍带宽优势付不起
任何 CPU 侧的收尾工作。** 1.5 倍不是随便说的：784 与 1178 是同一口径（MiB/s）下的数；换成 MB/s
是 821 vs 1236，比值 1.43（见附录第 3 条）。

> **⚠️ 2026-11-04 单位复核：上面这条账的口径是混的，别照抄。**
> 1178 这个数是 **MB/s**（1178 MB/s ÷ 1270.8 MB/s = 92.5%，见 \M2A_PUSH_RESULTS.md\），
> 不是 MiB/s；784 才是 MiB/s（= 822 MB/s）。混着用会把 PL 侧算快。
> 按纯 MB 口径重算：497 MB × (1/822 − 1/1178) = **183 ms**，不是 203 ms；
> 于是净亏是 **572 − 183 = 389 ms**，不是 369 ms。
> **结论不变** —— PL 单路径照样输，而且端到端 1381 ms 是实测值，不是这条估算推出来的；
> 这条估算只能当数量级，不能当证据。

### 5.2 四个选项（按"能不能真的赢"排序，全部是 RTL 改动，本轮均未实施）

| # | 改动 | 预期效果 | 判定 |
|---|---|---|---|
| 1 | **核内直接输出成品 `y[m]` float32**：在 fabric 里把 BFP8 指数和 x scale 乘掉 | 回读 29.6 MB → 约 1.9 MB，572 ms → 约 40 ms，合计约 850 ms/token（1.1 tok/s） | **只是追平，不是超越** |
| 2 | **按权重 beat 计数收流，而不是靠 `s_axis_tlast`** | 权重不必物理连续，arena/staging 整套问题（26.5 s 冷启动、碎片化、卡死风险）消失 | 必要但不充分 |
| 3 | **让回读一致**：换 AXI **ACP** 口，或 CPU 侧用 PL310 按物理地址 clean+invalidate（`0xF8F027B0`） | 保留块和接口也能快 | ARM 的 `cacheflush` 不够，必须走 L2 |
| 4 | **以上都做完，1178 vs 784 这点带宽差才开始有意义** | 1.5× 才可能变成端到端收益 | 最后一步 |

### 5.3 一个必须撤回的中间结论

开发中期曾记录"PL 数据通路外推 **2.004 tok/s**，M2b 目标达成"（`HANDOVER.md` §18）。那是**只算权重流**
的外推：613.6 MiB ÷ 1 229 MB/s = 499 ms，不含回读、不含 PS 侧收尾。接入引擎后端到端实测 **0.483 tok/s**，
两者差 4 倍。本报告不采用 2.004 作为结果，只把它当作"数据通路本身没问题"的证据——这一点它确实证明了（§4.3）。

另一个上限测量同样说明问题：把回读当作**免费**的同一份代码测得 **1058 ms/token = 0.945 tok/s**，
仍低于 CPU 的 1.075。**光修好回读一致性还不够**，必须让核直接产出 `y`。

## 6. 两个自定义算子

两个算子采用手写 RTL 实现，以 module reference 方式集成进 BD，与 `gemv_core` 走同一条路。
来源：`CORE_SIM_RESULTS.md`（含 2026-11-03 17:55 补记）、`synth/timing_full.rpt`、`util_full.rpt`。

### 6.1 交付了什么，验证到哪一步

| 算子 | 文件 | 行数（交付包/当前） | 功能 | 精化 | 自检 TB | WNS @166.67 MHz |
|---|---|---|---|---|---|---|
| GDN 状态递推 | `gdn_core.v` | 583 / 708 | 16 MAC lane，8 head × 64×64 状态，decode/reset/dump | 0 error | **PASS（7 项）** | **−16.636 ns**（完整 BD 内测得） |
| 因果卷积 + SiLU | `conv1d_silu.v` | 214 / 219 | 4 抽头因果卷积 + Q2.14 SiLU，6144 通道 | 0 error | **PASS（3 项）** | **−22.290 ns** |
| 64 点 FWHT | `fwht64.v` | 179 / 179 | 自然序 Hadamard，(1/8)Σ(−1)^popcount | 0 error | **PASS（8 项）** | **−0.710 ns** |
| BFP8 打包 | `bfp8_pack16.v` | 147 / 149 | 16 样本共享指数 + 尾数 | 0 error | **PASS（9 项）** | **−27.603 ns** |
| Conv 顶层 | `conv_core.v` | 146 / 146 | 卷积 + FWHT + 打包串联 | 0 error | **PASS（4 项）** | **−27.603 ns** |

五套回归一起跑：`pass=5 fail=0`。所有模块 WHS 为正（+0.036 ~ +0.213 ns），hold 不是问题。

**功能对不等于能跑。** 只有 `fwht64` 接近收敛（差 0.71 ns，最容易修）；`bfp8_pack16` 与 `conv_core`
差 27.6 ns、`conv1d_silu` 差 22.3 ns、`gdn_core` 差 16.6 ns——这些不是"改 bug"能救的，需要插流水。
所以这五个模块**不能当作"可运行的算子"交付**：它们没有进任何位流，从未上板。

### 6.2 仿真到底验证了什么

| 模块 | 证据 |
|---|---|
| `fwht64` | 冲激 → 64 个输出相等；常数 100 → 只有 `out[0] = 800`；伪随机向量与 64×64 参考 64/64 一致。实测顺序是**自然 Hadamard** `z[i] = (1/8)·Σ_j (−1)^popcount(i&j)·a[j]`（冲激@1 输出 `+125 −125 +125 …` 排除了 sequency 序）；首入→首出 257 拍 |
| `gdn_core` | `mode=1` 后 `mode=2` dump **4096/4096 全零**；连续两个 decode token 的 64 个输出值与 4096 字状态 dump 都与整数参考**逐位相同**，且 token 2 确实从非零状态开始 |
| `bfp8_pack16` | 16 × −32768 → exp 149、mant0 = −64（修复前 exp 150、尾数整体小 2^(16−E)）；伪随机块 9/9 全对 |
| `conv1d_silu` | 1536 输出拍；6144 通道 SiLU 全部落在 `x*sigmoid(x)` 的 **64 LSB** 内（修复前 \|x\| ≥ 1.75 区间误差达 **3574 LSB**，399 通道超标） |
| `conv_core` | 1536 输入拍 → 384/384 BFP8 输出拍，`tlast` 在第 383 拍，输出总线无 X |

### 6.3 首轮回归是 pass=2 fail=3，三个真 bug

| # | 位置 | 根因 | 修复 | 复验 |
|---|---|---|---|---|
| 1 | `bfp8_pack16.v:47` `abs17` | `17'd0 - $unsigned(x)` 先零扩展再取负，负样本回绕成 `65536+\|x\|`（`abs17(-32768) = 98304`），`e_pos` 恒为 16、exp 恒 150 | 显式符号扩展：`17'd0 - {1'b1, x[15:0]}` | 16×−32768 → exp 149 / mant −64 |
| 2 | `conv1d_silu.v` `sig_base` | 只有 8 个断点（seg 0..7）却按 `seg+1` 索引，`seg=7` 落到 `default`，导致 `b1 < b0`；`interp[23:12]` 是无符号切片，插值反号 | 索引加宽到 4 位，补第 9 个断点 `sigmoid(2.0) = 28862` | 6144 通道全部 ≤ 64 LSB |
| 3 | `conv1d_silu.v:32` | `m_axis_tlast` 声明为 `input` 却在第 212 行被 `assign`；xsim 容忍，Vivado 综合不会 | 改为 `output wire` | 精化与仿真均通过 |

另有一个在早期 `gdn_core.v` 修订里抓到的 bug：相位 4 没有重载 `k_cur`，仍持有相位 2 结束时的
`k_buf[63]`，于是每一行都用 `k[63]` 更新（与 stale-k 模型 4096/4096 吻合）。当前修订在 `ST_P4`
装载 `k_cur` 后通过。这也解释了为什么"仿真通过"必须写明是哪一版——**交付包里的 583 行 `gdn_core.v`
仍是修复前版本**，`work/rtl` 的 708 行修订才是通过的那版（附录第 7 条）。

### 6.4 一个需要澄清的注释错误

`gdn_core.v` 头注释写 `e : Q3.13`，实际是 **Q2.14**，所以 `(beta*k*e) >> 31` 是注释值的 2 倍。
状态 dump 与 `>>31` 模型 **4096/4096** 吻合、与 `>>32` 模型 **0/4096** 吻合——**实现是自洽的，
是注释标错**，不是数学错。但这仍应与 HLS 参考实现核对一次才能定论，否则下一个接手的人会以为
状态更新的增益差一倍。

### 6.5 双核位流：能构建，时序不收敛

| 指标 | 单核（gemv） | 双核（gemv + gdn） |
|---|---|---|
| WNS（约束 200 MHz，5 ns） | **−0.606 ns** | **−16.636 ns** |
| setup 失败端点 | 50 / 69 469 | **25 043 / 116 583** |
| hold | +0.036 ns，0 失败 | +0.036 ns，0 失败 |
| LUT / FF | 11 768（22.12%）/ 14 770（13.88%） | 28 308（53.21%）/ 31 412（29.52%） |
| BRAM / DSP | 2.5（1.79%）/ 20（9.09%） | 21（15.00%）/ 116（52.73%） |

双核最差路径在 `gdn_core` 内部：`sub_reg[2]_rep/C → o_mac_reg[0][1][20]/D`，数据路径 **21.320 ns**
（逻辑 13.735 / 布线 7.585），**17 级逻辑**（CARRY4×8 + DSP48E1×4 + LUT2×3 + LUT3 + LUT6），
即 4 个 DSP48E1 串成一条组合链（`alpha*S` → `k*e` → `beta*(k*e)` → 饱和 → `q*S_new` → 累加）。
修复方案已定位（在 `p4_snew` 与 `p4_prod` 之间插一级流水），**未实施**。单核位流在 200 MHz 下也只差
一点（−0.606 ns，最差路径在 AXI DMA 的 S2MM realigner 里，不在我们自己的核里），所以板上最终选
166.67 MHz 运行——这是"时序不收敛但仍可用"的工程折中，不是达标。

## 7. 投机解码（MTP）

### 7.1 板上没有草稿模型，接受率测不出来

| 检查 | 结果 |
|---|---|
| manifest 张量数 / 匹配 `mtp`·`draft`·`eagle` 的张量 | 320 / **0**（全部是 `model.language_model.*`） |
| 板上 `config.json` | **不存在**（设计文档里 `mtp_num_hidden_layers: 1` 无法验证） |
| 引擎里的 `mtp` / `draft` / `spec` / `verify` / `accept` | 0 命中 |
| `libqwen_kernels.so` 导出的 GEMV 入口 | 只有 `bfp8_gemv`（`00001c20 T bfp8_gemv`），无 batch 入口 |

所以本文不编造接受率，改为**实测验证成本 + 给盈亏平衡阈值**（来源 `MTP_RESULTS.md`）。

### 7.2 实测：验证 K 个位置并不便宜

`gemv_batch_bench.c`，7 个真实形状按个数加权到全部 150 个张量：

| 项 | B=2 | B=3 | B=4 |
|---|---|---|---|
| `T(B)/T(1)` | 1.716 | 2.453 | 3.231 |
| 每个额外验证位置的成本（一整趟 = 1.0） | **0.72** | **0.73** | **0.74** |

**这是决定性的一条**：这个内核是**算力受限**而非带宽受限，权重没有在 K 个位置间复用，
K 个 token 的验证约等于 K 趟前向。（该 bench 内核比引擎内核慢 2.5 倍，所以只用它的**缩放比例**，
不用绝对时间。）

### 7.3 盈亏平衡接受率

`T_K = 605 ms × ratio(K) + OTH_K`；边界 A：非 GEMV 工作重复 K 次（`OTH_K = K × 325 ms`），
边界 B：非 GEMV 工作完全摊薄（`OTH_K = 325 ms`）。token 数用标准的 `1 + α + α² + ... + α^K`
（比 `1 + αK` 严格）：

| K | T_K 边界 A | α* 边界 A | T_K 边界 B | α* 边界 B |
|---|---|---|---|---|
| 2 | 1688 ms | **0.53** | 1363 ms | **0.35** |
| 3 | 2459 ms | 0.73 | 1809 ms | 0.53 |
| 4 | 3255 ms | 0.82 | 2280 ms | 0.63 |

即：除非 MTP 头单 token 命中率高于 **53%**（K=2 最坏情况；若其余流水线完全摊薄则高于 35%），
投机在本板子上就是净亏。K=2 是唯一合理选择，K 越大阈值越差。α=0.30 时 K=2 只有 0.766×/0.948×
（**比不投机还差**，折算 0.55/0.68 tok/s）；α=0.75 时 1.274×/1.578×，折算 **0.91–1.13 tok/s**。

### 7.4 独立上限

实测 PS 读墙 1 259 MB/s，每 token 需 777.6 MB ⇒ 618 ms/token = **1.62 tok/s**。这是任何纯 CPU 侧实现
（投机或不投机）的绝对天花板。**投机到不了 2 tok/s**：它是对 0.717 乘 1.3–1.6 倍，而验证成本本身
压在同一堵墙上。要突破，必须让权重**一趟读出、同时算多个位置**——那是内核/RTL 工程，不是解码器改动。

## 8. 性能上限分析

一个 token 必须搬 **777.6 MB**（613.6 MiB 权重 + 128 MiB lm_head），这个数字在 BFP8 格式与词表裁剪
确定后就固定了，所有路径共用同一个分子：

| 口径 | 带宽 | 777.6 MB/token | 折算 tok/s | 现状 |
|---|---|---|---|---|
| DDR3 物理峰值（1066.667 MT/s × 4 B） | 4.27 GB/s | 174 ms | 5.75 | 当前用掉 22% |
| 单 HP 口 @166.67 MHz（64 bit × FCLK） | 1 333 MB/s | 556 ms（折算） | 1.80（折算） | DMA 实测 1 229–1 314（92–99%） |
| **PS 双核读墙（预取）** | **1 259 MB/s** | **618 ms** | **1.62** | M2a 已用掉 73% |
| M2a 实测有效（803 ms） | 924 MB/s（均摊） | 803 ms | **1.2454** | 距 618 ms 的地板只剩 **1.36×** |

第二轮把 920.7 ms 压到 803 ms 之后，离 618 ms 这堵墙只剩 **1.36 倍**（第一轮是 1.56 倍），

100 MHz 下同一个 HP 口实测只有 **772.7 MB/s**（理论上限 800 MB/s 的 96%），所以 166.67 MHz 的选择是对的
——但它也只把天花板从 1.04 抬到 1.80。

为什么 777.6 MB/token 卡死一切：**(a)** 每 token 必须把全部权重读一遍，batch 实测表明权重在多个位置间
不复用（§7.2），"多算几个 token 摊薄一次读取"不成立；**(b)** 分子不能降——权重已是 BFP8，再量化与
"不再量化"的团队约束冲突且无实现，词表已裁到 65536；**(c)** 分母已经到顶——A9 双核 + NEON + 预取
= 1 259 MB/s（DDR3 峰值的 29.5%），PL 单 HP 口 = 1 333 MB/s 上限、实测 92–99%；**(d)** 两个分母不能相加，
PS 与 PL 抢同一条 DDR3，"PS 跑 lm_head + PL 跑 GEMV 并行"只对 lm_head 那 128 MB 有效。

2 tok/s 需要 371 ms/token、即 **2.0 GB/s 的有效带宽**：超过 PS 单侧读墙 59%，接近单 HP 口的 1.5 倍。
单 HP 口在 166.67 MHz 下最多给 1.80 tok/s，200 MHz 下 2.5 tok/s 但时序不收敛。**想过 2.0 必须同时启用
第二个 HP 口**，那要改 PS7 配置、重新生成 `ps7_init`、重建 BOOT.BIN。

## 9. 结论

### 9.1 做到了什么

1. **14.8× 的端到端加速（两轮）**，全部板载实测、输出与基线逐字一致：0.0841 → **1.2454 tok/s**
   （803 ms/token），终值由两条独立路径确认（803.0 / 801.9 ms）。**77% 的收益不在内核里**，而在
   Python/ctypes/numpy 热路径（§3.2），内核侧只贡献 31.0 ms。
2. **一个功能完全正确的 PL GEMV 核**：与 NEON 内核逐位相同（max|abs| = 0.000e+00）、端到端 token id
   一致、57 344 个合成块和全对；单核位流已在板上跑通，DMA 侧带宽 1 178 MB/s（单次峰值 1 313.8 MB/s）。
3. **五个手写 Verilog 算子全部通过自检仿真**（首轮 pass=2 fail=3，三个真 bug 修复后 pass=5 fail=0），
   其中 `gdn_core` 的两 token 状态递推与整数参考逐位相同。
4. **三条可复核的硬结论 + 对原方案的修正**：PS 读墙 1 259 MB/s ⇒ 纯 PS 上限 1.62 tok/s；单 HP 口
   @166.67 MHz 上限 1 333 MB/s ⇒ 1.80 tok/s；MTP 每多一个位置要 0.72–0.74 趟、不划算。
   同时把原方案的每 token 访存 492.6 → 777.6 MB、PS 带宽 2.75 GB/s → 1.259 GB/s、TPS 上限 5.58 → 1.70、
   MTP 收益 12.73 TPS → 实测每 token 只省 18% 逐条改正（`MEASURED_CORRECTIONS.md`）。

### 9.2 没做到什么

| 缺口 | 现状 | 卡在哪 |
|---|---|---|
| **2.0 tok/s** | 未达成，最好 1.2454（纯 PS），PL 端到端 0.483 | 纯 PS 上限 1.62，已用掉 73% |
| PL 端到端提速 | 比 CPU 慢 1.49× | 29.6 MB/token 非一致回读，52 MB/s |
| 两个自定义算子上板 | 仿真 5/5 通过，但只有 `fwht64` 接近收敛（−0.710 ns） | 四个模块需插流水（−16.6 ~ −27.6 ns）；双核位流时序不收敛 |
| MTP 实测接受率 | 测不了（0/320 张量，无 config.json） | 权重里没有 MTP 头，引擎没有草稿路径 |
| 多轮对话验收 | 5 个 prompt 单轮连贯 | 没有多轮脚本 |

### 9.3 限制因素，一句话

**限制因素不是算力，是"每 token 777.6 MB 的强制访存"对"1 259 MB/s 的 PS 读墙 / 1 333 MB/s 的单 HP 口"
之比。** 这块板子搬这个模型，物理天花板 5.75 tok/s，纯 PS 现实天花板 1.62 tok/s；我们拿到了 1.2454（73%），
而 PL 侧唯一能赢的姿势是"权重流更快 + 结果不出 fabric"，本轮只完成了前半句。**2.0 tok/s 未达成，
原因是数据通路只建到单 HP 口 + 块和回读，而不是算法或模型选型的问题。**

## 10. 可复现性

### 10.1 主机侧

```powershell
# single-core bitstream / dual-core bitstream / RTL elaboration / RTL self-check sim
cd E:\ds_workplace\scratch\qwenwork\rtl
& 'D:\AMD\2025.2\2025.2\Vivado\bin\vivado.bat' -mode batch -source build_bd.tcl
& 'D:\AMD\2025.2\2025.2\Vivado\bin\vivado.bat' -mode batch -source build_bd_full.tcl
& 'D:\AMD\2025.2\2025.2\Vivado\bin\vivado.bat' -mode batch -source check_all.tcl
& 'D:\AMD\2025.2\2025.2\Vivado\bin\vivado.bat' -mode batch -source tb\run_sim.tcl
```

`(2)` 会破坏 Vivado 批处理，必须走无括号 junction `E:\ds_workplace\scratch\qwenwork`；
且要重定向 `HOME` / `USERPROFILE` / `APPDATA` 到工作区，否则工具会往 C 盘写。

### 10.2 板上侧

```sh
cd /mnt/qwen
dmesg -n 1                                    # silence i2c log spam, else serial stream corrupts
cat gemv_sys.bit.bin > /dev/xdevcfg           # 0.27 s, must be redone after every reboot
python3 pl_clk.py 0x00100600                  # FCLK0 = 166.67 MHz, not persistent
gcc -O3 -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard -shared -fPIC \
    -o libpl_apply.so pl_apply.c -lm
python3 -u pl_run.py                          # fabric GEMV
python3 -u pl_run.py --cpu                    # control: engine built-in kernels
# M2a kernel (needs -std=c99: gcc 4.9 defaults to gnu89 and rejects in-loop for-int decls)
gcc -std=c99 -O3 -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard -shared -fPIC \
    -DPFD=96 -DPFL=0 -o libqwen_kernels.so qwen_kernels.c -lm -fopenmp
python3 -u verify_m2a.py                      # independent re-check: avg ms and tok/s, 8 tokens
# MTP batch bench (needs -std=gnu99; gcc 4.9.2 defaults to gnu89 and rejects for-loop decls)
gcc -O3 -std=gnu99 -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard -fopenmp \
    -o gemv_batch_bench gemv_batch_bench.c -lm
./gemv_batch_bench qwen35_weights_64k.bin 6144 1024
python3 mtp_break_even.py
```

M2a 内核的开关是 `-DPFD=96 -DPFL=0`（预取距离 96 B、局部性 0；实测 d=96 最优，见 §3.5），
且**必须带 `-std=c99`**——gcc 4.9 默认 gnu89，会拒绝循环内的 `for (int i = ...)`。
原版备份：第一轮留 `.orig`，第二轮留 `.bak`（md5 校验过的逐字节原始版本）。
回滚：`cp qwen_real_engine.py.bak qwen_real_engine.py && cp libqwen_kernels.so.bak libqwen_kernels.so`。

### 10.3 板级坑（继任者必读）

| 现象 | 原因 | 处置 |
|---|---|---|
| MM2S 挂死后所有传输超时，重载位流也救不回 | `DMACR = 0x00011006`，软复位位（bit 2）永远读回 1 | 只能 `sync; reboot` |
| 每次传输多花约 1 秒 | `DMACR` 软复位是 **bit 2**，不是 bit 1（bit 1 保留、恒读 1） | 用 `1 << 2`，轮询上限 20×1 ms |
| 简单模式 MM2S 卡在 16 KB | `C_SG_LENGTH_WIDTH` 默认 14，即 2^14 | 设成 26 |
| 完成判断误判 | Idle 位在写完 LENGTH 后仍为 1 | 只看 IOC（DMASR bit 12） |
| 偶发随机错（每次个数不同） | DMA 绕过 cache，CPU 缓冲区是普通匿名 mmap | DMA 缓冲区改用 `/dev/mem` 非缓存视图 |
| `msync` 无效 | 匿名 MAP_SHARED 背后是 tmpfs，内核跳过 cache 维护 | 别指望 `msync`，用非缓存窗口 |
| 吞吐从 1 178 掉到 364 MB/s | 物理碎片随运行时间累积（150 → 4882 个 chunk） | 开机后尽早分配；`drop_caches`；看 chunk 数而不是只看 tok/s |
| 权重被悄悄破坏 | arena 缓存映射零填充留下的脏行会盖掉非缓存写入 | 填充后做一次 `cacheflush(arena)`（1.4 s） |
| `pl_run.py` 无声消失，`$?` = 135 | 访问 `0x43C00000` 触发 SIGBUS；即使 `cat gemv_sys.bit.bin > /dev/xdevcfg` 返回 0 也仍然 SIGBUS | 构造引擎前把 `FastBoardHw` 换成 `CpuOnlyHw` 桩（`QWEN_CPU_ONLY=1`） |
| 位流被换掉 / 加载长度不对 | `/mnt/qwen/run`、`mount_and_run.sh` 会加载 `system_dual.bit.bin`；`.bit.bin` 比真位流多 4 字节尾巴 | 不运行它们；正确长度 4 045 564 = `0x3DBAFC`，Linux 下用 `/dev/xdevcfg` |
| 200 MHz 不收敛 | 违例路径在 AXI DMA 的 S2MM realigner（5.606 ns，最高 178.4 MHz） | 用 166.67 MHz（1000/6） |

## 附录：源文件之间的数字冲突

写报告时遇到的不一致，按"哪个更可信"处理，不静默取一个：

| # | 冲突 | 处理 |
|---|---|---|
| 1 | 第一轮 935 ms 剖析的逐项相加为 958 ms（605+133+117+74+17+12） | 该剖析已作废，被第二轮插桩分解（§3.2）取代；605/325 只作为 M2b 的对照口径（§4.2）保留 |
| 2 | 单核位流记为"0 failing"，而 `timing_bd_166mhz.rpt` 是 WNS −0.606 ns、50 个 setup 失败端点 | "0 failing" 指 hold（+0.036 ns，0 失败）；setup 确有 50 个失败端点（§6.5） |
| 3 | 784 MB/s 与 821 MB/s 指同一条 CPU 路径 | 784 是 MiB/s、821 是 MB/s（474 MiB = 497 MB）；1.5× 是 MiB 口径，MB 口径为 1.43× |
| 4 | PL 峰值写成 1 318 MB/s | 无法溯源；可查的是 1 313.8（单次传输）与 1 229.4（warm 32 切片），另 `pl_sweep2` 区间为 1 140–1 324。报告用可溯源的两个值 |
| 5 | "2.004 tok/s，M2b 目标达成" | 纯权重流外推，不含回读；端到端实测 0.483，已撤回（§5.3） |
| 6 | README 只写五个算子"精化 0 错误"，未提首轮回归 | 首轮回归实为 `pass=2 fail=3`，三个 bug 修复后 `pass=5 fail=0`；以 `CORE_SIM_RESULTS.md` 补记为准（§6.1/§6.3） |
| 7 | 交付包 `02-RTL与仿真/` 的 5 个 `.v` 曾是修复前版本（`gdn_core.v` 583 行无 `k_cur` 装载、`bfp8_pack16.v` 的 `abs17` 未修、`conv1d_silu.v` 的 `tlast` 仍是 `input`），`work/rtl` 是修复版 | **已解决，无需再动**：交付包现在就是修复后版本 —— `gdn_core.v` **865 行 / 38 382 B**（含 `k_cur`）、`bfp8_pack16.v` 149 行、`conv1d_silu.v` 219 行且 `:32` 是 `output wire m_axis_tlast`。583 行那份是 `gdn_core_ref.v`（参考模型，故意保留）。**接收方不需要重新拷贝** |
| 8 | 预取距离"96 B 最优"vs 表格标注"256 B" | 内核按 `PFD=96` 编译；1 259 MB/s 那次测量的标注为 256 B。两者都记，不合并 |
| 9 | MTP "每 token 只省 18%" vs "21–26%"；`M2A_RESULTS.md` §3 标题写"七处改动"而表内实列 9 项 | 只采用实测比值 `T(B)/T(1)`；第一轮的九项改动清单已被第二轮插桩分解（§3.2）取代，不逐条重列 |
| 10 | lm_head 116 ms @1 100 MB/s vs 115 ms @1 113 MB/s | 同一改动的两次采样，差 1%；报告用 M2A 表内的 116 / 1 100 |
| 11 | 板上 `dequant_en=1` 四次运行结果不一致（corr 0.744 / 0.708 / 0.655 / 0.678） | **根因已定位并修复（仅仿真验证）**：`ST_HDR` 不清 `acc`/`row_acc_64`，行中间被截断的 burst 把部分和留给下一个 burst 的第 0 行。补丁 +13 行在 `gemv_core.v`，`dequant_en=0` 路径逐位不变。**未综合、未重出比特流**，板上仍是 `gemv_core_asrun_20261104.v` 那一版。详见 `GEMV_DEQUANT_RACE_FIX.md` |

> **单位更正（2026-11-04，二次修正）** —— 每 token 实际流量 **681.1 MB**（旧文写的 777.6 MB 是把整个权重文件当成了每 token 必读，错的），纯读墙 **1270.8 MB/s**（旧文 1259 MB/s 是 1 次取样）。因此复合下界是 **592 ms = 1.69 tok/s**，「每阶段取历史最佳」的天花板是 **653.3 ms = 1.53 tok/s**；本文中出现的 589 ms / 1.70 tok/s 与 618 ms / 1.62 tok/s **均作废**。推导见 `HARDWARE_LIMITS.md` §1/§2 与 `MEASURED_CORRECTIONS.md` §7，逐条勘误见 `DOC_CONSISTENCY.md`。
