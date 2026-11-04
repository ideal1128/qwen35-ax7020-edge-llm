# 最终带宽对齐报告（FINAL_BW_ALIGNMENT）

日期：2026-11-04
板卡：ALINX AX7020 / XC7Z020，2× Cortex-A9 @ 667 MHz 固定，armv7l Debian 8，gcc 4.9.2，1 GB DDR3，PL310 L2 512 KB
目标：`≤ 640 ms/token`（≥ 1.56 tok/s），ids 位级不变

---

## 0. 结论（TL;DR）

**640 ms 不可达。** 本轮实测 **672.5 ms/token（1.487 tok/s）**，基线 673.0 ms，差 0.5 ms。

- 唯一采纳的改动：`lm_head_argmax` 的预取距离 `PFD 256 → 48`（bf16 元素）。
  同一块板、同一次上电、背靠背 A/B 实测 **116.272 ms → 112.987~113.405 ms（−2.5% ~ −2.8%）**，
  ids 位级一致（`[48536, 6093, 37982, 35338, 40288, 36630, 37474, 10942]`）。
- 端到端只快了 0.5 ms（0.07%），因为 lm_head 只占一个 token 的 17%，
  而端到端本身的运行间波动就有 ±0.5%。
- 640 ms 需要 **1064 MB/s** 的平均有效带宽；实测 **1013 MB/s**（纯读墙 1270.8 MB/s 的 79.7%）。
- 把**每个阶段都换成它实测过的最好值**加总，也只能到 **653.3 ms**（1042 MB/s）——仍差 13.3 ms。
- 剩下那 13.3 ms 只能靠给 `bfp8_gemv` 再砍指令，而它的指令数已经是 int8×int16 的下限；
  再往下必须改成 int8×int8，那就要把激活量化从 15 bit 降到 7 bit = **改量化格式/精度，明确禁止**。

**顺带纠正一个流传在文档里的错误结论**：`bfp8_gemv` 从来不是 1459 MB/s，真实值是 **1120~1148 MB/s**。
详见第 2 节——这个数字是"字节归属错误"，把整个 643 MB 权重文件都算进了 QP_GEMV，实际每 token 只读 460 MB。

---

## 1. 实测 DDR3 读带宽（`scratch/lmh_bench.c` Part A，best of 5）

| 配置 | 缓冲区 | 线程 | 预取 | MB/s |
|---|---|---|---|---|
| 顺序单线程 | 64 MB | 1 | PLD +128 B | 940.7 |
| 顺序双线程 | 64 MB | 2 | PLD +128 B | 1260.7 |
| **顺序双线程** | **128 MB** | **2** | **PLD +256 B** | **1270.8 ← 采用为"墙"** |
| 双流（2 行 2048 B 交替） | 128 MB | 2 | PLD +128 B | 1264.2 |

结论：这块板 2 线程的顺序读上限就是 **1270 MB/s 左右**（单线程只有 941，即 74%），
双流和单流几乎没差别（−0.5%），说明限制在 DDR3 控制器/DDR 本身，不在访问模式。
1 GB/s 级别的 kernel 全都已经贴着这堵墙的 80~94%。

---

## 2. 重要更正：`bfp8_gemv` 的真实带宽是 1120 MB/s，不是 1459 MB/s

`HARDWARE_LIMITS.md` 里"bfp8_gemv = 1459 MB/s"是把 **643,362,048 B 的整个权重文件**都当成
QP_GEMV 的流量算出来的。实际不是：

- 24 层里有 6 层是全注意力层，它们的 4 个投影（q/k/v/o）是在 **Python 侧**通过 ctypes 调的，
  不计入 QP_GEMV；
- 权重文件里还有 **134,217,728 B 的 embed_tokens** 表（lm_head 用），那是另一个阶段的流量。

用真实 `libqwen_kernels.so` 直接测（`scratch/lmh_bench3.c`，链接 `.so` 本身，不是复刻）：

| 形状 | 字节 | ms | MB/s |
|---|---|---|---|
| M=131072 K=1024 | 136.315 MB | 121.658 | 1120 |
| M=4096 K=1024 | 4.260 MB | 3.804 | 1120 |
| M=1024 K=2048 | 2.130 MB | 1.896 | 1124 |
| M=512 K=1024 | 0.532 MB | 0.406 | 1311（<512 KB，吃到 L2） |
| M=3584 K=1024 | 3.727 MB | 3.248~3.261 | 1143~1148 |

我另外用 `scratch/lmh_bench2.c` 手写的同构复刻（相同宏、相同 `_Pragma`、相同 PLD）
给出 1116/1114/1106/1288/1127，与真实 `.so` 相差 0.5% 以内——说明复刻是忠实的，
后面所有变体对比都可信。

---

## 3. 每 token 的精确字节数（从 320 张量 manifest 求和，不是估算）

manifest 总字节 643,362,048，320 个张量，`file_offset + byte_size` 最大值正好等于文件长度（无空洞）。

| 用途 | 字节 | MB | 说明 |
|---|---|---|---|
| `bfp8_gemv`（126 次调用） | 460,062,720 | 460.06 | MLP 24×3 = 268,369,920 + GDN 18×3 = 191,692,800 |
| 注意力 4 个投影（24 次调用，Python 侧） | 44,728,320 | 44.73 | 6 层 × 7,455,360 |
| lm_head 表 | 134,217,728 | 134.22 | 65536×1024 bf16 |
| GDN 递推状态 | 37,748,736 | 37.75 | 18 层 ×（1,048,576 读 + 1,048,576 写） |
| conv1d / in_proj_a,b / norm | 4,349,184 | 4.35 | 18 层 × 229,376 + 各类 norm |
| **合计** | **681,106,688** | **681.1** | |

672.503 ms / 681.1 MB = **1013 MB/s** 平均有效带宽。
640 ms 需要 681.1 / 0.640 = **1064 MB/s**（墙的 83.7%）。

---

## 4. 各阶段 before / after

`QWEN_PROFILE=1 OMP_NUM_THREADS=2 ./real_chat_v3 --cps 20320,22909 --greedy --max 16`
（27 个 token 的累计均值）

| 阶段 | before ms | after ms | MB | after MB/s |
|---|---|---|---|---|
| `bfp8_gemv` | 441.0 | 443.3 | 460.06 | 1038 |
| `lm_head_argmax` | 117.7 | **115.9** | 134.22 | 1158 |
| `py_full_attn` | 51.6 | 50.6 | 44.73+ | 1024（含 4 投影 + attn_post + Python glue） |
| `gdn_recurrent_step` | 37.8 | 38.0 | 37.75 | 994 |
| `conv1d_silu` | 13.4（合并） | 8.30 | | |
| `gdn_gates` | 同上 | 5.02 | | |
| `rmsnorm_gated` | | 5.07 | | |
| `swiglu` | | 3.94 | | |
| `rmsnorm`/`add_residual`/`qk_l2norm`/`embed` | ~10.9（合并） | 1.88 | | |
| **总计** | **673.0** | **672.5** | 681.1 | **1013** |

（`bfp8_gemv` 从 441.0 变 443.3 是运行间噪声，同一份 kernel 没动；两次 profiled 运行
之间 QP_GEMV 的波动范围是 441.0~443.3。）

端到端三种口径（同一份 `.so`，三次运行）：
- profiled 均值：**672.503 ms/token**
- `[prefill]` 均值：676.3 / 677.6 / 682.7 ms/tok（含首 token 的权重 staging，偏高）
- `[gen] first`：673.2 / 671.6 / 672.9 ms

---

## 5. 试过但没有采纳的改动（全部是实测，不是估计）

### 5.1 `bfp8_gemv`——按真实 126 次调用序列回放（`scratch/lmh_bench5.c`）

这是最贴近现场的测法：按 18 层 GDN × 6 + 6 层全注意力 × 3 的**真实调用顺序和形状**
在真实权重文件上跑一遍，总流量正好 460.06 MB。

| 变体 | ms | MB/s | vs 当前 |
|---|---|---|---|
| **2 行/趟, PLD +96 B（当前配置）** | **420.171** | **1095** | — |
| 2 行/趟, PLD +128 B | 421.049 | 1093 | −0.2% |
| 2 行/趟, PLD +96 B + 指数表预取 | 418.279 | 1100 | +0.5%（噪声） |
| 2 行/趟, PLD +48 B | 430.988 | 1067 | −2.5% |
| 4 行/趟, PLD +96 B | 528.925 | 870 | **−21%** |
| 4 行/趟, PLD +48 B | 513.810 | 895 | −18% |

4 行/趟明显更差（16 个 NEON 寄存器装不下 8 个累加器 + 4 个宽化中间量，溢出到栈）。
指数表预取只有 0.5%，在噪声内，不值得为一个改动去动 kernel。
**当前配置已经是这个 kernel 的最优。**

> **口径说明：本文有两个"`bfp8_gemv` 最好值"，不要当成矛盾。**
> 上表的 **420.171 ms / 1095 MB/s** 与第 7 节采用的 **427.1 ms / 1077 MB/s** 是
> **两次不同的回放实验**（不同 harness、不同 best-of 次数），都是在同一台板子上
> 按 126 次真实调用序列、460.06 MB 实测出来的。第 7 节刻意取**较保守**的 427.1 ms，
> 所以 **653.3 ms 是一个偏保守的天花板**；若改用上表的 420.171 ms，
> 复合最好值会降到 **646.4 ms（1.547 tok/s）**。
> 两个数都远低于 640 ms 的目标，第 7 节的结论不受影响。

### 5.2 `lm_head_argmax`（`scratch/lmh_bench.c` Part B、`lmh_bench2.c` Part E、`lmh_bench5.c`）

| 变体 | ms | MB/s | id |
|---|---|---|---|
| 预取 256 元素 = 512 B（旧值） | 116.272~117.967 | 1138~1154 | 4036 |
| 预取 64 元素 = 128 B | 113.897 | 1178 | 4036 |
| **预取 48 元素 = 96 B（采纳）** | **112.987~113.282** | **1185~1188** | 4036 |
| 预取 32 元素 = 64 B | 114.201 | 1175 | 4036 |
| 4 行/趟（v4） | 114.6~115.4 | 1152~1171 | 4036 |
| 每 32 B 才预取一次（v2w） | 121.9~127.0 | 1056~1102 | 4036 |

原因：内层循环每迭代每行只前进 16 B（8 个 bf16），512 B 的预取提前量早就冲过
正在用的那两行、把有用的行挤出去了；96 B 刚好落在循环接下来真正要消费的窗口里。
**"4 行/趟"和"稀疏预取"都是负收益**，与"指令数不是瓶颈"的判断一致。

### 5.3 OpenMP fork/join 开销——**净开销为零**（`scratch/lmh_bench6.c`）

按真实的 126 个 `parallel for` 区域（真实 trip count、真实 `if (M>=64)`）空跑，best of 200：

| | ms | µs/区域 |
|---|---|---|
| 126 个 parallel 区域 | 1.465 | 11.63 |
| 同样 126 个区域串行 | 2.133 | 16.92 |
| **净开销** | **−0.667** | **−5.30** |

并行版本连 126 次 fork/join 一起算都比串行**快** 0.67 ms，说明 GOMP 线程池在这个
调用密度下是完全热的。**"持久线程池""提高 `if (M>=64)` 阈值"这两条路可以直接排除**，
最多值 0.7 ms，方向还是反的。

### 5.4 gemv 的运行间波动（同一 bench，8 次连跑）

427.590 / 427.139 / 427.065 / 427.956 / 427.718 / 428.582 / 428.985 ms
（第 0 次 485.869 ms 是 page cache 冷启动，应剔除）
均值 435.1，最好 427.1，**离散度 +1.9%**（剔除冷启动后 <0.5%）。

这条很重要：profiled 的 QP_GEMV 均值 443.3 ms 与独立 bench 的 427~435 ms 之间的 8~16 ms，
是引擎里其它 kernel 在 gemv 调用之间污染 L2/TLB 造成的，**不是 kernel 本身的开销，
在 `qwen_kernels_opt.c` 里改不掉**。

---

## 6. 为什么 gemv 不能再快：指令发射瓶颈的算术

`bfp8_gemv` 内层每 16 B（一行）需要：

| 指令 | 条数 | NEON 流水线周期（A9，128-bit op = 2） |
|---|---|---|
| `vld1q_s8`（权重） | 1 | 2 |
| `vmovl_s8` ×2（int8→int16） | 2 | 4 |
| `vmlal_s16` ×4 | 4 | 8 |
| `vld1q_s16`（x，两行共享） | 1 | 2 |
| 块尾 float 尾处理（摊到每 16 B） | ~1.5 | ~1.5 |
| **合计** | | **≈17.5** |

→ 单核 16 B / 17.5 周期 = 0.914 B/cycle，2 核 × 667 MHz = **1220 MB/s 上限**。
实测 1095（独立 bench 最好）/ 1038（profiled 均值）= **上限的 88% / 85%**。

对照 `lm_head_argmax`：每 32 B 只要 12 条指令 = 24 周期 = 1.33 B/cycle → 上限 1778 MB/s，
实测 1188 = 上限的 67%，**它是被内存墙卡住的（1270.8 的 93.5%），不是被指令卡住的**。

这个不对称正好解释了为什么 lm_head 还能靠调预取拿到 2.5%，
而 gemv 无论怎么调预取距离、行数、指数表预取都拿不到东西。

---

## 7. 640 ms 不可达的算术证明

1. 每 token 流量 **681.1 MB**（第 3 节，manifest 精确求和）。
2. 640 ms ⇒ 需要 **1064 MB/s** 平均带宽 = 纯读墙（1270.8）的 **83.7%**。
   当前 672.5 ms = 1013 MB/s = 79.7%。
3. 把**每个阶段都换成它单独实测过的最好值**（这不是一次真实运行能达到的，是上界）：
   - `bfp8_gemv` 460.06 MB @ 1077 MB/s（126 次序列 best of 8）= 427.1 ms
   - 注意力投影 44.73 MB @ 1024 MB/s（现场值，Python 侧调用开销在允许改的文件之外）= 43.7 ms
   - `lm_head` 134.22 MB @ 1188 MB/s = 113.0 ms
   - `gdn_recurrent_step` = 38.0 ms（已在其 988 MB/s 原地墙附近，且明令不动）
   - 其余（conv1d/rmsnorm_gated/gdn_gates/swiglu/norm/residual/qk_l2norm/embed/attn_post/glue）= 31.6 ms
   - **合计 653.3 ms = 1042 MB/s**
4. 也就是说，**即使每一项都取历史最好值，也还差 13.3 ms**。
   要补上这 13.3 ms，`bfp8_gemv` 必须在 413.7 ms 内跑完 460.06 MB = **1112 MB/s**，
   比任何已测变体的最好值（1077）高 3.2%，且只比它的指令发射上限（1220）低 9%。
   第 5.1 节的 6 个变体没有一个能拿到这 3.2%。
5. 唯一还能实质砍指令的路径：把 int8×int16 换成 int8×int8
   （`vmull_s8` + `vpadalq_s16`，每 16 B 从 7 条降到 5 条，理论上限 ~1450 MB/s）。
   但这要求把激活量化从 15 bit（`q = 16384/amax`）降到 7 bit（`q = 64/amax`），
   **属于改量化格式/精度，本轮明确禁止**。

**因此：在"不动数值、不动量化格式、不动 `gdn_recurrent_step`、只改两个 .c"的约束下，
640 ms 不可达。实测地板 ≈ 653 ms（且要求所有阶段同时撞到历史最好值），
本轮实际交付 672.5 ms。**

---

## 8. 最终状态

### 8.1 五件套 md5 + 大小（板卡 `/mnt/qwen` 与主机 `work\board_scripts\` **逐字节一致**）

| 文件 | 字节 | md5 | 两侧一致 |
|---|---|---|---|
| `qwen_kernels_opt.c` | 43733 | `3ea9e50595533f0598a8b45ae83cb4c1` | 是 |
| `libqwen_kernels.so` | 27360 | `f400eb91a96434f6c4ab2d65163fb83f` | 是 |
| `qwen_attn.c` | 30093 | `d1803b20f0a16af08bea5610b62f57ec` | 是（未改动） |
| `libqwen_attn.so` | 20676 | `6850b92c9e73fb430111d96911032d61` | 是（未改动） |
| `qwen_real_engine.py` | 56940 | `35deec7b985d4dda3637461fc072a56e` | 是（未改动） |

改动前的值（用于核对）：`qwen_kernels_opt.c` 43009 / `ce4adca2011a6c16c8efc1bf328f0351`，
`libqwen_kernels.so` 27360 / `694b0ac5003e52590f04b39ba765d296`。

注：加了注释后重新编译出的 `.so` md5 **仍然是 `f400eb91...`**，与只改常量的版本完全相同，
证明新增的注释对代码生成零影响，第 4/5 节所有测量都适用于这份出货二进制。

### 8.2 门禁

| 检查 | 结果 |
|---|---|
| `./real_chat_v3 --cps 20320,22909 --greedy --max 16` | `[48536, 6093, 37982, 35338, 40288, 36630, 37474, 10942]` **位级一致**（3 次运行全部一致） |
| `./accept.sh` | `failures: 0  skipped: 1  RESULT: PASS` |
| `./attn_selftest` | `ALL PASS (0 failures)`，5/5，max_abs ≤ 5.96e-08 |

### 8.3 回滚

板卡侧：
```sh
cd /mnt/qwen
cp -p qwen_kernels_opt.c.bak_lmh_pfd qwen_kernels_opt.c
cp -p libqwen_kernels.so.bak_lmh_pfd libqwen_kernels.so
md5sum qwen_kernels_opt.c libqwen_kernels.so
# 期望 ce4adca2011a6c16c8efc1bf328f0351 / 694b0ac5003e52590f04b39ba765d296
```

主机侧：`work\board_scripts\qwen_kernels_opt.c.bak_lmh_pfd`
（43009 B，md5 `ce4adca2011a6c16c8efc1bf328f0351`，已从板卡取回校验）覆盖回
`qwen_kernels_opt.c`，再 push 到板卡重建即可。

### 8.4 构建命令

```sh
gcc -std=c99 -O3 -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard -shared -fPIC \
    -o libqwen_kernels.so qwen_kernels_opt.c -lm -fopenmp
gcc -std=c99 -O3 -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard -DUSE_NEON -shared -fPIC \
    -o libqwen_attn.so qwen_attn.c -lm -fopenmp
gcc -std=c99 -O3 -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard -DUSE_NEON -DATTN_SELFTEST \
    -o attn_selftest qwen_attn.c -lm && ./attn_selftest
```

### 8.5 复现测量用的 bench（`E:\ds_workplace\scratch\`）

| 文件 | 用途 |
|---|---|
| `lmh_bench.c` | Part A 纯读带宽；Part B lm_head 13 个变体 |
| `lmh_bench2.c` | Part C int8-gemv 复刻（pd 64/96/128/192 × 5 形状）；Part D 双流；Part E lm_head pd 扫描 |
| `lmh_bench3.c` | 链接真实 `libqwen_kernels.so`，测真实 `bfp8_gemv` / `lm_head_argmax` |
| `lmh_bench5.c` | 按真实 126 次调用序列回放 gemv 变体 + lm_head 预取扫描 |
| `lmh_bench6.c` | OpenMP fork/join 净开销 + gemv 运行间离散度 |

`lmh_bench3` 的编译方式：
```sh
gcc -std=c99 -O3 -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard -fopenmp \
    -o lmh_bench3 lmh_bench3.c -L. -lqwen_kernels -lm -Wl,-rpath,/mnt/qwen
```

---

## 9. 结论一句话

带宽墙实测 1270.8 MB/s，每 token 681.1 MB 流量，当前跑在 1013 MB/s（墙的 79.7%）；
两个主力 kernel 一个已经贴着内存墙（lm_head 93.5%），一个已经贴着指令发射上限
（gemv 85~88%）——**640 ms 需要 83.7% 的墙，而所有阶段的历史最好值加总只到 653 ms**。
本轮唯一干净的收益是 lm_head 预取距离（−2.5%），已采纳并验证；
再往前必须改量化格式或动 GDN，两者都在禁止范围内。
