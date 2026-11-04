# GDN Recurrent Step 优化结果（2026-11-04，任务 task-2）

目标：把 `gdn_recurrent_step` 从 **52.389 ms/token** 降到 **≤35 ms/token**，
整 token 从 **687.51 ms** 降到 **≤675 ms**，且 token id 必须逐位一致。

结果：**37.779 ms/token（-27.9%），整 token 673.02 ms（-2.1%）**。
逐位一致、`accept.sh` PASS 全部满足；**35 ms 这一项未达到**，原因是它要求
1029 MB/s 的"边读边写"带宽，而本板实测上限只有 943–988 MB/s（见下文）。
整 token 目标（≤675 ms）已达成。

## 只改了一个函数

`board_scripts/qwen_kernels_opt.c` 第 9 节 `gdn_recurrent_step`（约 611–805 行）。
`qwen_attn.c` / `qwen_real_engine.py` / `chat_v3.py` 未改动，numpy 侧的
`LayerStateDict` 形状、`reset_state`、清零逻辑都不需要变。

### 状态布局：改成 4×4 分块的转置

```
state[h][((jg*ng + ig)*4 + l)*4 + t] = S[h][4*ig + t][4*jg + l]
```

一个 head 的 128×128 状态被切成 32×32 个 16 float 的小块；块内同一列 j0+l 的
四个 float 就是连续的四个 i。这样 `VLD4/VST4` 一次搬完整块，返回的
`val[t] = [S[i0+t][j0] .. S[i0+t][j0+3]]` 正好是"通道 = j"的四路累加器：

* `kv[j] = Σ_i k[i]*S[i][j]` → `vmlaq_lane_f32` 按标量广播，无水平求和
* `T = a*T + k*d` → 每个 i 一次向量 axpy

每 16 个状态元素从"8 次标量 load + 6 次 VTRN + 8 次 store"降到
"1 次 VLD4 + 8 次 VMLA + 4 次 VMUL + 1 次 VST4"。

状态数组全零初始化、且**只有这一个内核读写它**，所以这是纯内部约定变更
（0 在任何布局下都一样），外部无需知情。

### 逐位一致的依据

老内核的四路累加器通道是四个不同的 j，按 i 递增喂入，因此每一列的 kv/qs 都是
对 i 的严格递增求和。新内核每个块内 t=0..3 依次累加、块间按 ig 递增，
求和顺序完全相同；qk、d、out 和状态更新沿用同样的 NEON 运算序列
（先 `a*T` 再 `+k*d`，Cortex-A9 没有 FMA，不存在融合改变舍入的问题）。

### 其他两点

* 并行粒度从 16 个 head 改成 32 个列组/head（`collapse(2)`，2 线程）
* 每组 2 KB 处理完立刻进入第二轮，读到的行仍在 L1；同时用
  `__builtin_prefetch` 预取**下一组**的 2 KB。关掉预取会慢 0.45 ms/call
  （786 → 610 MB/s），这是本优化里第二大的单项收益。

## 实测数据（板端，热缓存）

| 指标 | 改前 | 改后 | 目标 |
|---|---|---|---|
| `gdn_recurrent_step` | 52.389 ms/token | **37.779 ms/token** | ≤35 ✗ |
| 整 token | 687.51 ms | **673.02 ms** | ≤675 ✓ |
| 内核微基准（1 次调用） | 3.49–3.52 ms | **2.64–2.67 ms** | — |
| 有效状态带宽 | 690 MB/s | **953 MB/s** | — |

其它 stage 不变（非 GDN 部分 635.12 → 635.24 ms/token，噪声内）：
`bfp8_gemv 441.0`、`forward_mlp 129.7`、`lm_head_argmax 117.7`、
`py_full_attn 51.6`、`conv1d_silu 8.25`、`gdn_gates 5.13`。

验证：
* `gdn_rec_verify.py`（零状态 + 随机非零状态各 6 步，out 与 state 按 uint32 逐位比较）→ `RESULT: PASS`
* `chat_v3.py --cps 20320,22909 --greedy --max 8` →
  `[48536, 6093, 37982, 35338, 40288, 36630, 37474, 10942]`，与改前完全一致（`--max 16` 同前缀）
* `./accept.sh` → `RESULT: PASS`（failures: 0）

## 为什么 35 ms 达不到

每个 token 必须把 18 个 GDN 层的状态（18 MB）完整读一遍（kv/qs 需要每个元素）
再写一遍（`S = a*S + k⊗d` 是稠密更新），合计 **36 MB/token，无法再减**。

35 ms 对应 36 MB / 35 ms = **1029 MB/s 的读写混合带宽**。板端实测
（`bench2.c`，18 MB 缓冲、2 线程、5 次取最好）：

| 模式 | MB/s |
|---|---|
| 纯读 | 1190 |
| 纯写 | 1880–2126 |
| 拷到另一个缓冲（64 KB 分块） | 961 |
| 原地读+写（2/8/64 KB 分块） | 820–879 |
| 原地读+写 + PLD 预取（8/64 KB） | 944 / **988** |

即"原地读+写"这一必需模式的实测上限是 988 MB/s（38 ms/token）。
当前内核端到端已经跑到 953 MB/s，是上限的 96%；要 35 ms 需要比
实测最好成绩再快 4–9%，属于 DDR 读/写切换开销，不是指令数问题
——把每元素指令数从 2.4 降到 1.3（v3→v6 四轮重构）只带来 <5% 变化，
而关掉预取立刻慢 17%。若将来整 token 还要再降，方向在
`bfp8_gemv`（441 ms/token，65%）而不在 GDN。

## 重建 / 回滚

```sh
cd /mnt/qwen
gcc -std=c99 -O3 -mcpu=cortex-a9 -mtune=cortex-a9 -mfpu=neon \
    -mfloat-abi=hard -fopenmp -shared -fPIC qwen_kernels_opt.c \
    -o libqwen_kernels.so -lm -fopenmp

# 回滚（原始版本）
cp -f qwen_kernels_opt.c.bak_task2 qwen_kernels_opt.c
cp -f libqwen_kernels.so.bak_task2 libqwen_kernels.so
```

## 文件与 md5

| 文件 | 大小 | md5 |
|---|---|---|
| `qwen_kernels_opt.c`（新版） | 43009 | `ce4adca2011a6c16c8efc1bf328f0351` |
| `qwen_kernels_opt.c.bak_task2`（原始） | 39674 | `e36e9c3524d25a985c91d7f49d94e7e2` |
| `libqwen_kernels.so`（新版） | 27360 | `694b0ac5003e52590f04b39ba765d296` |
| `libqwen_kernels.so.bak_task2`（原始） | 27172 | `9c82b90fe0cc262a5d517a0c2bd2943a` |
| `gdn_rec_verify.py`（A/B 工具，新增） | 6548 | `4ba6eafc50158c4dfde07fecae9aabed` |
| `gdn_test.c`（内核独立 harness） | 7856 | — |

`gdn_rec_verify.py` 用法（第三个参数是状态布局：`r` 行主序 / `t` 转置 / `b` 4×4 分块）：

```sh
python3 -u gdn_rec_verify.py /mnt/qwen/libqwen_kernels.so.bak_task2 \
                             /mnt/qwen/libqwen_kernels.so b
```
