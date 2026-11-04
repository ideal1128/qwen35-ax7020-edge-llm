# P1：全注意力 C + NEON 内核（已上板验证）

> 状态：**已部署、已实测、可一键回滚**
> 结果：注意力算子 **4.19× 加速**，长上下文每 token 省 **363 ms**，输出 ids 完全不变

---

## 1. 为什么做这个

`qwen_real_engine.py` 的 `PagedAttentionPool.compute_attention()` 是**纯 numpy**：

```python
k_exp = np.repeat(k_all, 4, axis=0)     # [8, T, 256] —— T=544 时 4.5 MB
v_exp = np.repeat(v_all, 4, axis=0)     # 同上
logits = np.einsum('hd,htd->ht', q, k_exp) * 0.0625
exp_l = np.exp(logits - max_l)
probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
out = np.einsum('ht,htd->hd', probs, v_exp).reshape(-1)
```

三层浪费：

1. **`np.repeat` 把 GQA 4:1 映射实体化** —— 每层每 token 新建 4.5 MB 数组，纯粹是搬运
2. **两个 einsum 走 numpy 通用路径** —— 完全没有利用 NEON
3. **代价随 T 线性增长** —— 实测 T=544 时单层就要 79.5 ms，6 层合计 477 ms

这就是长对话变慢（从 17 ms/tok 涨到 1214 ms/tok）的主要原因。

---

## 2. 设计

### 2.1 消除实体化

不建 `k_exp` / `v_exp`，直接按索引取：

```
logits[h][t] = Σ_d q[h][d] · k[h/4][t][d]
out[h][d]    = Σ_t probs[h][t] · v[h/4][t][d]
```

4 个 query head 共享 1 个 kv head，所以**读一次 K 行，同时给 4 个 q 行做点积**。

### 2.2 三阶段

| 阶段 | 结构 | 内存行为 |
|---|---|---|
| **1. logits** | kh 外层，t 次之，d 内层；8 head 并行计算 | K 每层只读一次 |
| **2. softmax** | 每个 head 独立，O(T) | 全在缓存 |
| **3. 输出** | dblk 外层（4 块 × 4 lane），t 内层；8 head 并行计算 | V 每层只读一次，累加器常驻寄存器 |

阶段 3 的关键约束：`acc[4]` 必须常驻寄存器，**V 只读一次**。若改成 t 外层、d 内层，
就得把 1024 个输出点反复写回内存。

### 2.3 行指针预算（第二版优化）

第一版内层循环对每个 token 都要做：

```c
int b = block_table[t / block_size];   /* 每次除法 */
int o = t % block_size;                 /* 每次取模 */
const float *vp = v_pool + b*stride_b + layer*stride_l + kh*stride_kh + o*head_dim + d;
```

而阶段 3 外层还有 64 个 lane 块，所以这套查表会重复 **64 × T** 次，T=544 时约 3.5 万次冗余查表。

第二版改成**每层每个 kv head 先建一次指针表**（`g_kptr` / `g_vptr`，各 2048 项 = 8 KB），
内层直接 `vptrs[t]` 流式读取：

```
第一版 T=544: 25.17 ms/层   第二版 T=544: 18.97 ms/层   (再快 25%)
```

### 2.4 NEON 关键代码

```c
/* 阶段 1：一次 K 装载，给 4 个 q 头做 FMA */
for (; d + 4 <= head_dim; d += 4) {
    float32x4_t kv = vld1q_f32(kp + d);
    for (i = 0; i < ratio; i++)
        acc[i] = vmlaq_f32(acc[i], vld1q_f32(q + i*head_dim + d), kv);
}

/* 阶段 3：一次 V 装载，给 4 个概率做 FMA */
for (t = 0; t < total_tokens; t++) {
    float32x4_t vv = vld1q_f32(vptrs[t] + d);
    acc[0] = vmlaq_n_f32(acc[0], vv, p0[t]);
    acc[1] = vmlaq_n_f32(acc[1], vv, p1[t]);
    ...
}
```

### 2.5 内存布局

`k_pool` / `v_pool` 是 numpy C 连续数组，形状
`(num_blocks=128, n_layers=6, n_kv_heads=2, block_size=16, head_dim=256)`。

元素 `[b][l][kh][o][d]` 的偏移是：

```
((((b * 6 + l) * 2 + kh) * 16 + o) * 256 + d)
```

逻辑 token `t` 对应的物理块是 `block_table[t / 16]`，块内偏移是 `t % 16`。

---

## 3. 实测结果（板上，非估算）

### 3.1 算子级微基准

`attn_microbench.py`，单层，5 次取平均：

| T | numpy | C+NEON 第一版 | C+NEON 第二版 | 加速比 |
|---|---|---|---|---|
| 20 | 3.31 ms | 0.80 ms | **0.61 ms** | **5.42×** |
| 100 | 16.43 ms | 3.90 ms | **2.81 ms** | **5.84×** |
| 300 | 50.38 ms | 12.41 ms | **9.09 ms** | **5.54×** |
| 544 | 79.52 ms | 25.17 ms | **18.97 ms** | **4.19×** |
| 1000 | 163.83 ms | 47.82 ms | **35.95 ms** | **4.56×** |
| 2000 | 290.67 ms | 100.30 ms | **75.44 ms** | **3.85×** |

数值一致性：`max_abs` 全部 ≤ 1.2e-07（float32 舍入级别）。

### 3.2 折算到整模型（6 层全注意力）

| T | numpy 总计 | C 总计 | 每 token 节省 |
|---|---|---|---|
| 20 | 19.9 ms | 3.7 ms | 16.2 ms |
| 300 | 302.3 ms | 54.5 ms | 247.7 ms |
| **544** | **477.1 ms** | **113.8 ms** | **363.3 ms** |
| 1000 | 983.0 ms | 215.7 ms | 767.3 ms |
| 2000 | 1744.0 ms | 452.7 ms | 1291.4 ms |

### 3.3 端到端（greedy，短上下文）

| 指标 | 改前 | 改后 | 变化 |
|---|---|---|---|
| prefill | 713.4 ms/tok | **706.7 ms/tok** | −0.9% |
| 生成 | 812.6 ms/tok | **794.3 ms/tok** | **−2.3%** |
| 吞吐 | 1.231 tok/s | **1.259 tok/s** | +2.3% |
| **ids** | `[48536, 6093, 37982, 35338, 40288, 36630, 37474, 10942]` | **完全相同** | — |

### 3.4 长上下文预测

改前 T=544 实测 1214 ms/tok。注意力部分从 477 ms 降到 114 ms，**预测 851 ms/tok**，即：

```
1214 ms → 851 ms    (长对话提速 1.43×)
```

这个数字由微基准直接推算（已扣掉 numpy 实测值），不是拍脑袋的模型估计。

---

## 4. 为什么 18.97 ms 还不到理论下界

T=544 时的理论下界：

- **访存**：K + V = 2 × 544 × 256 × 4 B × 2 kv head = 2.23 MB，按 1270.8 MB/s 算是 **1.75 ms**
- **计算**：8 head × 544 t × 256 d × 2（logits + 输出）= 2.23 M MAC，单核 4-wide NEON 约 **0.73 ms**

实测 18.97 ms，是下界的 10 倍。原因：

1. **单线程** —— 内核没有上 OpenMP，只用了单个 A9 核；双核并行能再快约 1.8×
2. **阶段 3 的访存模式是列优先** —— 外层 dblk 是 4 个点，内层是 t，每次只取 V 行的 16 字节。
   对行主序数组来说这种遍历 cache 不友好，实际流量约为理论值的 4 倍
3. **标量概率装载** —— `vmlaq_n_f32` 每个 t 要做 4 次标量 load + 4 次 FMA，标量向量混用会有流水线停顿

**要继续优化的方向**（按性价比排序）：

| 优化 | 预期收益 | 工作量 |
|---|---|---|
| 阶段 3 上 OpenMP（双核分行） | ~1.8× | 小 |
| 阶段 3 改成 t 外层 + 8 个 lane 块并行 | ~1.5× | 中 |
| 把 probs 预先转成 `[t][head]` 做 4 路向量连续 | ~1.2× | 小 |

但即使优化到 5 ms，整模型也只再快约 55 ms/token（8%）。**瓶颈永远是每 token 必须搬运的
681.1 MB 总流量（2026-11-04 更正：其中权重 504.8 MB、lm_head 表 134.2 MB、GDN 递推状态 37.7 MB；
旧文写的 777.6 MB 是错的，见 `MEASURED_CORRECTIONS.md` §7），不是注意力。**

---

## 5. 验证过程

### 5.1 算法与内存布局（主机侧）

`attn_alg_verify.py` 把 C 的 `full_attn_ref()` 逐行移植到 Python，和原 numpy 实现对比：

```
  tokens=   7 layer=3 blocks= 1  max_abs=1.362e-07  PASS
  tokens=  16 layer=0 blocks= 1  max_abs=7.755e-08  PASS
  tokens=  33 layer=1 blocks= 3  max_abs=8.849e-08  PASS
  tokens=  80 layer=2 blocks= 5  max_abs=6.536e-08  PASS
  tokens=  47 layer=5 blocks= 3  max_abs=7.154e-08  PASS
=== ALL PASS ===
```

覆盖：单块部分填充、刚好整块、跨块边界、多块、最后一层。

### 5.2 NEON vs 标量（板上）

```
=== qwen_attn self test (NEON=1) ===
case blocks=4   T=  16  max_abs=5.960e-08  PASS
case blocks=1   T=   7  max_abs=5.960e-08  PASS
case blocks=4   T=  33  max_abs=2.980e-08  PASS
case blocks=34  T= 544  max_abs=1.676e-08  PASS
case blocks=128 T=2048  max_abs=1.676e-08  PASS
=== ALL PASS (0 failures) ===
```

### 5.3 集成补丁

| 测试 | 结果 |
|---|---|
| `--check` | 正确报告 2 个调用点 |
| apply | 两处都替换，`py_compile` 通过 |
| 幂等 | 二次运行输出 `already patched` |
| `--revert` | 恢复后与原件 **md5 完全一致** |

**补丁发现的坑**：调用点是 **2 个**，不是 1 个 —— `_on_full_attn_cb`（C 回调路径，
`forward_token_c` 实际在用的）和 `_layer_fwd`（纯 Python 回退路径）。两处都必须换，
否则两条路径结果会不一致。

### 5.4 端到端 ids

改前改后 greedy 输出 ids **逐位相同**，见 §3.3。

---

## 6. 部署与回滚

```bash
cd /mnt/qwen

# 编译
gcc -std=c99 -O3 -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard \
    -DUSE_NEON -shared -fPIC -o libqwen_attn.so qwen_attn.c -lm -fopenmp

# 自测
gcc -std=c99 -O3 -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard \
    -DUSE_NEON -DATTN_SELFTEST -o attn_selftest qwen_attn.c -lm
./attn_selftest

# 打补丁（自动备份 .pre_attn）
python3 patch_engine_attn.py

# 回滚
python3 patch_engine_attn.py --revert
```

**回滚安全性**：`.so` 缺失或调用异常时 `_attn_dispatch` 自动回退 numpy，不会崩溃。

---

## 7. 板上产物

| 文件 | 大小 | md5 |
|---|---|---|
| `qwen_attn.c` | 30,093 B | `d1803b20f0a16af08bea5610b62f57ec` |
| `libqwen_attn.so` | 20,676 B | `6850b92c9e73fb430111d96911032d61` |
| `attn_selftest` | 板载编译 | `c2c8cd81a0513376d5ecf43fedd7dc29`（5/5 PASS） |
| `qwen_real_engine.py` | 56,940 B | `35deec7b985d4dda3637461fc072a56e` |

> 上面是加入 §8 的 `attn_post_f32` 之后的最终状态。`full_attn_f32` 的 ABI 和算法没变，
> 所以 §3 的性能数字依然成立。

---

## 8. 后续：attn_post_f32 —— 把 numpy 胶水也搬进 C

### 8.1 问题

`_on_full_attn_cb` / `_layer_fwd` 里除了 4 个 gemv（q / k / v / o），其余全是 numpy 胶水：
`rmsnorm_heads` ×2、`apply_rope` ×2、sigmoid gate、slice、KV append、residual。

这块板子的 numpy 1.8.2 没有 NEON ufunc，而且**每个小数组 op 有约 100 µs 的固定开销**：

| 操作 | 规模 | 实测 |
|---|---|---|
| `np.add` | 2048 | 39 µs |
| `np.mean(axis=-1)` | 8×256 | 244 µs |
| `np.exp` | 2048 | 173 µs |
| `np.sqrt` | 2048 | 295 µs |
| `rmsnorm_heads` | 8×256 | 539 µs |
| `apply_rope` | 8×256 | 297 µs |
| `sigmoid_gate` | 8×256 | 406 µs |

合计 **17.8 ms/token**，其中真正的注意力计算只占约 2.4 ms —— 也就是说 15.4 ms 全花在胶水上。

### 8.2 做法

在 `qwen_attn.c` 新增 `attn_post_f32()`，一次调用完成：

```
rmsnorm_heads(q/k) → rope(q/k) → KV append → attention → sigmoid gate
```

输出 `gated_attn[2048]`。`full_attn_f32` 的 ABI 不变，两条路径可以共存。

### 8.3 逐位等价

numpy 的 `add.reduce` 求和顺序在 1.8.2 上是实现细节，所以先用 `_attn_post_init()`
做一次性标定（本次标定到 mode 0，即顺序累加），再用相同的 4 lane 累加方式在 C 里复刻。

- `attn_post_verify.py` 8 个用例：`max_abs=0.000e+00, bitdiff=0`
- 负向用例（`nh=3,nkv=2`、`head_dim=512`）被正确拒绝
- 启动时自动标定；任何不一致自动回退 numpy（`_attn_post_numpy()`），且每个 pool 只尝试一次

### 8.4 效果（QWEN_PROFILE=1，26 token）

| 指标 | 改前 | 改后 |
|---|---|---|
| `py_full_attn` | 66.8 ms/token | **51.58 ms/token** |
| numpy 胶水 | 17.8 ms/token | **4.2 ms/token** |
| 每 token | 704.9 ms | **689.8 ms（1.450 tok/s）** |
| token ids | `[48536, 6093, 37982, 35338, 40288, 36630, 37474, 10942]` | **完全一致** |

改后的单次回调细分（162 次调用）：`gemv_q` 3.944 / `gemv_o` 2.222 / **`attn_post_C` 1.135** /
`gemv_k` 0.563 / `gemv_v` 0.550 / `residual` 0.086 / `rmsnorm_in` 0.068 ms/次。

### 8.5 相关文件

| 文件 | 用途 |
|---|---|
| `attn_post_verify.py` | 逐位校验（8 个正向 + 2 个负向用例） |
| `prof_attn_post.py` | 回调内部细分计时 |
| `patch_engine_attn_post.py` | 打补丁 / `--check` / `--revert` |
