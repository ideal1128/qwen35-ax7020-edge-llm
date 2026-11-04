# CORE_SIM_RESULTS.md -- functional simulation of the Qwen3.5 FPGA RTL blocks

Date: 2026-11-03
Simulator: Vivado Simulator 2025.2 (xvlog / xelab / xsim)
Host: Windows, Vivado at `D:\AMD\2025.2\2025.2\Vivado\bin`
Work area: `E:\ds_workplace\scratch\qwenwork` (junction to `...\work`; the real
path contains `(2)` and breaks the Vivado tools)

No RTL was modified by this work. Every result below was observed in a xsim
log; no test is reported as passing without a matching `PASS:` line.

## Summary

| # | Testbench | RTL under test | Result | Checks | Notes |
|---|-----------|----------------|--------|--------|-------|
| 1 | `tb_fwht64` | `fwht64.v` | **PASS** | 8/8 | impulse, constant, ordering, random vs 64x64 reference, saturation, bypass |
| 2 | `tb_bfp8_pack16` | `bfp8_pack16.v` | **FAIL** | 7/9 | correct for positive blocks; wrong exponent/mantissas whenever a block contains a negative sample |
| 3 | `tb_gdn_core` | `gdn_core.v` | **PASS** | 7/7 | mode=1 -> mode=2 dump all zero; two tokens match an integer reference (state + output) |
| 4 | `tb_conv1d_silu` | `conv1d_silu.v` | **FAIL** | 2/3 | beat count and bypass PASS; SiLU wrong for \|x\| >= 1.75 |
| 5 | `tb_conv_core` | `conv_core.v` (+ sub-blocks) | **FAIL** | 3/4 | 384 beats and tlast PASS; BFP8 exponent wrong because of the `bfp8_pack16` bug |

Open RTL defects: 3 confirmed + 1 latent synthesis issue (see "RTL bugs").
One defect found in `gdn_core.v` was fixed by someone else in a newer revision
of the file while these tests were running; that is documented in section 3.

## 1. fwht64 -- PASS (8/8)

Observed transform ordering: **natural (Hadamard) order**, exactly

    z[i] = (1/8) * sum_{j=0..63} (-1)^popcount(i & j) * a[j]

with an arithmetic `>>> 3` and saturation to [-32768, 32767]. This is not a
"looks about right" conclusion: a pseudo-random 64-element vector matched the
full 64x64 reference for all 64 outputs, and an impulse at index 1 produced
`+125 -125 +125 -125 ...` (the sign of output i is `(-1)^popcount(i&1)`), which
rules out sequency ordering.

| Check | Observed | Expected | Result |
|-------|----------|----------|--------|
| impulse 1000 at index 0 | all 64 outputs = 125 | all 125 | PASS |
| constant 100 | out[0] = 800, out[1..63] = 0 | 800 / 0 | PASS |
| impulse 1000 at index 1 | alternating +/-125 | +/-125 | PASS |
| ramp vector | 64/64 match reference | reference | PASS |
| pseudo-random vector | 64/64 match reference | reference | PASS |
| constant 32767 | out[0] = 32767 (saturated), rest 0 | saturate | PASS |
| bypass = 1 | 64/64 inputs passed through | pass-through | PASS |
| normal block after bypass | 64/64 match reference | reference | PASS |

Measured latency (matches the RTL header): 257 cycles from the first input
element to the first output element; 193 cycles from the last input element to
the first output element. `out_last` is asserted on the 64th beat and
`out_valid` is high for exactly 64 consecutive cycles.

## 2. bfp8_pack16 -- FAIL (7/9)

| Check | Observed | Expected | Result |
|-------|----------|----------|--------|
| 16 x 1000 | exp = 143, all mantissas = 125 | 143 / 125 | PASS |
| 16 x increasing 1000..16000 | exp = 147, mantissas 8..117 monotone | 147, monotone | PASS |
| 16 x 0 | exp = 0, mantissas = 0 | 0 / 0 | PASS |
| 16 x +32767 | exp = 148, mantissa clamped to 127 | 148 / 127 | PASS |
| 16 x -32768 | exp = **150**, mantissa = **-32** | exp = 149, mantissa = -64 | **FAIL** |
| pseudo-random block | exp = **150**, all mantissas wrong | exp = 148 | **FAIL** |
| out_ready = 0 for 8 cycles | out_valid/out_data held stable | hold | PASS |
| out_ready = 1 | beat consumed | consumed | PASS |

The failure is not limited to -32768: any block containing a negative sample is
mis-scaled. The pseudo-random block fails even though its most negative value
is only -28447.

## 3. gdn_core -- PASS (7/7)

| Check | Observed | Expected | Result |
|-------|----------|----------|--------|
| mode=1 (head_idx=8) then mode=2 dump of head 0 | 4096/4096 words = 0 | all zero | PASS |
| token 1, 64 output beats | 64/64 match reference | match | PASS |
| token 1 state dump | 4096/4096 match, all non-zero | match | PASS |
| token 2 (non-zero starting state), output | 64/64 match reference | match | PASS |
| token 2 state dump | 4096/4096 match | match | PASS |
| model comparison of the token-1 dump | 4096/4096 match the reference | match | PASS |
| mode=1 (head_idx=0) then dump | head 0 all zero | all zero | PASS |

Reference model (integer, mirrors the RTL arithmetic):
`u[j] = sum_i k[i]*S[i][j]`, `u_q = sat16(u,12)`, `e[j] = sat16(v[j]-u_q,0)`,
`S'[i][j] = sat16((alpha*S)>>>16 + (beta*k[i]*e[j])>>>31, 0)`,
`o[j] = sat16(sum_i q[i]*S'[i][j], 12)`.
Token 1 uses k,v = +/-16000, alpha = 32768, beta = 65535 and starts from an
all-zero state (so it exercises the beta path). Token 2 uses k = +/-30,
v = +/-300 on top of the token-1 state (so it also exercises the alpha term and
the non-zero `u` path).

**Revision note.** `gdn_core.v` was replaced by a different implementation
(29821 bytes, timestamp 17:36:22) while the first simulation was running. The
revision that was compiled first (still available as `gdn_core_ref.v`) had a
phase-4 bug: `ST_P4` never reloaded `k_cur`, so `ke = k_cur * e_buf[j_idx]`
used the leftover `k_buf[63]` for every row. Against that revision the dump
matched a "stale k" model on 4096/4096 words and the correct model on only
2048/4096. The current revision loads `k_cur <= k_buf[row_cnt]` in phase 4
(line 608) and passes all checks; the stale-k model now matches only 2048/4096
(i.e. it is no longer what the RTL does).

**Open format question (not a test failure).** The RTL header documents `e` as
Q3.13, but `e` is computed as `v - u_q` where both are Q2.14, so `e` is Q2.14
and the beta term needs `>>>32`, not `>>>31` (line 282/288). Measured: the dump
matches the `>>31` model 4096/4096 and the documented `>>32` model 0/4096, i.e.
the implemented beta*k*e term is 2x the documented value. Either the comment or
the shift needs to change; the test cannot decide which, so this is reported
for the owner to confirm.

## 4. conv1d_silu -- FAIL (2/3)

Configuration: W0 = 16384 (1.0 Q2.14), W1..W3 = 0, b = 0 for all 6144 channels,
so `y = x` and the output must be `x * sigmoid(x)`.

| Check | Observed | Expected | Result |
|-------|----------|----------|--------|
| one token | 1536/1536 output beats | 1536 | PASS |
| SiLU vs float reference | max error **3574 LSB**, 399 channels > 64 LSB | <= 64 LSB | **FAIL** |
| bypass = 1 | 6144/6144 channels unchanged | unchanged | PASS |

All 399 bad channels have `|x| >= 1.75`; the maximum error outside that window
is **19 LSB**, i.e. the datapath (conv sum, Q2.14 multiply, saturation) is
correct and only the sigmoid segment 7 is broken. The test ramp covers both
signs (x from -1.875 to +1.875).

## 5. conv_core -- FAIL (3/4)

Configuration: conv bypass = 1, FWHT disabled (bypass), so the raw samples are
packed 16 per BFP8 beat.

| Check | Observed | Expected | Result |
|-------|----------|----------|--------|
| beat count for one token | 384/384 | 384 | PASS |
| `m_axis_bfp_tlast` | on beat 383 | beat 383 | PASS |
| X on the output bus | none | none | PASS |
| beat 0 exponent / mantissa 0 | exp = **150**, mant0 = -12 | exp = 147 | **FAIL** |

The numeric failure is the same `abs17` defect as test 2: the block contains
negative samples, so the shared exponent is computed from `65536 + |x|`.

## RTL bugs

### Bug 1 (confirmed, high impact): `bfp8_pack16.v` line 47, `abs17` is wrong for negative inputs

    function [16:0] abs17;
        input signed [15:0] x;
        begin
            abs17 = x[15] ? (17'd0 - $unsigned(x)) : $unsigned(x);   // line 47
        end
    endfunction

`$unsigned(x)` is a 16-bit value, so it is zero-extended to 17 bits before the
subtraction. The 17-bit result wraps to `131072 - (65536 + x) = 65536 + |x|`
for x < 0. Measured: `abs17(-1000) = 66536`, `abs17(-32768) = 98304`.

Effect: `amax` always has bit 16 set, so `e_pos = 16`, `exp_byte = 150` and
`sh = 10` regardless of the real magnitude; every mantissa is scaled down by
`2^(16-E)` relative to the correct value. Any 16-sample block with one negative
value is wrong.

Suggested fix: sign-extend before negating, e.g.
`abs17 = x[15] ? (17'd0 - {1'b1, x}) : {1'b0, x};`

Observed vs expected, block of 16 x -32768:
`out_data = e0e0...e096`, expected `c0c0...c095` (exp 150 vs 149, mant -32 vs -64).

### Bug 2 (confirmed, range-limited): `conv1d_silu.v` sigmoid segment 7 is broken

`sig_base` (lines 56-71) has only 8 entries (segments 0..7) but `sigmoid_q15`
calls `sig_base(seg + 1)` (line 86). For `seg == 7` the call returns the
`default: 16384`, not `sigmoid(2.0)*32768 = 28862`. In addition line 88

    pos = b0 + interp[23:12];

takes an unsigned 12-bit slice of a 32-bit product; when `b1 < b0` (which only
happens in the broken segment 7) the negative interpolation becomes +4095.

Measured for x = -28680 (|x| = 1.7502): the RTL produces about -650 in Q2.14
where the true `x*sigmoid(x)` is about -4240; max error over the token is
3574 LSB. For |x| < 1.75 the max error is 19 LSB, so only this segment is bad.

Suggested fix: add `3'd8: sig_base = 16'd28862;` and take the interpolation as
signed (`$signed(interp[23:12])`).

### Bug 3 (latent, synthesis): `conv1d_silu.v` line 31 `m_axis_tlast` is declared an input but driven

    input  wire        m_axis_tlast,     // line 31
    ...
    assign m_axis_tlast = out_last;      // line 212

xvlog accepts this (the port net is aliased to the parent net, so simulation,
including `conv_core` tlast, works), but an input port may not be driven from
inside the module and Vivado synthesis will reject it. It should be
`output wire m_axis_tlast`.

### Fixed during testing: `gdn_core.v` phase 4 used a stale `k_cur`

Seen in the revision now preserved as `gdn_core_ref.v`; not present in the
current `gdn_core.v`. `ST_P4` only refreshed `q_cur` at `sub == 0`, so
`ke = k_cur * e_buf[j_idx]` used the `k_buf[63]` left over from phase 2, and
every row was updated with `k[63]` instead of `k[i]`. The current file loads
`k_cur <= k_buf[row_cnt]` in phase 4 and passes.

### Note (not a bug, but suspicious): `conv_core.v` `cfg_channel_len` is unused

The port is declared (line 40) but never read; the block is hard-wired to
6144 channels / 1536 beats. Either wire it up or drop the port.

## Reproduce

All testbenches and the batch script are in
`...\work\rtl\tb\` (accessible as `E:\ds_workplace\scratch\qwenwork\rtl\tb`).

One command runs all five (about 2-3 minutes):

    set HOME=E:\ds_workplace\scratch\vivado_home
    set USERPROFILE=%HOME%
    D:\AMD\2025.2\2025.2\Vivado\bin\vivado.bat -mode batch ^
        -source E:\ds_workplace\scratch\qwenwork\rtl\tb\run_sim.tcl

or one test at a time:

    D:\AMD\2025.2\2025.2\Vivado\bin\vivado.bat -mode batch ^
        -source E:\ds_workplace\scratch\qwenwork\rtl\tb\run_sim.tcl ^
        -tclargs tb_gdn_core

Manual xvlog/xelab/xsim run for a single test (the wrapper only automates this):

    cd E:\ds_workplace\scratch\qwenwork\sim\tb_fwht64
    D:\AMD\2025.2\2025.2\Vivado\bin\xvlog.bat ^
        E:\ds_workplace\scratch\qwenwork\rtl\fwht64.v ^
        E:\ds_workplace\scratch\qwenwork\rtl\tb\tb_fwht64.v
    D:\AMD\2025.2\2025.2\Vivado\bin\xelab.bat -timescale 1ns/1ps tb_fwht64 -s sim_tb_fwht64
    D:\AMD\2025.2\2025.2\Vivado\bin\xsim.bat sim_tb_fwht64 -runall

`-timescale 1ns/1ps` on xelab is required because `fwht64.v` has no
`` `timescale `` directive while the testbenches do.

Logs: `E:\ds_workplace\scratch\qwenwork\sim\<testbench>\xsim_out.txt`
(`xvlog_out.txt` and `xelab_out.txt` next to it). The Vivado tools write their
own `xvlog.log` / `xelab.log` / `xsim.log` in the same directory, which is why
the captured stdout uses `*_out.txt` names.

## Notes and limitations

* `conv1d_silu` and `conv_core` contain BRAM arrays with no reset port
  (`hist`). A real device has them zero after configuration; in simulation they
  start as X, so the testbenches preload `hist` to zero through a hierarchical
  reference before the first token. This does not touch the RTL.
* `gdn_core` state banks are X at time 0 as well; the mode=1 test writes all
  2048 words of every bank, which is also what makes the "dump is all zero"
  check meaningful.
* Coverage is functional, not exhaustive: no random backpressure on the AXI
  outputs of `gdn_core`, no multi-head isolation test, no non-trivial conv
  weights (W1..W3 were left at zero on purpose so the SiLU could be checked in
  isolation), and `conv_core` was only run with conv and FWHT bypassed.
* The BFP8 format deliberately discards mantissa 15 (the 128-bit beat holds one
  exponent byte and 15 mantissa bytes). The testbenches use that documented
  behaviour; the discarded 16th sample is not an error.

---

## 补记：三个 RTL bug 修复后，回归全部通过（2026-11-03 17:55）

上面第一轮回归是 `pass=2 fail=3`。三个 bug 已由 lead 修复，重跑 `run_sim.tcl`：

```
tb_fwht64        PASS
tb_bfp8_pack16   PASS
tb_gdn_core      PASS
tb_conv1d_silu   PASS
tb_conv_core     PASS
pass=5 fail=0
```

### 修复 1：`bfp8_pack16.v:47` abs17 负数溢出

```verilog
// 修复前：$unsigned(x) 在 17 位上下文里零扩展，取负得到 65536+|x| 再回绕
abs17 = x[15] ? (17'd0 - $unsigned(x)) : $unsigned(x);
// 修复后：先显式符号扩展
abs17 = x[15] ? (17'd0 - {1'b1, x[15:0]}) : {1'b0, x[15:0]};
```

验证：`16 x -32768 -> no abs overflow (exp=149, mant0=-64)` 通过。
修复前该用例得到 exp=150、尾数缩小 2^(16-E) 倍。

### 修复 2：`conv1d_silu.v` sig_base 缺第 9 个断点

sigmoid 在 0..2.0 上有 9 个断点（0.25 间隔），但 `sig_base` 只声明了 `seg` 0..7，
而 `b1 = sig_base(seg+1)` 在 seg=7 时索引 8，落进 `default`。
后果：`b1 - b0` 变成负数，`interp[23:12]` 是无符号切片，于是插值反号。

修复：索引位宽从 3 位扩到 4 位，补上 `4'd8 = 28862`（sigmoid(2.0)=0.8808），
并把 `default` 改成 28862（`a` 本来就被钳在 2.0）。

验证：`SiLU output within 64 LSB of x*sigmoid(x) for all 6144 channels` 通过。
修复前 |x|>=1.75 区间误差达 3574 LSB。

### 修复 3：`conv1d_silu.v:32` m_axis_tlast 方向错

端口声明为 `input`，但第 212 行 `assign m_axis_tlast = out_last;` 在驱动它。
xsim 容忍，Vivado 综合不会。改为 `output wire`。

### 待确认（不影响仿真）

`gdn_core.v` 头注释说 `e` 是 Q3.13，实际是 Q2.14，所以 `(beta*k*e)>>31` 是注释值的 2 倍。
状态 dump 与 `>>31` 模型 4096/4096 吻合、与 `>>32` 模型 0/4096 吻合。
即**实现是自洽的**，只是注释格式标错了 —— 应与 HLS 参考实现核对一次。

### 各模块独立时序（`create_clock -period 6.0`，166.67 MHz）

| 模块 | WNS | WHS |
|---|---|---|
| `fwht64` | **-0.710 ns** | +0.213 ns |
| `conv1d_silu` | -22.290 ns | +0.102 ns |
| `bfp8_pack16` | -27.603 ns | +0.143 ns |
| `conv_core` | -27.603 ns | +0.102 ns |
| `gdn_core` | -16.636 ns（在完整 BD 中测得） | +0.036 ns |

功能全部正确，但**除 `fwht64` 外都没收敛**。`fwht64` 只差 0.71 ns，最容易修。