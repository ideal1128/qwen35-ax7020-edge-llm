# gemv_core: functional repair + timing closure report

Target: `aclk` = 6.000 ns (166.67 MHz), WNS >= 0.000 ns on xc7z020clg400-2.
Flow: out-of-context `synth_design -mode out_of_context` -> `opt/place/route`,
`create_clock -period 6.000 [get_ports aclk]`, Vivado 2025.2.
Script: `rtl/ooc_gemv.tcl`.

---

## 1. Status of the five requested items

| # | Item | Status |
|---|---|---|
| 1 | Bug 2 + Bug 3 (output fixed point, saturation) | **DONE** |
| 2 | Bug 1 (`xscale` phase) | **DONE** |
| 3 | Bug 4 (error detection) | **DONE** |
| 4 | Independent testbench + realistic exponents | **DONE** |
| 5 | Pipeline for timing at 6.0 ns | **DONE, WNS = +0.461 ns** |

Not done (out of RTL scope, **required before end-to-end use**): the host-side
changes for the two protocol changes (see section 6).

---

## 2. What was wrong and what changed

### Bug 1 (fatal) - `xscale[b]` was never applied

The core computed `y[m] = sum_b block_dot[m][b] * 2^(e[m][b]-134)` and dropped the
per-block activation scale entirely. `xscale` is a general float
(`amax_b / 16384`, see `pl_quant`), not a power of two, and the block index is
lost once a row is accumulated, so the host cannot correct it afterwards.
Measured 19.7x spread across the blocks of one row.

Fix: a new stream phase between the `x` phase and the per-row exponent phase.
The host streams its existing `float xscale[nb]` array, 4 float32 per beat.
The RTL decodes the float and merges its exponent with the BFP8 byte exponent,
so the whole per-block scale stays a single power of two:

```
xscale   = (mant / 2^16) * 2^Ex          mant = {1'b1, f[22:7]},  Ex = eb - 127
contrib  = block_dot * mant * 2^(Ex + e - 110)      (accumulator carries 2^40)
```

Header byte `tdata[39:32]` = `xs_beats` = `ceil(nb/4)`; `xs_beats = 0` means
identity scale, which reproduces the old arithmetic exactly.

### Bug 2 (fatal, previously masked) - output was rounded to an integer

`row_out_rounded = (cur_row_sum_64 + 2^19) >>> 20` emitted `round(y)`. With
realistic data (`|y|` mean 0.77, max 1.82) that is 53.7 % mean relative error,
84 % of outputs off by more than 10 %, and 31 % rounding to exactly 0.

Fix: emit the fixed-point word and let the host scale it.

```
m_axis_tdata = clamp(row_acc, -(2^31)<<20, (2^31-1)<<20)[51:20]
             = floor(y * 2^20)   as int32          host: y = word * 2^-20
```

The `+2^19` rounding bias was removed (the accumulator starts each row at 0).

### Bug 3 - silent integer wrap / silent block drop

Old behaviour: `e <= 51` flush to zero, `e 52..144` exact, `e 145..150` the row
sum could wrap int64, `e 151..177` a single block could wrap int64, and
`e >= 178` forced the contribution to `0` - silently dropping the block.

Fix: saturating 64-bit add, saturating shift, and an **exact** left-shift
overflow test on `|prod|`, all reported through the sticky `err_arith_ovf` bit.
`e > 177` no longer zeroes anything. Clamping the *negative* shift side is
lossless (any right shift of 49+ is already an exact flush to 0 / -1) and is
deliberately not flagged.

### Bug 4 - a `tlast` in the exponent phase was swallowed

`ST_EXP` never looked at `s_axis_tlast`, so a premature `tlast` was ignored and
the FSM consumed the next header as weight data. The single-contiguous-burst
requirement is retained, but violations are now flagged and the FSM drains
instead of desynchronising. Added three sticky flags:

```
dbg[12] err_row_bnd       header bpr not a whole number of blocks, exponent
                          group cannot cover every block, or a burst ended
                          mid-row
dbg[11] err_tlast_phase   tlast seen in header / x / xscale / exponent phase
dbg[10] err_arith_ovf     shift clamp, exact left-shift overflow, or add sat
dbg[9]  sat_sticky        the output word was clamped to an int32 rail
```

Flags clear only on `aresetn` / `dbg_rst`.

### Pipeline

5 stages -> **12 stages**. New stages: float decode, `bd*mant` multiply,
two-stage 64-bit barrel shift, saturating accumulate, saturated output.

---

## 3. Timing progression (OOC, 6.000 ns)

| iter | change | WNS (ns) | TNS (ns) | failing endpoints | worst path |
|---|---|---|---|---|---|
| base | original 5-stage core | **-4.149** | -323.964 | 260 / 13937 | `row_exp_reg[0][9][2]/C` -> `of_data_reg/RAMC`, 9.867 ns, 23 levels (`CARRY4=11 LUT6=7 LUT3=1 LUT5=1 MUXF7=2 MUXF8=1`) |
| 1 | 11-stage, float xscale, exact ovf test | -0.682 | -1.821 | 9 / 16010 | `prod_q_reg[0]/C` -> `ovf_q_reg/D`, 6.681 ns, 19 levels (`CARRY4=12`) - the 49-bit `~prod+1` negate |
| 2 | 12-stage: split the barrel shifter over two stages; build `|prod|` on a second DSP from `|bd|` instead of a 49-bit negate | +0.215 | 0.000 | **0 / 16368** | `sh_q_reg[9]/C` -> `err_arith_ovf_reg/D`, 5.743 ns, 7 levels (74 % route) |
| 3 | register the overflow shift amount one stage early (`ovf_sh_q`) so the compare starts from a flop | **+0.461** | 0.000 | **0 / 16388** | DSP48E1 multiply, 4.291 ns, 1 logic level - structural floor |

Final: **WNS +0.461 ns, TNS 0.000, 0 failing endpoints. WHS +0.076 ns,
THS 0.000, 0 hold failures. WPWS +1.870 ns.**

Removal achieved: 4.149 + 0.461 = **4.61 ns** off the critical path.

The remaining critical path is the `bd_q * mant_q` DSP48E1 multiply (4.291 ns of
which 3.677 ns is the DSP itself). Further improvement would need the multiply
itself split, which is not possible on one DSP.

### Utilization (OOC, final)

| | final | original | delta |
|---|---|---|---|
| Slice LUTs | 3168 (5.95 %) | 2571 (4.83 %) | +597 |
| - LUT as Logic | 1767 | 1172 | +595 |
| - LUT as Memory | 1401 | 1399 | +2 |
| Slice Registers | 3575 (3.36 %) | 1685 (1.58 %) | +1890 |
| Block RAM Tile | 0 | 0 | 0 |
| DSP48E1 | 24 (10.91 %) | 20 (9.09 %) | +4 |
| F7 / F8 Muxes | 328 / 160 | 139 / 64 | +189 / +96 |

The `x` memory and the xscale memory stayed in LUTRAM (no BRAM added); the
register cost is the pipeline plus the 11-bit context shift registers.

---

## 4. Testbench: 10/10 PASS

The old `tb_gemv_core.v` was **vacuous** - its `calc_row_ref` was a transcription
of the RTL, so it reproduced every bug bit-exactly, and its exponent generator
(`e = 120 + seed%36`) overflowed the accumulator on its own data.

The replacement is an independent golden model: it evaluates the same expression
`pl_apply()` evaluates, in `double`:

```
y[m] = sum_b (double)dots[m][b] * 2^(exps[m][b]-134) * xscale[b]
dots[m][b] = sum_{i<64} w_int8 * x_int16           (exact integer)
xscale[b]  = amax_b / 16384                        (float32 on the wire)
```

and compares `W / 2^20` against it. The stimulus amplitude is auto-scaled so the
golden `|y|` lands near 1.2 - the regime the real model runs in.

| case | result |
|---|---|
| legacy `dequant_en=0`, K=128 M=3 | 6 raw block dots **bit-exact** |
| legacy `dequant_en=0`, K=1024 M=8 | 128 raw block dots **bit-exact** |
| dequant K=128 M=3, e 118..130, 16x xscale spread | pass, `|y|max` 1.39 |
| dequant K=1024 M=8, e 118..130, 16x spread | pass, `|y|max` 1.39 |
| dequant K=128 M=8, e 118..130, 16x spread | pass, `|y|max` 1.22 |
| dequant K=1024 M=4, **1024x** xscale spread | pass, `|y|max` 2.31 |
| dequant K=512 M=4, flat xscale | pass, `|y|max` 1.48 |
| dequant K=512 M=4, `xs_beats=0` identity path | pass, `|y|max` 38.75 |
| saturation (e 240..250) | clamps to an int32 rail, `sat_sticky` set |
| flush (e 40..60) | ~0, no spurious sticky flags |

**Numeric error over 33 compared rows: max absolute error 1.71e-5,
max relative error 2.35e-5.** This is exactly the expected fixed-point
quantisation floor: the xscale mantissa is truncated to 16 fraction bits
(2^-16 = 1.53e-5 systematic, always downward), plus the output truncation of
`floor(y * 2^20)` (< 9.5e-7). It is ~4000x smaller than the 5.4e-2 that Bug 2
alone was producing.

Also asserted: exactly one `tlast` per burst, on the final beat; the three sticky
error bits stay clear on every legal burst.

---

## 5. Interface changes (both are backward-visible)

1. **Header byte `tdata[39:32]` is now `xs_beats`.** Bytes `[31:0]` are unchanged
   (`[15:0]` x_beats, `[23:16]` bpr_m1, `[30:24]` exp_beats_m1, `[31]` dequant_en).
   The port list, module name and parameters are untouched.
2. **`m_axis_tdata` is now `y * 2^20` as int32**, not `round(y)`.
3. **Latency is 12 cycles** from the last accepted weight beat to the output push
   (the beat is visible on `m_axis` one cycle later), up from 5.
   The AXI-Stream weight stream is still **one contiguous burst**; the new
   xscale phase sits *inside* the burst, between the `x` phase and the first
   exponent phase, so the DMA is not split.

---

## 6. Still required outside the RTL (NOT done here)

* `pl_apply.c` / `pl_ffn.py`: after the `x` phase, stream the existing
  `float xscale[nb]` array as 4 floats per beat and put `xs_beats = ceil(nb/4)`
  in header byte `[39:32]`. If the host has no xscale array, send
  `xs_beats = 0` to get the identity path.
* `pl_apply.c`: multiply each received int32 by `2^-20` (`* 9.5367431640625e-7f`)
  before using it. Without this the results are 2^20 times too large.
* The AXI DMA receive length is unchanged (one beat per row).
* Full in-context synthesis of the whole system has not been re-run - only the
  out-of-context run above. The core is now at 6.0 % LUTs, so in-context
  congestion should be the only risk.

---

## 7. Files

| file | note |
|---|---|
| `rtl/gemv_core.v` | rewritten core, ASCII-only (0 non-ASCII bytes) |
| `rtl/tb/tb_gemv_core.v` | independent golden-model testbench |
| `rtl/ooc_gemv.tcl` | OOC synth/place/route script at 6.000 ns |
| `02-RTL与Vivado仿真/gemv_core_pipelined.v` | deliverable copy of the core |
| `02-RTL与Vivado仿真/tb_gemv_core.v` | deliverable copy of the testbench |
| `rtl/gemv_core_orig_20261103.v` | untouched backup of the original core |
| `rtl/gemv_core_v1.v` | **untouched** |

Reports: `synth/ooc_gemv_base/` (original), `synth/ooc_gemv_v2/`, `v3/`, `v4/`
(final).
