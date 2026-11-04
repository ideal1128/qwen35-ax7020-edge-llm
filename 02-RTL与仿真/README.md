# RTL 与仿真

## 源码

| 模块 | 功能 |
|---|---|
| `gemv_core.v` | **BFP8 GEMV 加速核心**：16 路并行 int8×int16 乘加，12 级流水，64 位饱和累加，AXI-Stream 从口 |
| `gemv_core_pipe.v` | GEMV 核心的深流水变体 |
| `gdn_core.v` | Gated DeltaNet 递归状态更新加速器 |
| `gdn_core_ref.v` | GDN 行为级参考模型 |
| `conv1d_silu.v` | 因果一维卷积 + SiLU 融合，含 FWHT |
| `conv_core.v` | 卷积算子核心 |
| `fwht64.v` | 64 点快速 Walsh-Hadamard 变换 |
| `bfp8_pack16.v` | BFP8 分组打包 / 解包 |

## 验证

`tb_gemv_core.v` `tb_gdn_core.v` `tb_conv_core.v` `tb_conv1d_silu.v` `tb_fwht64.v` `tb_bfp8_pack16.v`
六个模块均有独立 testbench，覆盖正常流程、边界条件与协议违例注入。

运行：`run_sim.bat`（或 `vivado -mode batch -source run_sim.tcl`）

## 综合与实现

`build_bd.tcl` 构建完整 Block Design（PS7 + AXI DMA + 加速核心 + AXI GPIO 调试通路），`ps7_params.tcl` 定义 PS 侧配置。`ooc_gemv.tcl` / `ooc_gdn.tcl` 用于模块级 out-of-context 综合。

## 资源占用

`util_*.rpt` 为各模块 out-of-context 综合资源报告。详见 `01-系统概述/HW_UTIL_ANALYSIS.md`。

## 设计说明

`REPORT_gemv_core.md`（接口协议、流水结构、累加器位宽推导）、`CORE_SIM_RESULTS.md`（仿真结果与位精确性）.
