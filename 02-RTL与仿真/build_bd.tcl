# build_bd.tcl -- PS7 + AXI DMA + gemv_core -> bitstream
#
# PS7 参数从交付的 system_dual.xsa (system.hwh) 里抽出来（ps7_params.tcl），
# 只覆盖三处：打开 HP1、打开 CLK0/RST0、FCLK0 设 200 MHz。
# 注意：位流不携带 PS 配置，DDR/MIO 由 BOOT.BIN 里的 ps7_init 决定，
#       这里填的参数只影响 PL 侧接口结构。
#
# 数据通路：
#   DDR --HP0--> AXI3/AXI4 转换 --128->64 转换 -- DMA MM2S --128-bit 流--> gemv_core
#   gemv_core --32-bit 流--> DMA S2MM --AXI4/AXI3 转换--> DDR (HP1)
#
# 用法: vivado -mode batch -source build_bd.tcl

set part xc7z020clg400-2
# NOTE: 路径里不能有括号。Vivado 拒绝带 "(" 的文件名，而 [file normalize] 会把
# scratch\qwenwork 这个 junction 解回真实路径 "...\qwen35_zynq7020(2)\..."，所以写死。
set here E:/ds_workplace/scratch/qwenwork/rtl
set root E:/ds_workplace/scratch/qwenwork
source [file join $here ps7_params.tcl]

create_project gemv_prj [file join $root synth bdprj] -part $part -force
set_property source_mgmt_mode All [current_project]
add_files -norecurse [file join $here gemv_core.v]
update_compile_order -fileset sources_1

create_bd_design gemv_sys
current_bd_design gemv_sys

# ---------------------------------------------------------------- PS7
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7 ps7
set valid [list_property [get_bd_cells ps7]]
set dict {}
set skipped 0
foreach {k v} $ps7_params {
    if {[lsearch -exact $valid CONFIG.$k] >= 0} {
        lappend dict CONFIG.$k $v
    } else {
        incr skipped
    }
}
puts "PS7: applying [expr {[llength $dict]/2}] params, skipped $skipped unknown"
set_property -dict $dict [get_bd_cells ps7]

# 覆盖：打开 HP1，FCLK0 拉到 200 MHz
foreach kv {
    {PCW_USE_M_AXI_GP0 1}
    {PCW_USE_S_AXI_HP0 1}
    {PCW_USE_S_AXI_HP1 1}
    {PCW_EN_CLK0_PORT 1}
    {PCW_EN_RST0_PORT 1}
    {PCW_FPGA0_PERIPHERAL_FREQMHZ 200}
} {
    lassign $kv k v
    if {[catch {set_property CONFIG.$k $v [get_bd_cells ps7]} err]} {
        puts "PS7-OVERRIDE-FAIL $k=$v : $err"
    }
}

# ---------------------------------------------------------------- DMA
# 流和内存同宽（32）时 DMA 不会生成 S2MM realigner；那个 realigner 里
# sig_btt_cntr_dup -> sig_btt_eq_0 的 6 级进位链是 200 MHz 下唯一的违例源。
# 输出量很小（每个 FFN 229 KB），内存侧窄一半没有实际代价。
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma dma
set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_sg_include_stscntrl_strm {0} \
    CONFIG.c_include_mm2s {1} \
    CONFIG.c_include_s2mm {1} \
    CONFIG.c_m_axi_mm2s_data_width {128} \
    CONFIG.c_m_axis_mm2s_tdata_width {128} \
    CONFIG.c_m_axi_s2mm_data_width {32} \
    CONFIG.c_s_axis_s2mm_tdata_width {32} \
    CONFIG.c_addr_width {32} \
    CONFIG.c_include_mm2s_dre {0} \
    CONFIG.c_include_s2mm_dre {0} \
    CONFIG.c_sg_length_width {26} \
    CONFIG.c_mm2s_burst_size {256} \
    CONFIG.c_s2mm_burst_size {256} \
    CONFIG.c_include_mm2s_sf {0} \
    CONFIG.c_include_s2mm_sf {0} \
] [get_bd_cells dma]

# ---------------------------------------------------------------- core
create_bd_cell -type module -reference gemv_core core

# ---------------------------------------------------------------- SmartConnect
# SmartConnect does the width conversion (128 -> 64) and the protocol conversion
# (AXI4 -> AXI3) internally.  An explicit dwc + axi_protocol_converter +
# register-slice chain was tried first and stalled on any MM2S transfer larger
# than 8 KB, so the interconnect is left to the IP that is built for it.
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect smc
set_property -dict [list CONFIG.NUM_SI {3} CONFIG.NUM_MI {4}] [get_bd_cells smc]

# ---------------------------------------------------------------- reset
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset rst

# ---------------------------------------------------------------- debug gpio
# 通道1 全输入: 读核心状态; 通道2 全输出: 软复位核心
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio dbg
set_property -dict [list \
    CONFIG.C_GPIO_WIDTH {32} \
    CONFIG.C_GPIO2_WIDTH {1} \
    CONFIG.C_IS_DUAL {1} \
    CONFIG.C_ALL_INPUTS {1} \
    CONFIG.C_ALL_OUTPUTS_2 {1} \
] [get_bd_cells dbg]
# ---------------------------------------------------------------- interfaces
connect_bd_intf_net [get_bd_intf_pins ps7/M_AXI_GP0]    [get_bd_intf_pins smc/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins smc/M00_AXI]      [get_bd_intf_pins dma/S_AXI_LITE]

connect_bd_intf_net [get_bd_intf_pins dma/M_AXI_MM2S]   [get_bd_intf_pins smc/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins smc/M01_AXI]      [get_bd_intf_pins ps7/S_AXI_HP0]

connect_bd_intf_net [get_bd_intf_pins dma/M_AXI_S2MM]   [get_bd_intf_pins smc/S02_AXI]
connect_bd_intf_net [get_bd_intf_pins smc/M02_AXI]      [get_bd_intf_pins ps7/S_AXI_HP1]
connect_bd_intf_net [get_bd_intf_pins smc/M03_AXI]      [get_bd_intf_pins dbg/S_AXI]

connect_bd_intf_net [get_bd_intf_pins dma/M_AXIS_MM2S]  [get_bd_intf_pins core/s_axis]
connect_bd_intf_net [get_bd_intf_pins core/m_axis]      [get_bd_intf_pins dma/S_AXIS_S2MM]

connect_bd_net [get_bd_pins core/dbg]       [get_bd_pins dbg/gpio_io_i]
connect_bd_net [get_bd_pins dbg/gpio2_io_o] [get_bd_pins core/dbg_rst]

# ---------------------------------------------------------------- clocks / resets
# 不同 IP 的时钟/复位脚名字不一样，逐个过滤，缺的只打日志不报错
set clk_pins {}
foreach p {ps7/FCLK_CLK0 smc/aclk dma/m_axi_mm2s_aclk dma/m_axi_s2mm_aclk \
           dma/s_axi_lite_aclk core/aclk rst/slowest_sync_clk \
           ps7/M_AXI_GP0_ACLK ps7/S_AXI_HP0_ACLK ps7/S_AXI_HP1_ACLK dbg/s_axi_aclk} {
    set o [get_bd_pins -quiet $p]
    if {[llength $o]} { lappend clk_pins $o } else { puts "NOPIN $p" }
}
connect_bd_net {*}$clk_pins

connect_bd_net [get_bd_pins ps7/FCLK_RESET0_N] [get_bd_pins rst/ext_reset_in]

set rst_pins {}
foreach p {rst/peripheral_aresetn smc/aresetn dma/axi_resetn core/aresetn dbg/s_axi_aresetn} {
    set o [get_bd_pins -quiet $p]
    if {[llength $o]} { lappend rst_pins $o } else { puts "NOPIN $p" }
}
connect_bd_net {*}$rst_pins

# ---------------------------------------------------------------- external
make_bd_intf_pins_external [get_bd_intf_pins ps7/DDR]
make_bd_intf_pins_external [get_bd_intf_pins ps7/FIXED_IO]

assign_bd_address

validate_bd_design
save_bd_design
puts "=== BD VALIDATED ==="

# ---------------------------------------------------------------- build
generate_target all [get_files gemv_sys.bd]
set wf [make_wrapper -files [get_files gemv_sys.bd] -top]
add_files -norecurse $wf
set_property top [file rootname [file tail [lindex $wf 0]]] [current_fileset]
update_compile_order -fileset sources_1

launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    puts "=== IMPL FAILED ==="
    exit 1
}
open_run impl_1
report_utilization -file [file join $root synth util_bd.rpt]
report_timing_summary -file [file join $root synth timing_bd.rpt] -max_paths 8
write_bitstream -force [file join $root synth gemv_sys.bit]
puts "=== BITSTREAM OK ==="
exit 0
