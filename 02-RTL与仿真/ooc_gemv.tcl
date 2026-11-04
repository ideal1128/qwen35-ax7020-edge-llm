# ooc_gemv.tcl -- out-of-context synth + place + route of gemv_core at 6.0 ns (166.67 MHz)
set part xc7z020clg400-2
set rtl  E:/ds_workplace/scratch/qwenwork/rtl
set out  E:/ds_workplace/scratch/qwenwork/synth/ooc_gemv
set top  gemv_core
if {[info exists ::env(OOC_TOP)]} { set top $::env(OOC_TOP) }
if {[info exists ::env(OOC_OUT)]} { set out $::env(OOC_OUT) }
if {[info exists ::env(OOC_RTL)]} { set rtl $::env(OOC_RTL) }
if {[info exists ::env(OOC_FILE)]} { set file $::env(OOC_FILE) } else { set file gemv_core.v }

file mkdir $out
read_verilog [file join $rtl $file]
synth_design -top $top -part $part -mode out_of_context
create_clock -name aclk -period 6.000 [get_ports aclk]
opt_design
place_design
route_design

report_utilization    -file [file join $out util.rpt]
report_timing_summary -file [file join $out timing.rpt] -max_paths 40 -delay_type max
report_timing_summary -file [file join $out timing_min.rpt] -max_paths 20 -delay_type min
report_design_analysis -file [file join $out analysis.rpt] -congestion -logic_level_distribution
puts "=== OOC DONE ==="
exit
