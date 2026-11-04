# synth.tcl -- out-of-context synthesis check for gemv_core
set top  gemv_core
set part xc7z020clg400-2

read_verilog E:/ds_workplace/scratch/qwenwork/rtl/gemv_core.v
synth_design -top $top -part $part

create_clock -name aclk -period 5.000 [get_ports aclk]
report_utilization      -file util.rpt
report_timing_summary   -file timing.rpt -max_paths 8 -delay_type max
report_clock_utilization -file clocks.rpt

puts "=== SYNTH DONE ==="
exit
