# synth_gdn.tcl -- fast out-of-context synthesis check for gdn_core at 6.0 ns
# Usage: vivado -mode batch -nojournal -nolog -source synth_gdn.tcl
set part xc7z020clg400-2
set rtl  E:/ds_workplace/scratch/qwenwork/rtl
set out  E:/ds_workplace/scratch/qwenwork/synth/synth_gdn
file mkdir $out

read_verilog [file join $rtl gdn_core.v]
synth_design -top gdn_core -part $part -mode out_of_context
create_clock -name aclk -period 6.000 [get_ports aclk]
report_utilization    -file [file join $out util.rpt]
report_timing_summary -file [file join $out timing.rpt] -max_paths 12 -delay_type max
report_timing_summary -file [file join $out timing_min.rpt] -max_paths 8 -delay_type min
puts "=== SYNTH_GDN DONE ==="
exit
