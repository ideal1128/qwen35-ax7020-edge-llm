set part xc7z020clg400-2
set here E:/ds_workplace/scratch/qwenwork/rtl
foreach top {conv1d_silu fwht64 bfp8_pack16 conv_core} {
    puts "########## $top ##########"
    set rc [catch {
        read_verilog [file join $here $top.v]
        if {$top eq "conv_core"} {
            read_verilog [file join $here conv1d_silu.v]
            read_verilog [file join $here fwht64.v]
            read_verilog [file join $here bfp8_pack16.v]
        }
        synth_design -top $top -part $part -flatten_hierarchy rebuilt
        create_clock -period 6.0 -name clk [get_ports aclk]
        set wns [get_property SLACK [get_timing_paths -max_paths 1 -delay_type max]]
        set whs [get_property SLACK [get_timing_paths -max_paths 1 -delay_type min]]
        set util [get_utilization]
        puts "RESULT $top WNS=$wns WHS=$whs"
        report_utilization -file [file join $here "util_$top.rpt"]
    } err]
    if {$rc} { puts "FAILED $top : $err" }
}
exit 0