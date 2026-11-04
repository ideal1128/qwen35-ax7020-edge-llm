set here [file normalize [file dirname [info script]]]
foreach f {conv1d_silu.v fwht64.v bfp8_pack16.v conv_core.v gdn_core.v gemv_core.v} {
    if {[file exists [file join $here $f]]} {
        read_verilog [file join $here $f]
        puts "READ $f"
    } else {
        puts "MISSING $f"
    }
}
foreach top {conv1d_silu fwht64 bfp8_pack16} {
}
synth_design -rtl -top conv1d_silu -part xc7z020clg400-2
puts "CONV1D_OK"
synth_design -rtl -top fwht64 -part xc7z020clg400-2
puts "FWHT_OK"
synth_design -rtl -top bfp8_pack16 -part xc7z020clg400-2
puts "PACK_OK"
synth_design -rtl -top conv_core -part xc7z020clg400-2
puts "CONVCORE_OK"
synth_design -rtl -top gdn_core -part xc7z020clg400-2
puts "GDNCORE_OK"
exit
