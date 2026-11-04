set here [file normalize [file dirname [info script]]]
read_verilog [file join $here gdn_core.v]
if {[file exists [file join $here conv_core.v]]} {
    read_verilog [file join $here conv_core.v]
    synth_design -rtl -top conv_core -part xc7z020clg400-2
    puts "CONV_RTL_OK"
}
synth_design -rtl -top gdn_core -part xc7z020clg400-2
puts "GDN_RTL_OK"
exit
