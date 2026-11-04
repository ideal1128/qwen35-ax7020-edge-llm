# run_sim.tcl -- batch compile + run of the RTL testbenches with xsim
#
# Usage (from any directory):
#   vivado.bat -mode batch -source run_sim.tcl
#   vivado.bat -mode batch -source run_sim.tcl -tclargs tb_fwht64
#
# Each testbench is compiled and run in its own directory under
#   E:/ds_workplace/scratch/qwenwork/sim/<testbench>/
# so runs never share a xsim snapshot.  The script prints every PASS/FAIL line
# from the simulation log and a one-line summary per testbench.
#
# Vivado tools write their own <tool>.log files in the working directory, so the
# captured stdout is redirected to *_out.txt to avoid a name clash.

set VIV_BIN  "D:/AMD/2025.2/2025.2/Vivado/bin"
set ROOT     "E:/ds_workplace/scratch/qwenwork"
set RTL_DIR  "$ROOT/rtl"
set TB_DIR   "$RTL_DIR/tb"
set SIM_ROOT "$ROOT/sim"

# Keep the tools' scratch files off the C: drive.
set env(HOME)         "E:/ds_workplace/scratch/vivado_home"
set env(USERPROFILE)  $env(HOME)
set env(APPDATA)      "$env(HOME)/AppData"
set env(LOCALAPPDATA) "$env(HOME)/AppData/Local"
file mkdir $env(APPDATA)
file mkdir $env(LOCALAPPDATA)
file mkdir "$SIM_ROOT/tmp"
set env(TMP)  "$SIM_ROOT/tmp"
set env(TEMP) "$SIM_ROOT/tmp"

# RTL file list per testbench (relative to $RTL_DIR).
array set RTL_OF {
    tb_fwht64      {fwht64.v}
    tb_bfp8_pack16 {bfp8_pack16.v}
    tb_gdn_core    {gdn_core.v}
    tb_conv1d_silu {conv1d_silu.v}
    tb_conv_core   {conv1d_silu.v fwht64.v bfp8_pack16.v conv_core.v}
}

set ALL_TESTS [list tb_fwht64 tb_bfp8_pack16 tb_gdn_core tb_conv1d_silu tb_conv_core]

if {[llength $argv] > 0} {
    set TESTS $argv
} else {
    set TESTS $ALL_TESTS
}

proc run_test {name} {
    global VIV_BIN RTL_DIR TB_DIR SIM_ROOT RTL_OF

    set wd [file join $SIM_ROOT $name]
    file mkdir $wd

    set lines {}
    lappend lines "@echo off"
    lappend lines "cd /d [file nativename $wd]"
    set cmd "call \"[file nativename $VIV_BIN/xvlog.bat]\""
    foreach f $RTL_OF($name) {
        append cmd " \"[file nativename [file join $RTL_DIR $f]]\""
    }
    append cmd " \"[file nativename [file join $TB_DIR $name.v]]\""
    lappend lines "$cmd > xvlog_out.txt 2>&1"
    lappend lines "if errorlevel 1 exit /b 1"
    lappend lines "call \"[file nativename $VIV_BIN/xelab.bat]\" -timescale 1ns/1ps $name -s sim_$name > xelab_out.txt 2>&1"
    lappend lines "if errorlevel 1 exit /b 2"
    lappend lines "call \"[file nativename $VIV_BIN/xsim.bat]\" sim_$name -runall > xsim_out.txt 2>&1"
    lappend lines "exit /b %errorlevel%"

    set bat [file join $wd "run_$name.bat"]
    set fh [open $bat w]
    puts $fh [join $lines "\n"]
    close $fh

    set rc [catch {exec cmd.exe /c [file nativename $bat]} msg]

    puts ""
    puts "================================================================"
    puts " $name"
    puts "================================================================"
    if {$rc != 0} {
        puts "COMPILE/ELAB/RUN FAILED: $msg"
        foreach f {xvlog_out.txt xelab_out.txt xsim_out.txt} {
            set p [file join $wd $f]
            if {[file exists $p]} {
                set h [open $p r]
                set txt [read $h]
                close $h
                if {[string length [string trim $txt]] > 0} {
                    puts "--- $f ---"
                    puts $txt
                }
            }
        }
        return 0
    }

    set p [file join $wd "xsim_out.txt"]
    if {[file exists $p]} {
        set h [open $p r]
        set txt [read $h]
        close $h
        foreach line [split $txt "\n"] {
            if {[regexp {PASS|FAIL|INFO: (max|channels|dump|latency)} $line]} {
                puts "  [string trim $line]"
            }
        }
        if {[regexp {RESULT: \S+ PASS} $txt]} {
            return 1
        }
    }
    return 0
}

set npass 0
set nfail 0
set summary {}
foreach t $TESTS {
    if {[info exists RTL_OF($t)]} {
        set ok [run_test $t]
        if {$ok} {
            incr npass
            lappend summary [format "  %-16s PASS" $t]
        } else {
            incr nfail
            lappend summary [format "  %-16s FAIL" $t]
        }
    } else {
        puts "unknown testbench: $t"
    }
}

puts ""
puts "================ SUMMARY ================"
foreach s $summary { puts $s }
puts "  pass=$npass fail=$nfail"
puts "========================================="
