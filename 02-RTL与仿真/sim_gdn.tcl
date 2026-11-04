# sim_gdn.tcl -- private gdn_core regression (gdn-pipe agent)
# Runs tb_gdn_core against rtl/gdn_core.v in its own directory so it never
# collides with the shared tb/run_sim.tcl work areas.
set VIV_BIN  "D:/AMD/2025.2/2025.2/Vivado/bin"
set ROOT     "E:/ds_workplace/scratch/qwenwork"
set RTL_DIR  "$ROOT/rtl"
set TB_DIR   "$RTL_DIR/tb"
set SIM_ROOT "$ROOT/sim"
set wd       "$SIM_ROOT/gdn_pipe_tb"

set env(HOME)         "E:/ds_workplace/scratch/vivado_home"
set env(USERPROFILE)  $env(HOME)
set env(APPDATA)      "$env(HOME)/AppData"
set env(LOCALAPPDATA) "$env(HOME)/AppData/Local"
file mkdir $env(APPDATA)
file mkdir $env(LOCALAPPDATA)
file mkdir "$SIM_ROOT/tmp"
set env(TMP)  "$SIM_ROOT/tmp"
set env(TEMP) "$SIM_ROOT/tmp"

file mkdir $wd
set lines {}
lappend lines "@echo off"
lappend lines "cd /d [file nativename $wd]"
set cmd "call \"[file nativename $VIV_BIN/xvlog.bat]\" \"[file nativename [file join $RTL_DIR gdn_core.v]]\" \"[file nativename [file join $TB_DIR tb_gdn_core.v]]\""
lappend lines "$cmd > xvlog_out.txt 2>&1"
lappend lines "if errorlevel 1 exit /b 1"
lappend lines "call \"[file nativename $VIV_BIN/xelab.bat]\" -timescale 1ns/1ps tb_gdn_core -s sim_tb_gdn_core > xelab_out.txt 2>&1"
lappend lines "if errorlevel 1 exit /b 2"
lappend lines "call \"[file nativename $VIV_BIN/xsim.bat]\" sim_tb_gdn_core -runall > xsim_out.txt 2>&1"
lappend lines "exit /b %errorlevel%"

set bat [file join $wd "run_gdn_pipe.bat"]
set fh [open $bat w]
puts $fh [join $lines "\n"]
close $fh

set rc [catch {exec cmd.exe /c [file nativename $bat]} msg]
puts "=== exec rc=$rc msg=$msg ==="
foreach f {xvlog_out.txt xelab_out.txt xsim_out.txt} {
    set p [file join $wd $f]
    if {[file exists $p]} {
        set h [open $p r]
        set txt [read $h]
        close $h
        puts "--- $f ---"
        foreach line [split $txt "\n"] {
            if {[regexp {PASS|FAIL|RESULT|ERROR|Error|\\$finish|max |dump } $line]} {
                puts "  [string trim $line]"
            }
        }
    }
}
puts "=== SIM_GDN DONE ==="
exit
