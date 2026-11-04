# run_dbg.tcl -- run the private diagnostic copy of tb_gdn_core
set VIV_BIN  "D:/AMD/2025.2/2025.2/Vivado/bin"
set ROOT     "E:/ds_workplace/scratch/qwenwork"
set RTL_DIR  "$ROOT/rtl"
set DBG_DIR  "$ROOT/gdn_dbg"
set wd       "$DBG_DIR/work"

set env(HOME)         "E:/ds_workplace/scratch/vivado_home"
set env(USERPROFILE)  $env(HOME)
set env(APPDATA)      "$env(HOME)/AppData"
set env(LOCALAPPDATA) "$env(HOME)/AppData/Local"
file mkdir $env(APPDATA)
file mkdir $env(LOCALAPPDATA)
set env(TMP)  "$DBG_DIR/tmp"
set env(TEMP) "$DBG_DIR/tmp"
file mkdir $env(TMP)
file mkdir $wd

set lines {}
lappend lines "@echo off"
lappend lines "cd /d [file nativename $wd]"
set cmd "call \"[file nativename $VIV_BIN/xvlog.bat]\" \"[file nativename [file join $RTL_DIR gdn_core.v]]\" \"[file nativename [file join $DBG_DIR dbg_gdn.v]]\""
lappend lines "$cmd > xvlog_out.txt 2>&1"
lappend lines "if errorlevel 1 exit /b 1"
lappend lines "call \"[file nativename $VIV_BIN/xelab.bat]\" -timescale 1ns/1ps dbg_gdn -s sim_dbg_gdn > xelab_out.txt 2>&1"
lappend lines "if errorlevel 1 exit /b 2"
lappend lines "call \"[file nativename $VIV_BIN/xsim.bat]\" sim_dbg_gdn -runall > xsim_out.txt 2>&1"
lappend lines "exit /b %errorlevel%"

set bat [file join $wd "run_dbg.bat"]
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
            if {[regexp {DBG|PASS|FAIL|RESULT|ERROR} $line]} { puts "  [string trim $line]" }
        }
    }
}
puts "=== DBG DONE ==="
exit
