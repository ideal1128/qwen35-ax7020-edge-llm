@echo off
setlocal
set VIVADO=D:\AMD\2025.2\2025.2\Vivado
set VHOME=E:\ds_workplace\scratch\vivado_home
set HOME=%VHOME%
set USERPROFILE=%VHOME%
set APPDATA=%VHOME%\AppData
set LOCALAPPDATA=%VHOME%\AppData\Local
set TMP=E:\ds_workplace\scratch\qwenwork\synth\tmp
set TEMP=%TMP%
set RTLDIR=E:\ds_workplace\scratch\qwenwork
set PATH=%VIVADO%\bin;%PATH%
if not exist "%RTLDIR%\synth" mkdir "%RTLDIR%\synth"
if not exist "%TMP%" mkdir "%TMP%"
cd /d "%RTLDIR%\synth"
call "%VIVADO%\bin\vivado.bat" -mode batch -nojournal -nolog -source "%RTLDIR%\rtl\build_bd.tcl" > bd_out.txt 2>&1
echo VIVADO_RC=%errorlevel%