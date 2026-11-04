@echo off
setlocal
set VIVADO=D:\AMD\2025.2\2025.2\Vivado
set VHOME=E:\ds_workplace\scratch\vivado_home
set HOME=%VHOME%
set USERPROFILE=%VHOME%
set APPDATA=%VHOME%\AppData
set LOCALAPPDATA=%VHOME%\AppData\Local
set TMP=E:\ds_workplace\scratch\qwenwork\sim\tmp
set TEMP=%TMP%
set RTLDIR=E:\ds_workplace\scratch\qwenwork
set PATH=%VIVADO%\bin;%PATH%
mkdir "%RTLDIR%\sim" 2>nul
mkdir "%TMP%" 2>nul
cd /d "%RTLDIR%\sim"
call "%VIVADO%\bin\xvlog.bat" "%RTLDIR%\rtl\gemv_core.v" "%RTLDIR%\tb\tb_gemv_core.v" > r1.txt 2>&1
echo XVLOG_RC=%errorlevel%
call "%VIVADO%\bin\xelab.bat" tb_gemv_core -s tbsim > r2.txt 2>&1
echo XELAB_RC=%errorlevel%
call "%VIVADO%\bin\xsim.bat" tbsim -runall > r3.txt 2>&1
echo XSIM_RC=%errorlevel%
type r3.txt