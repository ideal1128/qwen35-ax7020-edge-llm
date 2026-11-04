@echo off
REM ===========================================================================
REM  START THE DEMO  --  double-click this file
REM
REM  Opens an interactive serial console to the AX7020 on COM4 and starts the
REM  model automatically.  No PuTTY or other software needed.
REM
REM  Keys:  Enter = send line, Ctrl-C = interrupt the board,
REM         Ctrl-X = close this window.
REM
REM  If you see "CANNOT OPEN COM4", something else is holding the port.
REM  Close any other serial program (or another copy of this window) first.
REM ===========================================================================

setlocal
cd /d "%~dp0"

echo.
echo  Starting AX7020 serial console on COM4 and loading the model...
echo  (the first question takes 15-30 s of one-off warm-up -- be patient)
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0demo_console.ps1" -Port COM4 -Baud 115200 -Auto

echo.
echo  Console closed.
pause
