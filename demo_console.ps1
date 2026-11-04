# demo_console.ps1 -- interactive serial console for the AX7020 demo.
#
# You do NOT need PuTTY or Tera Term.  This script opens COM4 directly and
# gives you a terminal window you can type into.
#
# Usage (from PowerShell):
#     powershell -NoProfile -ExecutionPolicy Bypass -File demo_console.ps1
#
#     -Port COM4        serial port (default COM4)
#     -Baud 115200      baud rate (default 115200)
#     -Auto             on connect, start the model and wait for its prompt
#
# Keys:
#     Enter             send the line
#     Backspace         erase
#     Ctrl-C            send a real ^C to the BOARD (stops a long answer)
#                       and stay connected -- it does not close this window
#     Ctrl-X            close this window (the board keeps running)
#
# Why Ctrl-C no longer exits: previously it only broke out of the local
# read loop and never wrote anything to the port, so the board's Python kept
# generating for minutes with nobody watching, and the next answer looked
# truncated.  Now it forwards byte 0x03 so the board's SIGINT handler fires.
#
# ASCII only.  No external dependencies.

param(
    [string]$Port = "COM4",
    [int]$Baud = 115200,
    [switch]$Auto
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "  AX7020 serial console" -ForegroundColor Cyan
Write-Host "  port $Port  $Baud 8N1" -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan

# --- open the port -------------------------------------------------------
$sp = New-Object System.IO.Ports.SerialPort
$sp.PortName = $Port
$sp.BaudRate = $Baud
$sp.Parity = [System.IO.Ports.Parity]::None
$sp.DataBits = 8
$sp.StopBits = [System.IO.Ports.StopBits]::One
$sp.Handshake = [System.IO.Ports.Handshake]::None
$sp.ReadTimeout = 500
$sp.WriteTimeout = 2000
$sp.DtrEnable = $false
$sp.RtsEnable = $false
$sp.Encoding = [System.Text.Encoding]::UTF8

try {
    $sp.Open()
} catch {
    Write-Host ""
    Write-Host "  CANNOT OPEN $Port" -ForegroundColor Red
    Write-Host "  $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    Write-Host "  Common causes:" -ForegroundColor Yellow
    Write-Host "    - another program already holds the port (close it first)"
    Write-Host "    - wrong COM number (check Device Manager)"
    Write-Host "    - USB cable unplugged"
    Write-Host ""
    Read-Host "  press Enter to close"
    exit 1
}

Write-Host ""
Write-Host "  connected. clearing any leftover input on the board..." -ForegroundColor Green
Write-Host "  Ctrl-C = interrupt the board.  Ctrl-X = close this window." -ForegroundColor Green
Write-Host ""

# The board's shell may still hold a half-typed command from whatever ran
# before (a line that was never terminated).  Send Ctrl-C and an empty line
# so the first thing we type is not appended to that leftover.
try {
    $sp.Write([byte[]]@(3), 0, 1)          # Ctrl-C
    Start-Sleep -Milliseconds 150
    $nl = [System.Text.Encoding]::ASCII.GetBytes("`r")
    $sp.Write($nl, 0, $nl.Length)
    Start-Sleep -Milliseconds 250
    $n = $sp.BytesToRead
    if ($n -gt 0) {
        $buf = New-Object byte[] $n
        $read = $sp.Read($buf, 0, $n)
        if ($read -gt 0) {
            [Console]::Write([System.Text.Encoding]::UTF8.GetString($buf, 0, $read))
        }
    }
} catch { }

# --- helpers -------------------------------------------------------------
function Send-Line([string]$text) {
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($text + "`r")
    $sp.Write($bytes, 0, $bytes.Length)
}

function Send-Bytes([byte[]]$bytes) {
    $sp.Write($bytes, 0, $bytes.Length)
}

function Pump-Output {
    # move whatever the board sent into the console
    try {
        $n = $sp.BytesToRead
        if ($n -gt 0) {
            $buf = New-Object byte[] $n
            $read = $sp.Read($buf, 0, $n)
            if ($read -gt 0) {
                $s = [System.Text.Encoding]::UTF8.GetString($buf, 0, $read)
                [Console]::Write($s)
            }
        }
    } catch {
        # transient read errors are normal on a busy port
    }
}

function Wait-For([string]$marker, [int]$timeoutSec) {
    # pump output until $marker appears; returns $false on timeout or Ctrl-X
    $acc = ""
    $deadline = (Get-Date).AddSeconds($timeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            $n = $sp.BytesToRead
            if ($n -gt 0) {
                $buf = New-Object byte[] $n
                $read = $sp.Read($buf, 0, $n)
                if ($read -gt 0) {
                    $s = [System.Text.Encoding]::UTF8.GetString($buf, 0, $read)
                    [Console]::Write($s)
                    $acc += $s
                    if ($acc.Length -gt 4096) { $acc = $acc.Substring($acc.Length - 4096) }
                    if ($acc.Contains($marker)) { return $true }
                }
            }
        } catch { }
        if ([Console]::KeyAvailable) {
            $k = [Console]::ReadKey($true)
            if ($k.Key -eq [ConsoleKey]::X -and ($k.Modifiers -band [ConsoleModifiers]::Control)) {
                return $false
            }
        }
        Start-Sleep -Milliseconds 40
    }
    return $false
}

# --- optional: start the model -------------------------------------------
$abort = $false
if ($Auto) {
    Start-Sleep -Milliseconds 400
    Pump-Output
    Write-Host ""
    Write-Host "  [auto] starting the model on the board..." -ForegroundColor Yellow
    Write-Host "  [auto] loading the engine takes a few seconds.  The FIRST question" -ForegroundColor Yellow
    Write-Host "  [auto] also pays a one-off 15-30 s weight warm-up, so do not worry" -ForegroundColor Yellow
    Write-Host "  [auto] if it looks idle.  Press Ctrl-X to cancel." -ForegroundColor Yellow
    Write-Host ""
    # No --max here.  The old preamble used "--max 2" to warm the page cache,
    # which printed a 2-token answer and looked exactly like truncation.
    Send-Line "cd /mnt/qwen && ./real_chat_v3"
    if (Wait-For "you>" 120) {
        Write-Host ""
        Write-Host "  [auto] ---------------------------------------------------------" -ForegroundColor Green
        Write-Host "  [auto] ready.  Type a question and press Enter.  Chinese is fine." -ForegroundColor Green
        Write-Host "  [auto] board commands:  /clear   /think on|off   /stats   /quit" -ForegroundColor Green
        Write-Host "  [auto] Ctrl-C interrupts a long answer; Ctrl-X closes this window." -ForegroundColor Green
        Write-Host "  [auto] ---------------------------------------------------------" -ForegroundColor Green
        Write-Host ""
    } else {
        Write-Host ""
        Write-Host "  [auto] did not see the board prompt within 120 s." -ForegroundColor Red
        Write-Host "  [auto] You can still type commands by hand below." -ForegroundColor Red
        Write-Host ""
    }
}

# --- interactive loop ----------------------------------------------------
$line = New-Object System.Text.StringBuilder

while ($true) {
    Pump-Output

    if ([Console]::KeyAvailable) {
        $k = [Console]::ReadKey($true)

        if ($k.Key -eq [ConsoleKey]::C -and ($k.Modifiers -band [ConsoleModifiers]::Control)) {
            # forward a REAL ^C to the board so a running generation stops,
            # then stay connected
            try { Send-Bytes ([byte[]]@(3)) } catch { }
            $null = $line.Clear()
            [Console]::Write("^C")
            continue
        }
        elseif ($k.Key -eq [ConsoleKey]::X -and ($k.Modifiers -band [ConsoleModifiers]::Control)) {
            Write-Host ""
            Write-Host "  closing console (the board keeps whatever it was doing)" -ForegroundColor Yellow
            break
        }
        elseif ($k.Key -eq [ConsoleKey]::Enter) {
            Write-Host ""
            Send-Line $line.ToString()
            $null = $line.Clear()
            Start-Sleep -Milliseconds 40
        }
        elseif ($k.Key -eq [ConsoleKey]::Backspace) {
            if ($line.Length -gt 0) {
                $null = $line.Remove($line.Length - 1, 1)
                [Console]::Write("`b `b")
            }
        }
        elseif ($k.KeyChar -ne [char]0) {
            $null = $line.Append($k.KeyChar)
            [Console]::Write($k.KeyChar)
        }
    }

    Start-Sleep -Milliseconds 15
}

$sp.Close()
$sp.Dispose()
