# Reboot the AX7020 and interrupt u-boot autoboot, then run a command script.
# Usage: uboot.ps1 -Commands "cmd1","cmd2" [-Port COM4] [-SkipReboot]
param(
    [string]$Port = "COM4",
    [int]$Baud = 115200,
    [string[]]$Commands = @(),
    [switch]$SkipReboot,
    [int]$CatchSeconds = 30
)

$ErrorActionPreference = "Stop"

# powershell -File joins array args with commas, so accept "cmd1;;cmd2;;cmd3" too
$expanded = @()
foreach ($c in $Commands) { $expanded += ($c -split ';;') }
$Commands = $expanded

$sp = New-Object System.IO.Ports.SerialPort $Port, $Baud, "None", 8, "One"
$sp.ReadTimeout = 200
$sp.WriteTimeout = 1000
$sp.DtrEnable = $true
$sp.Open()
Start-Sleep -Milliseconds 200

function Drain([int]$ms) {
    $sb = New-Object System.Text.StringBuilder
    $end = (Get-Date).AddMilliseconds($ms)
    while ((Get-Date) -lt $end) {
        try { $sb.Append($sp.ReadExisting()) | Out-Null } catch {}
        Start-Sleep -Milliseconds 50
    }
    return $sb.ToString()
}

if (-not $SkipReboot) {
    Write-Host "=== sending reboot ==="
    $sp.Write("reboot`r")
    Start-Sleep -Milliseconds 1500
}

Write-Host "=== waiting for u-boot, spamming keys for $CatchSeconds s ==="
$buf = New-Object System.Text.StringBuilder
$end = (Get-Date).AddSeconds($CatchSeconds)
$caught = $false
while ((Get-Date) -lt $end) {
    $sp.Write(" ")
    Start-Sleep -Milliseconds 120
    $chunk = ""
    try { $chunk = $sp.ReadExisting() } catch {}
    if ($chunk) {
        $buf.Append($chunk) | Out-Null
        if ($chunk -match "Zynq>|U-Boot>|=>\s*$|Hit any key") { }
        if ($chunk -match "Zynq>|U-Boot>") { $caught = $true }
    }
    if ($caught -and ($buf.ToString() -match "Zynq>\s*$|U-Boot>\s*$")) { break }
}
Write-Host $buf.ToString()

if ($Commands.Count -gt 0) {
    Start-Sleep -Milliseconds 300
    foreach ($c in $Commands) {
        Write-Host "=== u-boot cmd: $c ==="
        $sp.Write("$c`r")
        Start-Sleep -Milliseconds 700
        $out = Drain 1200
        Write-Host $out
    }
}
$sp.Close()
Write-Host "=== DONE (caught=$caught) ==="
