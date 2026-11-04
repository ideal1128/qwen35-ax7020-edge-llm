param(
    [Parameter(Mandatory=$true)][string]$Remote,
    [Parameter(Mandatory=$true)][string]$Out,
    [int]$TimeoutSec = 240
)
# Download a file from the board over the serial console (base64).
$ErrorActionPreference = 'Continue'
$sp = New-Object System.IO.Ports.SerialPort 'COM4',115200,None,8,one
$sp.ReadTimeout = 500
$sp.WriteTimeout = 5000

try {
    $sp.Open()
    Start-Sleep -Milliseconds 200
    $sp.Write("stty -echo`r`n")
    Start-Sleep -Milliseconds 250
    $sp.DiscardInBuffer()
    $sp.Write("base64 '" + $Remote + "' ; echo __RC=`$?__`r`n")

    $sb = New-Object System.Text.StringBuilder
    $end = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $end) {
        try { $chunk = $sp.ReadExisting() } catch { $chunk = "" }
        if ($chunk) { [void]$sb.Append($chunk) } else { Start-Sleep -Milliseconds 40 }
        if ($sb.ToString().Contains("__RC=")) {
            # 再多读一点，确保尾部数据到齐
            Start-Sleep -Milliseconds 400
            try { $chunk2 = $sp.ReadExisting(); if ($chunk2) { [void]$sb.Append($chunk2) } } catch {}
            break
        }
    }
    $text = $sb.ToString()
    $lines = $text -split "`r?`n"
    $b64 = ($lines | Where-Object { $_ -match '^[A-Za-z0-9+/=]+$' -and $_.Length -ge 8 }) -join ''
    if (-not $b64) { Write-Output "GET_ERROR: no base64 payload captured"; exit 1 }
    $bytes = [Convert]::FromBase64String($b64)
    [System.IO.File]::WriteAllBytes($Out, $bytes)
    Write-Output ("GOT " + $bytes.Length + " bytes -> " + $Out)
    Write-Output ("LOCAL_MD5=" + (Get-FileHash -Path $Out -Algorithm MD5).Hash.ToLower())
} catch {
    Write-Output ("SERIAL_ERROR: " + $_.Exception.Message)
} finally {
    if ($sp.IsOpen) { $sp.Close() }
    $sp.Dispose()
}
