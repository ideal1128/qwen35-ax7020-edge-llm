param(
    [string]$Cmd = "",
    [string]$Put = "",
    [string]$Dest = "",
    [int]$TimeoutSec = 60,
    [switch]$Login
)
# Board console helper (COM4 @115200 8N1). Supports:
#   -Cmd  "<shell command>"      run a command, print output
#   -Put  <local file> -Dest <remote path>   upload via base64 heredoc
$ErrorActionPreference = 'Continue'
$sp = New-Object System.IO.Ports.SerialPort 'COM4',115200,None,8,one
$sp.ReadTimeout = 500
$sp.WriteTimeout = 5000

function Read-Until([string]$marker, [int]$sec) {
    $sb = New-Object System.Text.StringBuilder
    $end = (Get-Date).AddSeconds($sec)
    while ((Get-Date) -lt $end) {
        try { $chunk = $sp.ReadExisting() } catch { $chunk = "" }
        if ($chunk) { [void]$sb.Append($chunk) }
        else { Start-Sleep -Milliseconds 40 }
        if ($marker -and $sb.ToString().Contains($marker)) { break }
    }
    return $sb.ToString()
}

function Send-Cmd([string]$text, [int]$sec) {
    # CRLF-normalise then append the sentinel
    $norm = $text -replace "`r`n", "`n" -replace "`n", "`r`n"
    $sp.Write($norm + "`r`n")
    return (Read-Until "__RC=" $sec)
}

try {
    $sp.Open()
    Start-Sleep -Milliseconds 200
    $sp.DiscardInBuffer()

    if ($Login) {
        $sp.Write("`r`n")
        $t = Read-Until "login:" 6
        if ($t -match 'login:') {
            $sp.Write("root`r`n")
            Start-Sleep -Milliseconds 500
            $t2 = Read-Until "assword" 6
            if ($t2 -match 'assword') {
                $sp.Write("root`r`n")
                Start-Sleep -Milliseconds 800
                [void](Read-Until "#" 10)
            }
        }
        $sp.Write("stty -echo`r`n")
        Start-Sleep -Milliseconds 300
        [void](Read-Until "" 1)
        Write-Output "LOGIN_OK"
    }

    if ($Put) {
        if (-not (Test-Path $Put)) { Write-Output "PUT_ERROR: local file missing"; }
        else {
            $bytes = [System.IO.File]::ReadAllBytes((Resolve-Path $Put))
            $b64 = [Convert]::ToBase64String($bytes)
            $lines = @()
            for ($i = 0; $i -lt $b64.Length; $i += 200) {
                $lines += $b64.Substring($i, [Math]::Min(200, $b64.Length - $i))
            }
            $payload = "rm -f /tmp/_up.b64`ncat > /tmp/_up.b64 << 'XEOF'`n" +
                       ($lines -join "`n") +
                       "`nXEOF`ntr -d '\r' < /tmp/_up.b64 | base64 -d > '" + $Dest + "'`n" +
                       "md5sum '" + $Dest + "'`n" +
                       "echo __RC=$?__"
            $out = Send-Cmd $payload ($TimeoutSec + 60)
            $localMd5 = (Get-FileHash -Path $Put -Algorithm MD5).Hash.ToLower()
            Write-Output ("LOCAL_MD5=" + $localMd5)
            Write-Output $out
        }
    }

    if ($Cmd) {
        $sp.Write("stty -echo`r`n")
        Start-Sleep -Milliseconds 200
        $sp.DiscardInBuffer()
        $out = Send-Cmd ($Cmd + " ; echo __RC=`$?__") $TimeoutSec
        Write-Output $out
    }
} catch {
    Write-Output ("SERIAL_ERROR: " + $_.Exception.Message)
} finally {
    if ($sp.IsOpen) { $sp.Close() }
    $sp.Dispose()
}
