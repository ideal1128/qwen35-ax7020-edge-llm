# Log in on the board console after a reboot.
param([string]$Port = "COM4", [int]$Baud = 115200, [string]$Cmd = "")

$sp = New-Object System.IO.Ports.SerialPort $Port, $Baud, "None", 8, "One"
$sp.ReadTimeout = 200
$sp.Open()
Start-Sleep -Milliseconds 300

function ReadFor([int]$ms) {
    $sb = New-Object System.Text.StringBuilder
    $end = (Get-Date).AddMilliseconds($ms)
    while ((Get-Date) -lt $end) {
        try { $sb.Append($sp.ReadExisting()) | Out-Null } catch {}
        Start-Sleep -Milliseconds 60
    }
    $sb.ToString()
}

$sp.Write("`r")
$out = ReadFor 1500
if ($out -match "login:") {
    $sp.Write("root`r"); Start-Sleep -Milliseconds 800
    $o2 = ReadFor 1200
    if ($o2 -match "Password:") { $sp.Write("root`r"); Start-Sleep -Milliseconds 1500; $o2 = ReadFor 1500 }
    $out += $o2
}
if ($Cmd -ne "") {
    $sp.Write("stty -echo`r"); Start-Sleep -Milliseconds 300
    try { $sp.DiscardInBuffer() } catch {}
    $sp.Write("$Cmd`r")
    $out += ReadFor 4000
}
$sp.Close()
Write-Host $out
