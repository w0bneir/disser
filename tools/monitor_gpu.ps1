<#
.SYNOPSIS
  Сохраняет телеметрию NVIDIA GPU в CSV до нажатия Ctrl+C.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$OutputPath,
    [ValidateRange(1, 60)]
    [int]$IntervalSeconds = 1
)

$nvidiaSmi = Get-Command nvidia-smi -ErrorAction Stop
$parent = Split-Path -Parent $OutputPath
if ($parent) {
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
}

'timestamp,name,memory.used.mib,memory.free.mib,utilization.gpu.percent,temperature.gpu.c' | Set-Content -Encoding utf8 $OutputPath
Write-Output "GPU monitoring is writing to $OutputPath. Stop it with Ctrl+C."
while ($true) {
    $row = & $nvidiaSmi.Source --query-gpu=timestamp,name,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits 2>$null | Select-Object -First 1
    if ($row) {
        Add-Content -Encoding utf8 -Path $OutputPath -Value $row
    }
    Start-Sleep -Seconds $IntervalSeconds
}
