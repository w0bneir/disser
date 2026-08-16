<#
.SYNOPSIS
  Проверяет, достаточно ли свободной VRAM для ручного Stable Audio guidance.

.DESCRIPTION
  Ничего не загружает в GPU и ничего не меняет. Код завершения 0 означает,
  что общий и свободный объём VRAM достаточны для экспериментального backend-а.
#>
[CmdletBinding()]
param(
    [int]$MinimumTotalMiB = 12000,
    [int]$MinimumFreeMiB = 10000
)

$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($null -eq $nvidiaSmi) {
    Write-Error 'nvidia-smi не найден: установите NVIDIA driver и перезапустите PowerShell.'
    exit 3
}

$line = & $nvidiaSmi.Source --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader,nounits 2>$null | Select-Object -First 1
if ([string]::IsNullOrWhiteSpace($line)) {
    Write-Error 'NVIDIA GPU не ответила на запрос nvidia-smi.'
    exit 3
}

$parts = $line -split ',' | ForEach-Object { $_.Trim() }
if ($parts.Count -lt 4) {
    Write-Error "Не удалось разобрать вывод nvidia-smi: $line"
    exit 3
}

$totalMiB = [int]$parts[2]
$freeMiB = [int]$parts[3]
Write-Output "GPU: $($parts[0])"
Write-Output "Driver: $($parts[1])"
Write-Output "VRAM: всего $totalMiB MiB; свободно $freeMiB MiB"

if ($totalMiB -lt $MinimumTotalMiB -or $freeMiB -lt $MinimumFreeMiB) {
    Write-Error "Запуск заблокирован: требуется >= $MinimumTotalMiB MiB всего и >= $MinimumFreeMiB MiB свободно."
    exit 2
}

Write-Output 'GPU preflight: OK'
