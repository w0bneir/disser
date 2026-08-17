<#
.SYNOPSIS
  Создаёт неизменяемый технический паспорт выполненного эксперимента.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ResultsDirectory
)

$resolved = Resolve-Path $ResultsDirectory -ErrorAction Stop
$output = Join-Path $resolved 'run_context.txt'
$python = Get-Command python -ErrorAction Stop
$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
$pythonVersion = & $python.Source --version 2>&1
$pipCheck = & $python.Source -m pip check 2>&1
$torchProbe = & $python.Source -c 'import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_properties(0).total_memory)' 2>&1

@(
    "captured_at_utc=$([DateTime]::UtcNow.ToString('o'))"
    "git_commit=$(git rev-parse HEAD)"
    "git_branch=$(git branch --show-current)"
    "python=$pythonVersion"
    '--- pip check ---'
    "$pipCheck"
    '--- torch ---'
    "$torchProbe"
    '--- nvidia-smi ---'
    $(if ($nvidiaSmi) { & $nvidiaSmi.Source } else { 'nvidia-smi not found' })
) | Set-Content -Encoding utf8 $output

Write-Output "Контекст запуска сохранён: $output"
