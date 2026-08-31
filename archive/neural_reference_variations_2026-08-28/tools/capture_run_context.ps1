<#
.SYNOPSIS
  Создаёт неизменяемый технический паспорт выполненного эксперимента.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ResultsDirectory,
    [string]$PythonPath
)

$resolved = Resolve-Path $ResultsDirectory -ErrorAction Stop
$output = Join-Path $resolved 'run_context.txt'
if ($PythonPath) {
    $pythonExecutable = (Resolve-Path $PythonPath -ErrorAction Stop).Path
}
elseif ($env:CONDA_PREFIX -and (Test-Path (Join-Path $env:CONDA_PREFIX 'python.exe'))) {
    $pythonExecutable = Join-Path $env:CONDA_PREFIX 'python.exe'
}
else {
    $pythonExecutable = (Get-Command python -ErrorAction Stop).Source
}
$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
$pythonVersion = & $pythonExecutable --version 2>&1
$pipCheck = & $pythonExecutable -m pip check 2>&1
$torchProbe = & $pythonExecutable -c 'import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_properties(0).total_memory)' 2>&1
$gitStatus = @(git status --porcelain 2>&1)
$gitWorktreeClean = $LASTEXITCODE -eq 0 -and $gitStatus.Count -eq 0

@(
    "captured_at_utc=$([DateTime]::UtcNow.ToString('o'))"
    "git_commit=$(git rev-parse HEAD)"
    "git_branch=$(git branch --show-current)"
    "git_worktree_clean=$($gitWorktreeClean.ToString().ToLowerInvariant())"
    '--- git status --porcelain ---'
    $(if ($gitStatus.Count -eq 0) { 'clean' } else { $gitStatus })
    "python_executable=$pythonExecutable"
    "python=$pythonVersion"
    '--- pip check ---'
    "$pipCheck"
    '--- torch ---'
    "$torchProbe"
    '--- nvidia-smi ---'
    $(if ($nvidiaSmi) { & $nvidiaSmi.Source } else { 'nvidia-smi not found' })
) | Set-Content -Encoding utf8 $output

Write-Output "Контекст запуска сохранён: $output"
