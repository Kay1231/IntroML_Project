param(
    [string]$Config = "",
    [int]$LimitMath = 100,
    [int]$LimitCode = 50,
    [string[]]$Benchmark = @("gsm8k", "humaneval"),
    [switch]$AllowCodeExecution,
    [switch]$SkipExisting
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$ProjectBase = Resolve-Path (Join-Path $ProjectRoot "..")
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$env:HF_HUB_DISABLE_XET = "1"
$env:HF_HUB_ENABLE_HF_TRANSFER = "0"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:HF_HOME = Join-Path $ProjectBase "hf_home"
$env:HF_HUB_CACHE = Join-Path $ProjectBase "hf_cache"
$env:HF_DATASETS_CACHE = Join-Path $ProjectBase "hf_datasets_cache"
New-Item -ItemType Directory -Force -Path $env:HF_HOME, $env:HF_HUB_CACHE, $env:HF_DATASETS_CACHE | Out-Null

if (-not $Config) {
    $Config = Join-Path $ProjectRoot "configs\improved_experiment.yaml"
}

$PythonArgs = @(
    "-m", "model_merging_level1.evaluate",
    "--config", $Config,
    "--include-base",
    "--include-experts",
    "--include-merges",
    "--limit-math", "$LimitMath",
    "--limit-code", "$LimitCode"
)

foreach ($Item in $Benchmark) {
    $PythonArgs += @("--benchmark", $Item)
}

if ($AllowCodeExecution) {
    $PythonArgs += "--allow-code-execution"
}

if ($SkipExisting) {
    $PythonArgs += "--skip-existing"
}

conda run -n IntroML python @PythonArgs
