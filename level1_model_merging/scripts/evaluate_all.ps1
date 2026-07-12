param(
    [int]$LimitMath = 100,
    [int]$LimitCode = 50,
    [string[]]$Benchmark = @("gsm8k", "humaneval"),
    [switch]$AllowCodeExecution
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

$PythonArgs = @(
    "-m", "model_merging_level1.evaluate",
    "--config", (Join-Path $ProjectRoot "configs\default_experiment.yaml"),
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

conda run -n IntroML python @PythonArgs
