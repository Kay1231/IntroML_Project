param(
    [string]$Config = "",
    [string[]]$Only = @(),
    [switch]$Cpu,
    [switch]$Force
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
New-Item -ItemType Directory -Force -Path $env:HF_HOME, $env:HF_HUB_CACHE | Out-Null

if (-not $Config) {
    $Config = Join-Path $ProjectRoot "configs\improved_experiment.yaml"
}

$PythonArgs = @(
    "-m", "model_merging_level1.run_merges",
    "--config", $Config
)

if (-not $Cpu) {
    $PythonArgs += "--cuda"
}

if ($Force) {
    $PythonArgs += "--force"
}

foreach ($Name in $Only) {
    $PythonArgs += @("--only", $Name)
}

conda run -n IntroML python @PythonArgs
