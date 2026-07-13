param(
    [string]$Config = "",
    [switch]$IncludeStale
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$env:PYTHONPATH = Join-Path $ProjectRoot "src"

if (-not $Config) {
    $Config = Join-Path $ProjectRoot "configs\improved_experiment.yaml"
}

$PythonArgs = @(
    "-m", "model_merging_level1.summarize",
    "--config", $Config
)

if ($IncludeStale) {
    $PythonArgs += "--include-stale"
}

conda run -n IntroML python @PythonArgs
