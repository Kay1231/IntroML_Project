param(
    [string]$Config = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$env:PYTHONPATH = Join-Path $ProjectRoot "src"

if (-not $Config) {
    $Config = Join-Path $ProjectRoot "configs\improved_experiment.yaml"
}

conda run -n IntroML python -m model_merging_level1.merge_configs --config $Config
