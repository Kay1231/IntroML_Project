$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$env:PYTHONPATH = Join-Path $ProjectRoot "src"

conda run -n IntroML python -m model_merging_level1.summarize --config (Join-Path $ProjectRoot "configs\default_experiment.yaml")
