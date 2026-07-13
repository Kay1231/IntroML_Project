param()

$ErrorActionPreference = "Stop"

$Script = Join-Path $PSScriptRoot "run_table1_subset.ps1"

& $Script `
    -Configs @("vitB_r16_knots_ties_subset", "vitB_r16_ties_subset") `
    -Datasets @("mnist") `
    -SkipHeads
