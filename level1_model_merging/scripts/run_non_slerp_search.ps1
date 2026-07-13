param(
    [int]$LimitMath = 100,
    [int]$LimitCode = 50,
    [switch]$SkipExisting
)

$ErrorActionPreference = "Stop"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Pipeline = Join-Path $ScriptRoot "run_improved_pipeline.ps1"
$Candidates = @(
    "task_code080_math020_l100",
    "ties_code080_math020_d080"
)

$Args = @(
    "-LimitMath", "$LimitMath",
    "-LimitCode", "$LimitCode",
    "-Only"
) + $Candidates

if ($SkipExisting) {
    $Args += "-SkipExisting"
}

& $Pipeline @Args
