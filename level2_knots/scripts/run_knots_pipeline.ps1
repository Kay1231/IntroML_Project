param(
    [int]$LimitMath = 100,
    [int]$LimitCode = 50,
    [string[]]$Only = @(),
    [switch]$SkipMerge,
    [switch]$SkipEval,
    [switch]$IncludeBaseExperts,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RepoRoot = (Resolve-Path (Join-Path $ProjectRoot "..")).Path
$Level1Root = Join-Path $RepoRoot "level1_model_merging"
$Config = Join-Path $ProjectRoot "configs\knots_qwen_lora_v2.yaml"

$env:PYTHONPATH = "$ProjectRoot\src;$Level1Root\src"
$env:HF_HOME = Join-Path $RepoRoot "hf_home"
$env:HF_DATASETS_CACHE = Join-Path $RepoRoot "hf_datasets_cache"
$env:HF_HUB_CACHE = Join-Path $RepoRoot "hf_cache"

$Merges = @(
    @{ Name = "knots_ties_code020_math080_d080"; Method = "ties"; Code = "0.20"; Math = "0.80"; Density = "0.80" },
    @{ Name = "knots_ta_code020_math080"; Method = "task_arithmetic"; Code = "0.20"; Math = "0.80"; Density = "1.00" }
)

if ($Only.Count -gt 0) {
    $Allowed = @{}
    foreach ($Name in $Only) {
        $Allowed[$Name] = $true
    }
    $Merges = @($Merges | Where-Object { $Allowed.ContainsKey($_.Name) })
    if ($Merges.Count -eq 0) {
        throw "No requested KnOTS merge names matched the configured list."
    }
}

if (-not $SkipMerge) {
    foreach ($Merge in $Merges) {
        Write-Host "Running KnOTS merge $($Merge.Name)..."
        $Args = @(
            "run", "-n", "IntroML",
            "python", "-m", "level2_knots.knots_merge_lora",
            "--config", $Config,
            "--name", $Merge.Name,
            "--method", $Merge.Method,
            "--code-weight", $Merge.Code,
            "--math-weight", $Merge.Math,
            "--density", $Merge.Density,
            "--torch-dtype", "bfloat16"
        )
        if ($Force) {
            $Args += "--force"
        }
        & conda @Args
        if ($LASTEXITCODE -ne 0) {
            throw "KnOTS merge failed for $($Merge.Name)"
        }
    }
}

if (-not $SkipEval) {
    Write-Host "Evaluating Level 2 KnOTS models..."
    $EvalArgs = @(
        "run", "-n", "IntroML",
        "python", "-m", "model_merging_level1.evaluate",
        "--config", $Config,
        "--include-merges",
        "--benchmark", "gsm8k",
        "--benchmark", "humaneval",
        "--limit-math", "$LimitMath",
        "--limit-code", "$LimitCode",
        "--allow-code-execution",
        "--skip-existing"
    )
    if ($IncludeBaseExperts) {
        $EvalArgs += @("--include-base", "--include-experts")
    }
    & conda @EvalArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Level 2 evaluation failed"
    }

    Write-Host "Summarizing Level 2 KnOTS results..."
    & conda run -n IntroML python -m model_merging_level1.summarize --config $Config
    if ($LASTEXITCODE -ne 0) {
        throw "Level 2 summarization failed"
    }
}

Write-Host "Level 2 KnOTS pipeline finished."
