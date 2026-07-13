param(
    [int]$LimitMath = 100,
    [int]$LimitCode = 50,
    [string[]]$Only = @(),
    [switch]$Cpu,
    [switch]$Force,
    [switch]$SkipExisting,
    [switch]$SkipCodeExecution,
    [switch]$SkipMaterialize
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$ProjectBase = Resolve-Path (Join-Path $ProjectRoot "..")
$Config = Join-Path $ProjectRoot "configs\lora_experiment.yaml"

$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$env:HF_HUB_DISABLE_XET = "1"
$env:HF_HUB_ENABLE_HF_TRANSFER = "0"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:HF_HOME = Join-Path $ProjectBase "hf_home"
$env:HF_HUB_CACHE = Join-Path $ProjectBase "hf_cache"
$env:HF_DATASETS_CACHE = Join-Path $ProjectBase "hf_datasets_cache"
New-Item -ItemType Directory -Force -Path $env:HF_HOME, $env:HF_HUB_CACHE, $env:HF_DATASETS_CACHE | Out-Null

if (-not $SkipMaterialize) {
    Write-Host "Materializing LoRA experts..."
    $MaterializeArgs = @(
        "-m", "model_merging_level1.materialize_lora_experts",
        "--config", $Config
    )
    if ($Cpu) {
        $MaterializeArgs += @("--device-map", "cpu")
    }
    if ($Force) {
        $MaterializeArgs += "--force"
    }
    conda run -n IntroML python @MaterializeArgs
}

Write-Host "Generating MergeKit configs..."
$GenerateArgs = @(
    "-m", "model_merging_level1.merge_configs",
    "--config", $Config
)
foreach ($Name in $Only) {
    $GenerateArgs += @("--only", $Name)
}
conda run -n IntroML python @GenerateArgs

Write-Host "Running merges..."
$MergeArgs = @(
    "-m", "model_merging_level1.run_merges",
    "--config", $Config
)
if (-not $Cpu) {
    $MergeArgs += "--cuda"
}
if ($Force) {
    $MergeArgs += "--force"
}
foreach ($Name in $Only) {
    $MergeArgs += @("--only", $Name)
}
conda run -n IntroML python @MergeArgs

Write-Host "Evaluating base, LoRA experts, and merged models..."
$EvalArgs = @(
    "-m", "model_merging_level1.evaluate",
    "--config", $Config,
    "--include-base",
    "--include-experts",
    "--limit-math", "$LimitMath",
    "--limit-code", "$LimitCode",
    "--benchmark", "gsm8k",
    "--benchmark", "humaneval"
)
if ($Only.Count -eq 0) {
    $EvalArgs += "--include-merges"
} else {
    $MergeOutputDir = Join-Path $ProjectRoot "merged_models_lora"
    foreach ($Name in $Only) {
        $EvalArgs += @("--model", "$Name=$(Join-Path $MergeOutputDir $Name)")
    }
}
if (-not $SkipCodeExecution) {
    $EvalArgs += "--allow-code-execution"
}
if ($SkipExisting) {
    $EvalArgs += "--skip-existing"
}
conda run -n IntroML python @EvalArgs

Write-Host "Summarizing LoRA results..."
conda run -n IntroML python -m model_merging_level1.summarize --config $Config
