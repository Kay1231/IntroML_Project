param(
    [int]$LimitMath = 100,
    [int]$LimitCode = 50,
    [switch]$SkipTraining,
    [switch]$SkipMaterialize,
    [switch]$SkipMergeKit,
    [switch]$SkipAdapterMerges,
    [switch]$SkipCodeExecution,
    [switch]$SkipExisting,
    [switch]$Cpu,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$ProjectBase = Resolve-Path (Join-Path $ProjectRoot "..")
$Config = Join-Path $ProjectRoot "configs\train_lora_v2_experiment.yaml"

$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$env:HF_HUB_DISABLE_XET = "1"
$env:HF_HUB_ENABLE_HF_TRANSFER = "0"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:HF_HOME = Join-Path $ProjectBase "hf_home"
$env:HF_HUB_CACHE = Join-Path $ProjectBase "hf_cache"
$env:HF_DATASETS_CACHE = Join-Path $ProjectBase "hf_datasets_cache"
New-Item -ItemType Directory -Force -Path $env:HF_HOME, $env:HF_HUB_CACHE, $env:HF_DATASETS_CACHE | Out-Null

if (-not $SkipTraining) {
    & (Join-Path $PSScriptRoot "train_lora_experts.ps1") -Task all -ConfigPath $Config
}

if (-not $SkipMaterialize) {
    Write-Host "Materializing v2 LoRA experts..."
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

if (-not $SkipMergeKit) {
    Write-Host "Running v2 MergeKit merges..."
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
    conda run -n IntroML python @MergeArgs
}

if (-not $SkipAdapterMerges) {
    Write-Host "Running v2 PEFT adapter merges..."
    $AdapterMerges = @(
        @{ Name = "lora_svd_v2_code080_math020"; Combination = "svd"; Code = "0.80"; Math = "0.20"; Rank = "16"; Density = "" },
        @{ Name = "lora_ties_svd_v2_code080_math020"; Combination = "ties_svd"; Code = "0.80"; Math = "0.20"; Rank = "16"; Density = "0.80" }
    )
    foreach ($Merge in $AdapterMerges) {
        $AdapterArgs = @(
            "-m", "model_merging_level1.merge_lora_adapters",
            "--config", $Config,
            "--name", $Merge.Name,
            "--code-weight", $Merge.Code,
            "--math-weight", $Merge.Math,
            "--combination-type", $Merge.Combination,
            "--svd-rank", $Merge.Rank
        )
        if ($Merge.Density) {
            $AdapterArgs += @("--density", $Merge.Density)
        }
        if ($Cpu) {
            $AdapterArgs += @("--device-map", "cpu")
        }
        if ($Force) {
            $AdapterArgs += "--force"
        }
        conda run -n IntroML python @AdapterArgs
    }
}

Write-Host "Evaluating v2 base, experts, MergeKit merges, and PEFT adapter merges..."
$EvalArgs = @(
    "-m", "model_merging_level1.evaluate",
    "--config", $Config,
    "--include-base",
    "--include-experts",
    "--include-merges",
    "--limit-math", "$LimitMath",
    "--limit-code", "$LimitCode",
    "--benchmark", "gsm8k",
    "--benchmark", "humaneval"
)

$AdapterMergeDir = Join-Path $ProjectRoot "merged_models_trained_lora_v2"
$EvalArgs += @("--model", "lora_svd_v2_code080_math020=$(Join-Path $AdapterMergeDir 'lora_svd_v2_code080_math020')")
$EvalArgs += @("--model", "lora_ties_svd_v2_code080_math020=$(Join-Path $AdapterMergeDir 'lora_ties_svd_v2_code080_math020')")

if (-not $SkipCodeExecution) {
    $EvalArgs += "--allow-code-execution"
}
if ($SkipExisting) {
    $EvalArgs += "--skip-existing"
}
conda run -n IntroML python @EvalArgs

Write-Host "Summarizing v2 results..."
conda run -n IntroML python -m model_merging_level1.summarize --config $Config
