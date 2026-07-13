param(
    [ValidateSet("all", "math", "code")]
    [string]$Task = "all",
    [switch]$SmokeTest,
    [int]$MaxSamples = 0,
    [int]$MaxSteps = -1,
    [int]$MaxSeqLength = 0,
    [string]$ResumeFromCheckpoint = "",
    [string]$ConfigPath = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$ProjectBase = Resolve-Path (Join-Path $ProjectRoot "..")
if ($ConfigPath) {
    $Config = Resolve-Path $ConfigPath
} else {
    $Config = Join-Path $ProjectRoot "configs\train_lora_experiment.yaml"
}

$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$env:HF_HUB_DISABLE_XET = "1"
$env:HF_HUB_ENABLE_HF_TRANSFER = "0"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:HF_HOME = Join-Path $ProjectBase "hf_home"
$env:HF_HUB_CACHE = Join-Path $ProjectBase "hf_cache"
$env:HF_DATASETS_CACHE = Join-Path $ProjectBase "hf_datasets_cache"
New-Item -ItemType Directory -Force -Path $env:HF_HOME, $env:HF_HUB_CACHE, $env:HF_DATASETS_CACHE | Out-Null

$Tasks = @()
if ($Task -eq "all") {
    $Tasks = @("math", "code")
} else {
    $Tasks = @($Task)
}

foreach ($CurrentTask in $Tasks) {
    Write-Host "Training $CurrentTask LoRA..."
    $Args = @(
        "-m", "model_merging_level1.train_lora",
        "--config", $Config,
        "--task", $CurrentTask
    )
    if ($SmokeTest) {
        $Args += "--smoke-test"
    }
    if ($MaxSamples -gt 0) {
        $Args += @("--max-samples", "$MaxSamples")
    }
    if ($MaxSteps -ge 0) {
        $Args += @("--max-steps", "$MaxSteps")
    }
    if ($MaxSeqLength -gt 0) {
        $Args += @("--max-seq-length", "$MaxSeqLength")
    }
    if ($ResumeFromCheckpoint) {
        $Args += @("--resume-from-checkpoint", $ResumeFromCheckpoint)
    }
    conda run -n IntroML python @Args
}
