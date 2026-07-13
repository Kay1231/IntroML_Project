param(
    [string[]]$Configs = @("vitB_r16_knots_ties_subset", "vitB_r16_ties_subset"),
    [string[]]$Datasets = @("mnist", "svhn", "gtsrb"),
    [string]$PythonExe = "D:\MySoftware\Miniconda3\envs\IntroML\python.exe",
    [switch]$SkipHeads
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RepoRoot = (Resolve-Path (Join-Path $ProjectRoot "..")).Path

$env:PYTHONPATH = $ProjectRoot
$env:HF_HOME = Join-Path $RepoRoot "hf_home"
$env:HF_DATASETS_CACHE = Join-Path $RepoRoot "hf_datasets_cache"
$env:HF_HUB_CACHE = Join-Path $RepoRoot "hf_cache"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONBREAKPOINT = "0"

if (-not (Test-Path $PythonExe)) {
    Write-Warning "Configured Python executable was not found: $PythonExe. Falling back to python on PATH."
    $PythonExe = "python"
}

Set-Location $ProjectRoot

if (-not $SkipHeads) {
    Write-Host "Generating CLIP heads for $($Datasets -join '/')..."
    & $PythonExe .\scripts\generate_subset_clip_heads.py `
        --datasets $Datasets `
        --cache-dir (Join-Path $RepoRoot "hf_cache") `
        --output-dir .\clip_heads\ViT-B-32-CLIP
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to generate CLIP heads."
    }
}

foreach ($Config in $Configs) {
    Write-Host "Running official KnOTS Table 1 subset config: $Config"
    & $PythonExe -m eval_scripts.8vision_pertask_subset `
        --config-name $Config `
        --data-root data `
        --results-dir results_subset `
        --prepare-device cpu `
        --eval-device cuda `
        --datasets $Datasets
    if ($LASTEXITCODE -ne 0) {
        throw "Subset evaluation failed for $Config"
    }
}

Write-Host "Official Table 1 subset run finished."
