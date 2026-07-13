param(
    [int]$LimitMath = 100,
    [int]$LimitCode = 50,
    [int]$MaxNewTokensMath = 384,
    [int]$MaxNewTokensCode = 384,
    [string]$ProjectRoot = "D:\IntroML\project\level1_model_merging"
)

$ErrorActionPreference = "Stop"

$Config = Join-Path $ProjectRoot "configs\improved_experiment.yaml"
$ProjectBase = Resolve-Path (Join-Path $ProjectRoot "..")
$LogDir = Join-Path $ProjectRoot "run_logs"
$EvalDir = Join-Path $ProjectRoot "results_improved\eval"
$CondaExe = "D:\MySoftware\Miniconda3\Scripts\conda.exe"

New-Item -ItemType Directory -Force -Path $LogDir, $EvalDir | Out-Null

$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$env:HF_HUB_DISABLE_XET = "1"
$env:HF_HUB_ENABLE_HF_TRANSFER = "0"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:HF_HOME = Join-Path $ProjectBase "hf_home"
$env:HF_HUB_CACHE = Join-Path $ProjectBase "hf_cache"
$env:HF_DATASETS_CACHE = Join-Path $ProjectBase "hf_datasets_cache"

$Models = @(
    @{ Alias = "base"; Ref = "Qwen/Qwen2.5-1.5B" },
    @{ Alias = "expert_code"; Ref = "Qwen/Qwen2.5-Coder-1.5B-Instruct" },
    @{ Alias = "expert_math"; Ref = "Qwen/Qwen2.5-Math-1.5B-Instruct" },
    @{ Alias = "slerp_code090_math010"; Ref = (Join-Path $ProjectRoot "merged_models_improved\slerp_code090_math010") },
    @{ Alias = "slerp_code080_math020"; Ref = (Join-Path $ProjectRoot "merged_models_improved\slerp_code080_math020") },
    @{ Alias = "slerp_code070_math030"; Ref = (Join-Path $ProjectRoot "merged_models_improved\slerp_code070_math030") },
    @{ Alias = "slerp_code020_math080"; Ref = (Join-Path $ProjectRoot "merged_models_improved\slerp_code020_math080") }
)

$Benchmarks = @("gsm8k", "humaneval")

foreach ($Model in $Models) {
    if ($Model.Ref -like "$ProjectRoot*" -and -not (Test-Path -LiteralPath $Model.Ref)) {
        Write-Host "Skip $($Model.Alias): merged model directory does not exist."
        continue
    }

    foreach ($Benchmark in $Benchmarks) {
        $Alias = $Model.Alias
        $SummaryPath = Join-Path $EvalDir "$($Alias)_$($Benchmark)_summary.json"
        if (Test-Path -LiteralPath $SummaryPath) {
            Write-Host "Skip $($Alias) $($Benchmark): summary already exists."
            continue
        }

        $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $OutPath = Join-Path $LogDir "eval_${Alias}_${Benchmark}_${Stamp}.out.log"
        $ErrPath = Join-Path $LogDir "eval_${Alias}_${Benchmark}_${Stamp}.err.log"

        $Args = @(
            "run", "-n", "IntroML",
            "python", "-m", "model_merging_level1.evaluate",
            "--config", $Config,
            "--model", "$($Alias)=$($Model.Ref)",
            "--benchmark", $Benchmark
        )

        if ($Benchmark -eq "gsm8k") {
            $Args += @("--limit-math", "$LimitMath", "--max-new-tokens-math", "$MaxNewTokensMath")
        } else {
            $Args += @("--limit-code", "$LimitCode", "--max-new-tokens-code", "$MaxNewTokensCode", "--allow-code-execution")
        }

        Write-Host "Running $Alias $Benchmark..."
        $Process = Start-Process -FilePath $CondaExe `
            -ArgumentList $Args `
            -WorkingDirectory $ProjectRoot `
            -RedirectStandardOutput $OutPath `
            -RedirectStandardError $ErrPath `
            -NoNewWindow `
            -PassThru `
            -Wait

        if ($Process.ExitCode -ne 0) {
            Write-Host "FAILED $Alias $Benchmark with exit code $($Process.ExitCode)."
            Write-Host "stdout: $OutPath"
            Write-Host "stderr: $ErrPath"
            exit $Process.ExitCode
        }

        Write-Host "Finished $Alias $Benchmark."
    }
}

Write-Host "Summarizing results..."
$SummaryOut = Join-Path $LogDir ("summarize_improved_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".out.log")
$SummaryErr = Join-Path $LogDir ("summarize_improved_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".err.log")
$SummaryArgs = @(
    "run", "-n", "IntroML",
    "python", "-m", "model_merging_level1.summarize",
    "--config", $Config
)
$SummaryProcess = Start-Process -FilePath $CondaExe `
    -ArgumentList $SummaryArgs `
    -WorkingDirectory $ProjectRoot `
    -RedirectStandardOutput $SummaryOut `
    -RedirectStandardError $SummaryErr `
    -NoNewWindow `
    -PassThru `
    -Wait

exit $SummaryProcess.ExitCode
