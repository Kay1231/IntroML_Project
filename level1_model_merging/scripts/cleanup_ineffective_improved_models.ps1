param(
    [switch]$WhatIf
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$MergeRoot = Join-Path $ProjectRoot "merged_models_improved"

if (-not (Test-Path -LiteralPath $MergeRoot)) {
    Write-Host "No merged_models_improved directory found."
    exit 0
}

$ResolvedMergeRoot = (Resolve-Path -LiteralPath $MergeRoot).Path
$ObsoleteModels = @(
    "slerp_code035_math065",
    "slerp_code050_math050",
    "task_code025_math025_l050",
    "task_code035_math035_l060",
    "ties_code025_math025_d020",
    "ties_code035_math035_d030"
)

foreach ($Name in $ObsoleteModels) {
    $Target = Join-Path $ResolvedMergeRoot $Name
    if (-not (Test-Path -LiteralPath $Target)) {
        Write-Host "Skip ${Name}: not found."
        continue
    }

    $ResolvedTarget = (Resolve-Path -LiteralPath $Target).Path
    if (-not $ResolvedTarget.StartsWith($ResolvedMergeRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to delete outside merged_models_improved: $ResolvedTarget"
    }

    Remove-Item -LiteralPath $ResolvedTarget -Recurse -Force -WhatIf:$WhatIf
}
