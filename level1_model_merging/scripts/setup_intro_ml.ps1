param(
    [switch]$InstallTorch
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

if ($InstallTorch) {
    conda install -n IntroML -y pytorch pytorch-cuda=12.4 -c pytorch -c nvidia
}

conda run -n IntroML python -m pip install --upgrade pip
conda run -n IntroML python -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
