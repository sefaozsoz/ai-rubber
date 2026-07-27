# AI Rubber kurulum scripti (Windows / PowerShell)
# Kullanim: .\scripts\setup.ps1 [-VenvPath C:\Users\<kullanici>\venvs\ai-rubber]
param([string]$VenvPath = "$env:USERPROFILE\venvs\ai-rubber")

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

# 1. Sanal ortam
if (-not (Test-Path "$VenvPath\Scripts\python.exe")) {
    python -m venv $VenvPath
}
$py = "$VenvPath\Scripts\python.exe"
$pip = "$VenvPath\Scripts\pip.exe"

# 2. Ucuncu parti repolar
if (-not (Test-Path "third_party\ProPainter")) {
    git clone --depth 1 https://github.com/sczhou/ProPainter.git third_party\ProPainter
}
if (-not (Test-Path "third_party\sam2")) {
    git clone --depth 1 https://github.com/facebookresearch/sam2.git third_party\sam2
}

# 3. Bagimliliklar
& $pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124
& $pip install --no-cache-dir -r requirements.txt
& $pip install --no-cache-dir -e third_party\sam2

# 4. SAM2 agirliklari (~180 MB). ProPainter kendi agirliklarini ilk calistirmada indirir.
$ckpt = "weights\sam2.1_hiera_small.pt"
if (-not (Test-Path $ckpt)) {
    New-Item -ItemType Directory -Force weights | Out-Null
    Invoke-WebRequest "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt" -OutFile $ckpt
}

Write-Output "Kurulum tamam. Calistirmak icin: $py app.py"
