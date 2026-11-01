param([switch]$Cpu)
$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectDirectory
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 diperlukan untuk membuat environment.' }
}
$projectPython = Join-Path $projectDirectory '.venv\Scripts\python.exe'
& $projectPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Pembaruan pip gagal.' }
$torchIndex = if ($Cpu) { 'https://download.pytorch.org/whl/cpu' } else { 'https://download.pytorch.org/whl/cu121' }
& $projectPython -m pip install torch==2.5.1 torchvision==0.20.1 --index-url $torchIndex
if ($LASTEXITCODE -ne 0) { throw 'Instalasi PyTorch gagal.' }
& $projectPython -m pip install -r requirements-tested.txt
if ($LASTEXITCODE -ne 0) { throw 'Instalasi dependensi gagal.' }
& $projectPython scripts/download_models.py
if ($LASTEXITCODE -ne 0) { throw 'Persiapan model gagal.' }
Write-Host 'Selesai. Buka Run Rally Vision.bat.'
