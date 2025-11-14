# PowerShell script to run GraphSteal pipeline
# Usage: .\run_all.ps1

$env:PYTHONPATH = $PSScriptRoot

Write-Host "=== Step 1: Pre-train graph diffusion model ===" -ForegroundColor Green
cd src
python run_train_diffusion.py

Write-Host "`n=== Step 2: Train the classifier ===" -ForegroundColor Green
python run_qm9_classifier.py

Write-Host "`n=== Step 3: Reconstruct training graphs ===" -ForegroundColor Green
python run_reconstruct.py

cd ..



