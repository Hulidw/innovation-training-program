param([int]$Steps = 300)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
python -m pip install -r requirements.txt
python train.py --mode compare --steps $Steps --output runs/compare
