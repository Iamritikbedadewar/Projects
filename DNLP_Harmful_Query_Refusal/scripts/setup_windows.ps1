[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python launcher 'py' was not found. Install Python 3.10, 3.11, or 3.12 first."
}

if (-not (Test-Path ".venv")) {
    $SelectedVersion = $null
    foreach ($Candidate in @("3.11", "3.12", "3.10")) {
        & py "-$Candidate" -c "import sys" 2>$null
        if ($LASTEXITCODE -eq 0) {
            $SelectedVersion = $Candidate
            break
        }
    }
    if ($null -eq $SelectedVersion) {
        throw "No supported Python was found. Install Python 3.10, 3.11, or 3.12."
    }
    & py "-$SelectedVersion" -m venv .venv
}

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
& $Python -m pip install --upgrade pip
& $Python -m pip install -r requirements.txt
& $Python -m pip install -e .

Write-Host ""
Write-Host "Setup complete."
Write-Host "Next, while online and before opening private data, run:"
Write-Host "  .\.venv\Scripts\python.exe .\scripts\download_public_models.py"
