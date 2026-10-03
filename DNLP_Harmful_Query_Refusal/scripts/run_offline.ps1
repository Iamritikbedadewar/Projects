[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("inspect", "references", "experiment", "test", "audit")]
    [string]$Command,

    [string]$DataDir,

    [string]$Models = "kimcnn,flan_zero,flan_finetuned",

    [string]$Folds = "0,1,2,3,4",

    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto",

    [ValidateSet("full", "short")]
    [string]$PromptVariant = "full",

    [ValidateSet("gold", "generic")]
    [string]$RationaleMode = "gold",

    [string]$ReferenceSidecar,

    [switch]$SkipBertScore,

    [switch]$SavePrivateErrors
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Virtual environment not found. Run scripts\setup_windows.ps1 first."
}

$env:HF_HUB_OFFLINE = "1"
$env:HF_DATASETS_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
$env:DO_NOT_TRACK = "1"
$env:WANDB_DISABLED = "true"
$env:TOKENIZERS_PARALLELISM = "false"

if (
    $Command -in @("inspect", "references", "experiment") -and
    [string]::IsNullOrWhiteSpace($DataDir)
) {
    throw "-DataDir is required for '$Command'."
}

switch ($Command) {
    "inspect" {
        & $Python .\scripts\inspect_schema.py --data-dir $DataDir
    }
    "references" {
        if ([string]::IsNullOrWhiteSpace($ReferenceSidecar)) {
            $ReferenceSidecar = Join-Path $ProjectRoot "private_outputs\silver_reference_rationales.json"
        }
        & $Python .\scripts\build_reference_sidecar.py `
            --data-dir $DataDir `
            --output $ReferenceSidecar
    }
    "experiment" {
        $Arguments = @(
            "-m", "dnlp_refusal.run_experiments",
            "--data-dir", $DataDir,
            "--models", $Models,
            "--folds", $Folds,
            "--device", $Device,
            "--prompt-variant", $PromptVariant,
            "--rationale-mode", $RationaleMode
        )
        if ($SkipBertScore) {
            $Arguments += "--skip-bertscore"
        }
        if (-not [string]::IsNullOrWhiteSpace($ReferenceSidecar)) {
            $Arguments += @("--reference-sidecar", $ReferenceSidecar)
        }
        if ($SavePrivateErrors) {
            $Arguments += "--save-private-errors"
        }
        & $Python @Arguments
    }
    "test" {
        & $Python -m pytest
    }
    "audit" {
        & $Python .\scripts\privacy_audit.py
    }
}
