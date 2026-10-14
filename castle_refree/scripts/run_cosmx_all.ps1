$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$workspaceRoot = Split-Path $repoRoot -Parent
$castlePython = Join-Path $workspaceRoot ".venv-castle/Scripts/python.exe"
$scviPython = Join-Path $workspaceRoot ".venv-scvi/Scripts/python.exe"
$scvivaPython = Join-Path $workspaceRoot ".venv-scviva/Scripts/python.exe"

# Prefer an explicit override, then an R installation on PATH, then a portable
# installation in the drive-level R directory used by this project on Windows.
$rterm = $env:RCTD_R_EXECUTABLE
if (-not $rterm) {
    $rCommand = Get-Command Rterm.exe, Rscript.exe -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($rCommand) {
        $rterm = $rCommand.Source
    }
}
if (-not $rterm) {
    $portableRRoot = Join-Path (Split-Path $workspaceRoot -Parent) "R"
    $portableR = Get-ChildItem $portableRRoot -Directory -Filter "R-*" -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending |
        ForEach-Object {
            @(
                (Join-Path $_.FullName "bin/x64/Rterm.exe"),
                (Join-Path $_.FullName "bin/Rscript.exe")
            )
        } |
        Where-Object { Test-Path $_ } |
        Select-Object -First 1
    if ($portableR) {
        $rterm = $portableR
    }
}
if (-not $rterm -or -not (Test-Path $rterm)) {
    throw "Rterm/Rscript was not found. Put R on PATH or set RCTD_R_EXECUTABLE."
}
$runState = Join-Path $repoRoot "results/cosmx_run_state"
$matplotlibCache = Join-Path $runState "matplotlib"
New-Item -ItemType Directory -Force $runState, $matplotlibCache | Out-Null
$env:MPLCONFIGDIR = $matplotlibCache
Set-Location $repoRoot

function Invoke-Checked {
    param(
        [string]$Name,
        [string]$Executable,
        [string[]]$Arguments
    )
    $started = Get-Date
    Write-Output "[$($started.ToString('s'))] START $Name"
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
    $finished = Get-Date
    Write-Output "[$($finished.ToString('s'))] DONE $Name ($([math]::Round(($finished-$started).TotalMinutes, 2)) minutes)"
}

$castleDone = Join-Path $repoRoot "results/castle_refree_cosmx_frontal_cortex_beta2/castle_embeddings_and_qc.npz"
if (-not (Test-Path $castleDone)) {
    $castleArgs = @(
        "-m", "castle_refree.cli",
        "--config", "castle_refree/config/cosmx_frontal_cortex_beta2.yaml"
    )
    $castleCheckpoint = Join-Path $repoRoot "results/castle_refree_cosmx_frontal_cortex_beta2/checkpoint.pt"
    if (Test-Path $castleCheckpoint) {
        $castleArgs += @("--resume", $castleCheckpoint)
    }
    Invoke-Checked "CASTLE" $castlePython $castleArgs
} else {
    Write-Output "CASTLE output already exists; skipping."
}

$scviDone = Join-Path $repoRoot "results/scvi_cosmx_frontal_cortex/adata_scvi.h5ad"
if (-not (Test-Path $scviDone)) {
    Invoke-Checked "scVI" $scviPython @(
        "castle_refree/scripts/run_scvi_proseg.py",
        "--input", "data/cosmx_frontal_cortex.h5ad",
        "--output", "results/scvi_cosmx_frontal_cortex",
        "--max-epochs", "200",
        "--batch-size", "512",
        "--n-latent", "10",
        "--seed", "42"
    )
} else {
    Write-Output "scVI output already exists; skipping."
}

$scvivaDone = Join-Path $repoRoot "results/scviva_cosmx_frontal_cortex/adata_scviva.h5ad"
if (-not (Test-Path $scvivaDone)) {
    Invoke-Checked "scVIVA" $scvivaPython @(
        "castle_refree/scripts/run_scviva_proseg.py",
        "--input", $scviDone,
        "--output", "results/scviva_cosmx_frontal_cortex",
        "--expression-embedding-key", "X_scVI",
        "--max-epochs", "600",
        "--batch-size", "512",
        "--learning-rate", "0.0005",
        "--k-nn", "20",
        "--seed", "0"
    )
} else {
    Write-Output "scVIVA output already exists; skipping."
}

$splitDone = Join-Path $repoRoot "results/cosmx_frontal_cortex_rctd_split/split_cosmx_frontal_cortex.rds"
if (-not (Test-Path $splitDone)) {
    Invoke-Checked "RCTD + SPLIT" $rterm @(
        "--vanilla", "--slave", "-f",
        "castle_refree/R/run_cosmx_frontal_cortex_rctd_split.R"
    )
} else {
    Write-Output "RCTD + SPLIT output already exists; skipping."
}

Write-Output "[$((Get-Date).ToString('s'))] ALL COSMX METHODS COMPLETE"
