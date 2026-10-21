$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$workspaceRoot = Split-Path $repoRoot -Parent
$castlePython = Join-Path $workspaceRoot ".venv-castle/Scripts/python.exe"
$scviPython = Join-Path $workspaceRoot ".venv-scvi/Scripts/python.exe"
$scvivaPython = Join-Path $workspaceRoot ".venv-scviva/Scripts/python.exe"
$runtimeTemp = Join-Path $workspaceRoot ".runtime-temp"
$matplotlibCache = Join-Path $repoRoot "results/cosmx_run_state/matplotlib-hvg2000"
$torchCache = Join-Path $workspaceRoot ".torch-cache"
$numbaCache = Join-Path $workspaceRoot ".numba-cache"
New-Item -ItemType Directory -Force $runtimeTemp, $matplotlibCache, $torchCache, $numbaCache | Out-Null
$env:TEMP = $runtimeTemp
$env:TMP = $runtimeTemp
$env:MPLCONFIGDIR = $matplotlibCache
$env:TORCH_HOME = $torchCache
$env:NUMBA_CACHE_DIR = $numbaCache
Set-Location $repoRoot

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Executable failed with exit code $LASTEXITCODE" }
}

$hvgInput = Join-Path $repoRoot "data/cosmx_frontal_cortex_hvg2000.h5ad"
if (-not (Test-Path $hvgInput)) {
    Invoke-Checked $scviPython @("castle_refree/scripts/prepare_cosmx_hvg2000.py")
}

$castleOutput = Join-Path $repoRoot "results/castle_refree_cosmx_frontal_cortex_hvg2000_beta2/castle_embeddings_and_qc.npz"
if (-not (Test-Path $castleOutput)) {
    $args = @("-m", "castle_refree.cli", "--config", "castle_refree/config/cosmx_frontal_cortex_hvg2000_beta2_gpu.yaml")
    $checkpoint = Join-Path $repoRoot "results/castle_refree_cosmx_frontal_cortex_hvg2000_beta2/checkpoint.pt"
    if (Test-Path $checkpoint) { $args += @("--resume", $checkpoint) }
    Invoke-Checked $castlePython $args
}

$scviOutput = Join-Path $repoRoot "results/scvi_cosmx_frontal_cortex_hvg2000/adata_scvi.h5ad"
if (-not (Test-Path $scviOutput)) {
    Invoke-Checked $scviPython @(
        "castle_refree/scripts/run_scvi_proseg.py",
        "--input", $hvgInput,
        "--output", "results/scvi_cosmx_frontal_cortex_hvg2000",
        "--max-epochs", "200", "--batch-size", "128", "--n-latent", "10",
        "--accelerator", "gpu", "--devices", "1", "--seed", "42"
    )
}

$scvivaOutput = Join-Path $repoRoot "results/scviva_cosmx_frontal_cortex_hvg2000/adata_scviva.h5ad"
if (-not (Test-Path $scvivaOutput)) {
    Invoke-Checked $scvivaPython @(
        "castle_refree/scripts/run_scviva_proseg.py",
        "--input", $scviOutput,
        "--output", "results/scviva_cosmx_frontal_cortex_hvg2000",
        "--expression-embedding-key", "X_scVI",
        "--max-epochs", "600", "--batch-size", "128",
        "--accelerator", "gpu", "--devices", "1",
        "--learning-rate", "0.0005", "--k-nn", "20", "--seed", "0"
    )
}
