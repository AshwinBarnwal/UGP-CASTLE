#!/usr/bin/env bash
set -euo pipefail

echo "== host =="
hostname
uname -a
echo "== GPU =="
nvidia-smi || true
echo "== CUDA compiler =="
nvcc --version || true
echo "== Python/conda =="
command -v python || true
python --version || true
command -v mamba || command -v conda || true
echo "== storage =="
df -h .
echo "== memory =="
free -h

