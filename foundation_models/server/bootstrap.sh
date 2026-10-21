#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$PWD}"
cd "$ROOT"
mkdir -p external results/foundation_models logs/foundation_models
mkdir -p foundation_models/resources

if command -v mamba >/dev/null 2>&1; then CONDA=mamba; else CONDA=conda; fi
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

if [ ! -d external/UCE ]; then
  git clone https://github.com/snap-stanford/UCE.git external/UCE
fi
git -C external/UCE checkout 9c416007be15ad6753dc84af4468c1dc10421ab9

if [ ! -d external/nicheformer ]; then
  git clone https://github.com/theislab/nicheformer.git external/nicheformer
fi
git -C external/nicheformer checkout 485cadbc5caa15119adfd54228f8a8af835fcabc

if [ ! -s foundation_models/resources/hgnc_complete_set.txt ]; then
  curl -L --fail --retry 5 \
    https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt \
    -o foundation_models/resources/hgnc_complete_set.txt
fi

$CONDA create -y -n ugp-uceserver python=3.10 pip
$CONDA run -n ugp-uceserver pip install torch==2.1.1 --index-url "$TORCH_INDEX_URL"
$CONDA run -n ugp-uceserver pip install -r foundation_models/requirements-uceserver.txt

$CONDA create -y -n ugp-nicheformer python=3.10 pip
$CONDA run -n ugp-nicheformer pip install torch==2.5.1 --index-url "$TORCH_INDEX_URL"
$CONDA run -n ugp-nicheformer pip install -r foundation_models/requirements-nicheformer.txt
$CONDA run -n ugp-nicheformer pip install -r foundation_models/requirements-benchmark.txt

echo "Environments created. Verify torch sees CUDA before launching."
$CONDA run -n ugp-uceserver python -c 'import torch; print("UCE torch", torch.__version__, "CUDA", torch.cuda.is_available(), torch.version.cuda)'
$CONDA run -n ugp-nicheformer python -c 'import torch; print("Nicheformer torch", torch.__version__, "CUDA", torch.cuda.is_available(), torch.version.cuda)'

