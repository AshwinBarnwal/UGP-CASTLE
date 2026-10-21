#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$PWD}"
MODEL_DIR="$ROOT/external/UCE/model_files"
mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"

download() {
  local id="$1" name="$2" md5="$3"
  if [ ! -s "$name" ]; then
    curl -L --fail --retry 5 --continue-at - "https://ndownloader.figshare.com/files/$id" -o "$name"
  fi
  echo "$md5  $name" | md5sum --check -
}

download 42706555 species_offsets.pkl ddf9143508857e62607f323b338ff383
download 42706558 species_chrom.csv 42a9b5871b1ed955fd05414b3ab89081
download 42706585 all_tokens.torch 3523ea431c403f95701b6eeb0c30b997
download 42715213 protein_embeddings.tar.gz 6756d7fa8348e0f70b5e5b11e324f023
download 43423236 33l_8ep_1024t_1280.torch eed62d4fe51873c87bcf9660faa07707

if [ ! -d protein_embeddings ]; then
  tar -xzf protein_embeddings.tar.gz
fi

