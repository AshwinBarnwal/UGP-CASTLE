"""Stream spatial cells through the official Hugging Face Nicheformer model.

This avoids materializing a mostly-zero 188k x 20,310 aligned H5AD. Tokenization
matches the released CosMx procedure: library-size normalization to 10,000,
division by the CosMx technology mean, ranking, and fixed-vocabulary tokens.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from tqdm import tqdm


def tokenize_chunk(
    matrix,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
    technology_mean: np.ndarray,
    context_length: int,
    technology_token: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_cells = matrix.shape[0]
    aligned = np.zeros((n_cells, len(technology_mean)), dtype=np.float32)
    selected = matrix[:, source_indices]
    if sparse.issparse(selected):
        selected = selected.toarray()
    aligned[:, target_indices] = np.asarray(selected, dtype=np.float32)
    totals = aligned.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1
    scores = aligned * (10000.0 / totals)
    scores /= technology_mean[None, :]

    gene_slots = context_length - 3
    token_ids = np.zeros((n_cells, context_length), dtype=np.int64)
    token_ids[:, :3] = np.array([5, technology_token, 4], dtype=np.int64)
    for row in range(n_cells):
        nonzero = np.flatnonzero(scores[row] > 0)
        if len(nonzero):
            order = nonzero[np.argsort(-scores[row, nonzero], kind="stable")[:gene_slots]]
            token_ids[row, 3:3 + len(order)] = order + 30
    attention = token_ids != 0
    return token_ids, attention


def main() -> None:
    from transformers import AutoModel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adata", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--technology-mean", type=Path, required=True)
    parser.add_argument("--technology-token", type=int, choices=(7, 8, 9), default=8,
                        help="Nicheformer token: MERFISH=7, CosMx=8, Xenium=9")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="theislab/Nicheformer")
    parser.add_argument("--revision", default="0ba4ba162f9ad3d6c8b713761954c31be2fa69ed")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--read-chunk-size", type=int, default=512)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--limit-cells", type=int, default=None, help="Smoke-test prefix; omit for full run")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the planned Nicheformer run")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support bf16; rerun with --precision fp16")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(args.adata, backed="r")
    n_cells = min(adata.n_obs, args.limit_cells or adata.n_obs)
    mapping = pd.read_csv(args.mapping, sep="\t")
    mapping = mapping[mapping["status"].eq("matched")].copy()
    mapping["source_index"] = mapping["source_index"].astype(int)
    mapping["model_index"] = mapping["model_index"].astype(int)
    mapping = mapping.sort_values("source_index")
    source_indices = mapping["source_index"].to_numpy()
    target_indices = mapping["model_index"].to_numpy()

    technology_mean = np.load(args.technology_mean).astype(np.float32)
    technology_mean = np.nan_to_num(technology_mean, nan=0.0, posinf=0.0, neginf=0.0)
    technology_mean = np.where((technology_mean % 1) >= 0.5,
                               np.ceil(technology_mean), np.floor(technology_mean))
    technology_mean[technology_mean <= 0] = 1

    model = AutoModel.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)
    model.eval().cuda()
    context_length = int(model.config.context_length)
    embedding_dim = int(model.config.dim_model)
    output_path = args.output_dir / "nicheformer_embeddings.npy"
    progress_path = args.output_dir / "progress.json"
    cell_path = args.output_dir / "cells.tsv"
    if not cell_path.exists():
        pd.Series(adata.obs_names[:n_cells].astype(str)).to_csv(cell_path, sep="\t", index=False, header=False)

    start_cell = 0
    mode = "w+"
    if args.resume and output_path.exists() and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if int(progress.get("cells_total", -1)) != n_cells:
            raise RuntimeError("Existing output has a different cell count; use a separate output directory")
        start_cell = int(progress.get("cells_completed", 0))
        mode = "r+"
    output = np.lib.format.open_memmap(output_path, mode=mode, dtype=np.float32,
                                       shape=(n_cells, embedding_dim))
    autocast_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args.precision)
    started = time.time()

    for chunk_start in tqdm(range(start_cell, n_cells, args.read_chunk_size), desc="Nicheformer cells"):
        chunk_end = min(chunk_start + args.read_chunk_size, n_cells)
        X = adata[chunk_start:chunk_end, :].X
        ids, mask = tokenize_chunk(
            X, source_indices, target_indices, technology_mean,
            context_length, args.technology_token,
        )
        chunk_outputs = []
        for local_start in range(0, len(ids), args.batch_size):
            local_end = min(local_start + args.batch_size, len(ids))
            input_ids = torch.from_numpy(ids[local_start:local_end]).cuda(non_blocking=True)
            attention = torch.from_numpy(mask[local_start:local_end]).cuda(non_blocking=True)
            with torch.inference_mode():
                if autocast_dtype is None:
                    embedded = model.get_embeddings(input_ids=input_ids, attention_mask=attention,
                                                    layer=-1, with_context=False)
                else:
                    with torch.autocast("cuda", dtype=autocast_dtype):
                        embedded = model.get_embeddings(input_ids=input_ids, attention_mask=attention,
                                                        layer=-1, with_context=False)
            chunk_outputs.append(embedded.float().cpu().numpy())
        output[chunk_start:chunk_end] = np.concatenate(chunk_outputs, axis=0)
        output.flush()
        progress = {
            "status": "running" if chunk_end < n_cells else "complete",
            "cells_completed": chunk_end,
            "cells_total": n_cells,
            "elapsed_seconds_this_session": time.time() - started,
            "model": args.model,
            "model_revision": args.revision,
            "embedding_shape": [n_cells, embedding_dim],
            "precision": args.precision,
            "batch_size": args.batch_size,
            "read_chunk_size": args.read_chunk_size,
            "matched_genes": int(len(mapping)),
            "technology_token": args.technology_token,
            "pooling": "official get_embeddings: last layer, context removed, mean over sequence",
        }
        progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    print(json.dumps(progress, indent=2))


if __name__ == "__main__":
    main()

