"""Colour cached no-NICHE UMAPs by the dataset's published cell labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    folder = args.folder if args.folder.is_absolute() else ROOT / args.folder
    metadata = json.loads((folder / "method.json").read_text(encoding="utf-8"))
    cache = np.load(folder / "plot_data.npz")
    rows = cache["source_rows"]
    source = ad.read_h5ad(ROOT / "data" / "cosmx_frontal_cortex.h5ad", backed="r")
    labels = source.obs["cell_type"].astype(str).to_numpy()[rows]
    names = sorted(np.unique(labels))
    colors = dict(zip(names, sns.color_palette("tab10", len(names))))
    methods = metadata["methods"]
    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 6), squeeze=False)
    for ax, method in zip(axes[0], methods):
        key = f"umap_{method.lower().replace(' ', '_').replace('-', '_')}"
        xy = cache[key]
        for label in names:
            use = labels == label
            color = (0.65, 0.65, 0.65) if label == "Unknown" else colors[label]
            ax.scatter(xy[use, 0], xy[use, 1], s=1.2, c=[color], label=label,
                       linewidths=0, alpha=0.75, rasterized=True)
        ax.set_title(method, weight="bold")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_xticks([])
        ax.set_yticks([])
        sns.despine(ax=ax, left=True, bottom=True)
    handles, legend_labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(0.5, -0.01),
               ncol=min(5, len(names)), frameon=False, markerscale=5)
    fig.suptitle(f"{metadata['cohort']}: UMAP coloured by published cell type",
                 fontsize=16, weight="bold")
    fig.subplots_adjust(top=0.88, bottom=0.18, wspace=0.08)
    fig.savefig(folder / "00_umap_published_cell_types.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(folder / "00_umap_published_cell_types.png")


if __name__ == "__main__":
    main()
