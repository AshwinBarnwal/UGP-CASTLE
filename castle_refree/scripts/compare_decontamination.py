"""Final decontamination comparison: CASTLE versions, resolVI, SPLIT, nucleus-only and raw.

Real-data tests are recomputed for every method on the *same* cells and genes
(the intersection of what all methods cover, scored crop cells only):

- removal: share of counts removed (over-correction guard);
- nucleus: per-cell correlation of purified with nucleus-only profiles;
- off-type: share of other cell types' marker transcripts, split by whether a
  cell touches a different type;
- clustering: K-means on PCA of the purified counts vs. Janesick labels (ARI/NMI);
- genes per cell: median genes with a rounded count of at least one, and the
  share of near-empty cells (< 10 counts) after correction;
- Chromium similarity: cosine similarity of per-type log pseudo-bulk profiles
  with the Chromium scFFPE-seq reference (Bilous et al. 2026). The Janesick
  labels were transferred from this reference, so reference-based methods
  (SPLIT) are favoured by construction;
- neighbourhood contamination test (Mitchel et al.; Bilous et al. 2026): for
  target type i and contaminating type j, a balanced logistic regression
  predicts from z-scored log expression whether an i cell lies within 15 um of
  a j cell; the score is how many of j's 20 Wilcoxon markers (defined once on
  raw counts) rank among the 20 largest |coefficients|. Lower is cleaner.

Semi-synthetic tests (spill/niche) come from ``evaluate_decontamination.py`` and
exist only for methods refitted on those bundles (CASTLE versions, resolVI).
SPLIT uses its existing purified counts for the real slide; nucleus-only and raw
need no fitting. Labels are used for scoring only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

SCRIPT_DIR = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ev = _load("evaluate_decontamination", SCRIPT_DIR / "evaluate_decontamination.py")
split_io = _load(
    "evaluate_split_janesick_labels",
    SCRIPT_DIR.parent / "src" / "castle_refree" / "evaluate_split_janesick_labels.py",
)
crop_tools = _load("make_xenium_crop", SCRIPT_DIR / "make_xenium_crop.py")


def read_xlsx_sheet(path: Path, index: int) -> pd.DataFrame:
    """First-row-header sheet from an .xlsx using only the standard library."""
    with zipfile.ZipFile(path) as book:
        shared = [
            "".join(t.text or "" for t in item.iter(f"{XLSX_NS}t"))
            for item in ET.fromstring(book.read("xl/sharedStrings.xml"))
        ]
        rows = []
        for row in ET.fromstring(book.read(f"xl/worksheets/sheet{index}.xml")).iter(f"{XLSX_NS}row"):
            values = []
            for cell in row.findall(f"{XLSX_NS}c"):
                node = cell.find(f"{XLSX_NS}v")
                value = None if node is None else node.text
                if cell.get("t") == "s" and value is not None:
                    value = shared[int(value)]
                values.append(value)
            rows.append(values)
    return pd.DataFrame(rows[1:], columns=rows[0])


def chromium_pseudobulk(h5_path: Path, annotation_xlsx: Path, genes: np.ndarray) -> pd.DataFrame:
    """Summed Chromium counts per coarse cell type over the given panel genes."""
    annotation = read_xlsx_sheet(annotation_xlsx, 1).set_index("Barcode")["Annotation"]
    with h5py.File(h5_path, "r") as handle:
        m = handle["matrix"]
        counts = sparse.csc_matrix(
            (m["data"][:], m["indices"][:], m["indptr"][:]), shape=tuple(m["shape"][:])
        )  # features x cells
        names = m["features/name"][:].astype(str)
        barcodes = m["barcodes"][:].astype(str)
    coarse = annotation.reindex(barcodes).map(crop_tools.COARSE_TYPES).to_numpy()
    first = pd.Series(np.arange(len(names)), index=names)
    first = first[~first.index.duplicated(keep="first")]  # gene symbols repeat in the 10x feature list
    rows = first.reindex(genes).fillna(-1).to_numpy(dtype=np.int64)
    present = rows >= 0
    profiles = {}
    for t in pd.unique(coarse[pd.notna(coarse)]):
        summed = np.asarray(counts[:, np.flatnonzero(coarse == t)].sum(axis=1)).ravel()
        profile = np.full(len(genes), np.nan)
        profile[present] = summed[rows[present]]
        profiles[t] = profile
    return pd.DataFrame(profiles, index=genes).T


def log_profile(summed: np.ndarray) -> np.ndarray:
    return np.log1p(1e4 * summed / max(np.nansum(summed), 1e-9))


def chromium_similarity(
    purified: np.ndarray, types: np.ndarray, reference: pd.DataFrame
) -> dict[str, float]:
    usable = ~reference.isna().any(axis=0).to_numpy()
    scores = {}
    for t in reference.index:
        rows = types == t
        if rows.sum() < 20:
            continue
        a = log_profile(purified[rows][:, usable].sum(axis=0))
        b = log_profile(reference.loc[t].to_numpy()[usable])
        scores[t] = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
    return scores


def wilcoxon_markers(raw: np.ndarray, types: np.ndarray, per_type: int = 20) -> dict[str, np.ndarray]:
    """Top genes per type by the Wilcoxon rank-sum statistic (type vs. all others)."""
    expression = ev.lognorm(raw)
    markers = {}
    for t in np.unique(types):
        if t == "Other":
            continue
        inside = types == t
        statistic = mannwhitneyu(expression[inside], expression[~inside], axis=0).statistic
        markers[t] = np.argsort(-statistic / (inside.sum() * (~inside).sum()))[:per_type]
    return markers


def neighbourhood_contamination(
    purified: np.ndarray, types: np.ndarray, near_type: dict[str, np.ndarray],
    markers: dict[str, np.ndarray], min_cells: int = 100,
) -> dict[tuple[str, str], int]:
    """Markers of j among the top-20 |coefficients| predicting proximity to j in type i."""
    expression = ev.lognorm(purified)
    scores = {}
    for target in markers:
        rows = np.flatnonzero(types == target)
        if rows.size == 0:
            continue
        features = expression[rows]
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-9)
        for source in markers:
            if source == target:
                continue
            y = near_type[source][rows]
            if y.sum() < min_cells or (~y).sum() < min_cells:
                continue
            model = LogisticRegression(max_iter=500, class_weight="balanced")
            model.fit(features, y)
            top = np.argsort(-np.abs(model.coef_[0]))[:20]
            scores[(target, source)] = int(np.isin(top, markers[source]).sum())
    return scores


def clustering_scores(x: np.ndarray, labels: np.ndarray, seeds: tuple[int, ...]) -> tuple[float, float]:
    embedding = PCA(n_components=30, random_state=0).fit_transform(ev.lognorm(x))
    k = len(np.unique(labels))
    ari, nmi = [], []
    for seed in seeds:
        predicted = KMeans(n_clusters=k, n_init=3, random_state=seed).fit_predict(embedding)
        ari.append(adjusted_rand_score(labels, predicted))
        nmi.append(normalized_mutual_info_score(labels, predicted))
    return float(np.mean(ari)), float(np.mean(nmi))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-root", type=Path, default=Path("data/castle_decontam_sim"))
    parser.add_argument("--fits", type=Path, default=Path("results/decontamination/seed7"))
    parser.add_argument("--split-export", type=Path, default=Path("results/split_janesick_python_export"))
    parser.add_argument("--labels", type=Path, default=Path("data/xenium_r1_supervised_labels.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/decontamination/comparison_seed7.csv"))
    parser.add_argument("--marker-genes", type=int, default=5)
    parser.add_argument("--chromium-h5", type=Path, default=Path("data/chromium_reference/Chromium_filtered_feature_bc_matrix.h5"))
    parser.add_argument("--chromium-annotation", type=Path, default=Path("data/chromium_reference/Cell_Barcode_Type_Matrices.xlsx"))
    parser.add_argument("--neighbour-radius", type=float, default=15.0, help="um, as in Bilous et al.")
    args = parser.parse_args()

    null_bundle = args.sim_root / "null"
    bundle_cells = ev.read_list(null_bundle / "cells.tsv")
    bundle_genes = ev.read_list(null_bundle / "genes.tsv")
    raw_all = ev.read_bundle_matrix(null_bundle, "counts.mtx", bundle_cells, bundle_genes)
    nucleus_all = ev.read_bundle_matrix(null_bundle, "counts_nucleus.mtx", bundle_cells, bundle_genes)

    # Purified counts per method, as (cells, genes, matrix).
    methods: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        "raw": (bundle_cells, bundle_genes, raw_all),
        "nucleus_only": (bundle_cells, bundle_genes, nucleus_all),
    }
    split_counts, split_genes, split_cells = split_io.load_export(args.split_export)
    in_crop = pd.Index(split_cells).isin(bundle_cells)
    methods["SPLIT"] = (split_cells[in_crop], split_genes, split_counts[in_crop].toarray())
    fitted = sorted(
        p.name for p in args.fits.iterdir()
        if (p / "null" / "castle_embeddings_and_qc.npz").exists() and not p.name.endswith("_twin")
    )
    for name in fitted:
        fit = ev.Fit(args.fits / name / "null", null_bundle)
        methods[name] = (fit.cells, fit.genes, fit.purified)

    # Common evaluation set: scored crop cells and genes every method covers.
    crop = pd.read_csv(null_bundle / "crop_cells.csv", dtype={"cell_id": str}).set_index("cell_id")
    cells = bundle_cells[crop.loc[bundle_cells, "score"].to_numpy(dtype=bool)]
    genes = bundle_genes
    for c, g, _ in methods.values():
        cells = cells[pd.Index(cells).isin(c)]
        genes = genes[pd.Index(genes).isin(g)]

    def aligned(entry: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        c, g, m = entry
        return m[np.ix_(pd.Index(c).get_indexer(cells), pd.Index(g).get_indexer(genes))]

    raw = aligned(methods["raw"])
    nucleus = aligned(methods["nucleus_only"])
    types = crop.loc[cells, "coarse_type"].to_numpy()
    fine = (
        pd.read_csv(args.labels, encoding="utf-8-sig", dtype={"Barcode": str})
        .set_index("Barcode")["Cluster"].reindex(cells).fillna("Unlabeled").to_numpy()
    )
    labelled = types != "Other"
    markers = ev.marker_sets(nucleus[labelled], types[labelled], genes, args.marker_genes)
    edges = pd.read_csv(null_bundle / "edges_contact.csv", dtype={"cell_i": str, "cell_j": str})
    type_of = crop["coarse_type"]
    ti, tj = edges["cell_i"].map(type_of), edges["cell_j"].map(type_of)
    hetero_edges = edges[(ti != tj) & (ti != "Other") & (tj != "Other") & ti.notna() & tj.notna()]
    hetero = pd.Index(cells).isin(pd.concat([hetero_edges["cell_i"], hetero_edges["cell_j"]]).unique())
    enough = (nucleus.sum(axis=1) >= 10) & (raw.sum(axis=1) >= 20)
    clusterable = fine != "Unlabeled"

    reference = chromium_pseudobulk(args.chromium_h5, args.chromium_annotation, genes)
    wilcoxon = wilcoxon_markers(raw, types)
    # Proximity of each evaluated cell to each type, using every crop cell as a
    # potential neighbour (centroid distance <= radius).
    spatial = pd.read_csv(null_bundle / "spatial.csv", dtype={"cell_id": str}).set_index("cell_id")
    crop_ids = spatial.index.to_numpy()
    crop_types = crop.loc[crop_ids, "coarse_type"].to_numpy()
    tree = cKDTree(spatial[["x", "y"]].to_numpy())
    eval_xy = spatial.loc[cells, ["x", "y"]].to_numpy()
    eval_pos = pd.Index(crop_ids).get_indexer(cells)
    neighbour_lists = tree.query_ball_point(eval_xy, r=args.neighbour_radius)
    near_type = {
        t: np.array([
            any(crop_types[k] == t for k in found if k != self_index)
            for found, self_index in zip(neighbour_lists, eval_pos)
        ])
        for t in wilcoxon
    }

    simulated = {}
    summary_path = args.fits.parent / "summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        seed = args.fits.name.removeprefix("seed")
        if "seed" in summary and seed.isdigit():
            summary = summary[summary["seed"] == int(seed)]
        simulated = summary.drop_duplicates("model", keep="last").set_index("model").to_dict("index")
    # Comparators fitted outside the CASTLE suite (resolVI, SPLIT modes) are scored here.
    for name in fitted:
        if name in simulated:
            continue
        if all((args.fits / name / v / "castle_embeddings_and_qc.npz").exists() for v in ("null", "spill", "niche")):
            simulated[name] = ev.evaluate(args.fits / name, args.sim_root.resolve(), name, args.marker_genes)

    rows = []
    for name, entry in methods.items():
        purified = aligned(entry)
        row: dict[str, object] = {"method": name, "cells": int(len(cells)), "genes": int(len(genes))}
        removed = 1 - purified.sum(axis=1) / np.maximum(raw.sum(axis=1), 1e-9)
        row["median_cell_removed"] = float(np.median(removed))
        row["total_removed"] = float(1 - purified.sum() / raw.sum())
        if name != "nucleus_only":
            r_raw = ev.rowwise_pearson(ev.lognorm(raw[enough]), ev.lognorm(nucleus[enough]))
            r_pur = ev.rowwise_pearson(ev.lognorm(purified[enough]), ev.lognorm(nucleus[enough]))
            row["nucleus_r_gain"] = float((r_pur - r_raw).mean())
            row["nucleus_fraction_cells_improved"] = float(np.mean(r_pur > r_raw))
        off = ev.off_type_fraction(purified, types, markers)
        row["offtype_hetero"] = float(np.nanmean(off[labelled & hetero]))
        row["offtype_homo"] = float(np.nanmean(off[labelled & ~hetero]))
        row["kmeans_ari"], row["kmeans_nmi"] = clustering_scores(
            purified[clusterable], fine[clusterable], seeds=(0, 1, 2)
        )
        row["median_genes_per_cell"] = float(np.median((purified >= 0.5).sum(axis=1)))
        row["near_empty_fraction"] = float(np.mean(purified.sum(axis=1) < 10))
        similarity = chromium_similarity(purified, types, reference)
        row["chromium_cosine_mean"] = float(np.mean(list(similarity.values())))
        row["chromium_cosine_T_cell"] = similarity.get("T_cell", np.nan)
        contamination = neighbourhood_contamination(purified, types, near_type, wilcoxon)
        row["contam_markers_mean"] = float(np.mean(list(contamination.values())))
        row["contam_T_cell_from_Epithelial"] = contamination.get(("T_cell", "Epithelial"), np.nan)
        row["contam_pairs"] = len(contamination)
        for key in (
            "spill_eta_delta_spearman", "spill_injected_removed_fraction", "spill_cell_pearson",
            "spill_entry_pearson", "niche_extra_removed_fraction", "niche_baseline_removed_rate",
        ):
            value = simulated.get(name, {}).get(key, np.nan)
            row[key] = float(value) if value is not None else np.nan
        rows.append(row)
        print(json.dumps(row), flush=True)

    table = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
