#!/usr/bin/env Rscript

# Paper-style downstream evaluation of SPLIT purified Xenium counts against
# Janesick supervised cell labels. Run from any working directory; paths are
# resolved relative to the project root.

suppressPackageStartupMessages({
  library(Matrix)
  library(Seurat)
  library(igraph)
  library(ggplot2)
  library(patchwork)
})

required <- c("uwot")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) {
  stop("Install required R packages first: install.packages(c(",
       paste(sprintf("'%s'", missing), collapse = ", "), "))")
}

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (!length(script_arg)) stop("Run this file with Rscript so its project paths can be resolved.")
script_path <- normalizePath(sub("^--file=", "", script_arg[[1]]), winslash = "/")
project_root <- normalizePath(file.path(dirname(script_path), "..", ".."), winslash = "/")

split_path <- file.path(project_root, "results", "xenium_rep1_split_full_combined.rds")
labels_path <- file.path(project_root, "data", "xenium_r1_supervised_labels.csv")
castle_cells_path <- file.path(
  project_root, "results", "castle_refree_xenium_breast_beta2", "cells.tsv"
)
output_dir <- file.path(project_root, "results", "split_janesick_evaluation_paperstyle")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

seed <- 42L
target_k <- 19L
leiden_neighbors <- 15L
umap_neighbors <- 15L
resolutions <- seq(0.2, 2.0, by = 0.2)
scale_factor <- 10000
paper_min_counts <- 10
paper_min_genes <- 5
paper_min_cells_per_gene <- 10

for (path in c(split_path, labels_path, castle_cells_path)) {
  if (!file.exists(path)) stop("Required input file not found: ", path)
}

message("[1/10] Loading SPLIT matrix and metadata...")
flush.console()
split_result <- readRDS(split_path)
if (is.null(split_result$purified_counts) || is.null(split_result$cell_meta)) {
  stop("The SPLIT RDS must contain purified_counts and cell_meta.")
}
counts <- split_result$purified_counts
if (!inherits(counts, "sparseMatrix")) counts <- Matrix::Matrix(counts, sparse = TRUE)
counts <- methods::as(counts, "dgCMatrix")
cell_meta <- as.data.frame(split_result$cell_meta)
split_ids <- colnames(counts)
if (is.null(split_ids) || is.null(rownames(cell_meta))) {
  stop("SPLIT counts need column barcodes and cell_meta needs row-name barcodes.")
}
if (anyDuplicated(split_ids)) stop("Duplicate cell barcodes in the SPLIT matrix.")
if (!identical(split_ids, rownames(cell_meta))) {
  stop("SPLIT matrix column barcodes and cell_meta row names are not in identical order.")
}

labels <- read.csv(labels_path, colClasses = "character", check.names = FALSE)
if (!all(c("Barcode", "Cluster") %in% colnames(labels))) {
  stop("Ground-truth CSV must have Barcode and Cluster columns.")
}
labels$Barcode <- trimws(labels$Barcode)
labels$Cluster <- trimws(labels$Cluster)
if (anyDuplicated(labels$Barcode)) stop("Duplicate Barcode values in ground-truth CSV.")
is_unlabeled <- is.na(labels$Cluster) | !nzchar(labels$Cluster) |
  tolower(labels$Cluster) %in% c("unlabeled", "unlabelled")
labeled <- labels[!is_unlabeled, c("Barcode", "Cluster"), drop = FALSE]
if (length(unique(labeled$Cluster)) != target_k) {
  stop("Expected K=", target_k, " labeled types in CSV; found ",
       length(unique(labeled$Cluster)), ".")
}

castle_ids <- trimws(readLines(castle_cells_path, warn = FALSE))
if (anyDuplicated(castle_ids)) stop("Duplicate cell barcodes in CASTLE cells.tsv.")

message("[2/10] Applying paper-style cell and gene QC...")
flush.console()
# Apply the paper's Xenium QC thresholds to the purified matrix, then evaluate
# only labeled barcodes present in both SPLIT and CASTLE.
cell_totals <- Matrix::colSums(counts)
detected_genes <- Matrix::colSums(counts > 0)
qc_cells <- cell_totals >= paper_min_counts & detected_genes >= paper_min_genes
if (!any(qc_cells)) stop("No cells remain after the paper-style cell QC filters.")
gene_keep <- Matrix::rowSums(counts[, qc_cells, drop = FALSE] > 0) >=
  paper_min_cells_per_gene
if (sum(gene_keep) < 2L) stop("Fewer than two genes remain after gene QC.")
counts <- counts[gene_keep, qc_cells, drop = FALSE]
split_ids_qc <- colnames(counts)

gt_ids <- unique(labeled$Barcode)
shared_ids <- gt_ids[gt_ids %in% split_ids_qc & gt_ids %in% castle_ids]
if (!length(shared_ids)) stop("No labeled barcodes are shared by SPLIT and CASTLE after QC.")
labels_by_id <- setNames(labeled$Cluster, labeled$Barcode)
y <- unname(labels_by_id[shared_ids])
if (length(unique(y)) != target_k) {
  stop("The shared, QC-filtered set does not contain all ", target_k,
       " ground-truth cell types.")
}

counts <- counts[, shared_ids, drop = FALSE]
if (any(!is.finite(counts@x)) || any(counts@x < 0)) {
  stop("Purified counts contain negative or non-finite values.")
}

message(sprintf("[3/10] Preparing Seurat object for %s shared cells...",
               format(length(shared_ids), big.mark = ",")))
flush.console()
# Paper-style expression processing: library-size log normalization, centered
# PCA without per-gene variance scaling, and 15-neighbor graphs for Leiden and UMAP. The
# Leiden sweep and K-means scores are the requested evaluation extensions.
obj <- Seurat::CreateSeuratObject(
  counts = counts,
  meta.data = cell_meta[shared_ids, , drop = FALSE],
  assay = "Xenium",
  project = "SPLIT_Janesick"
)
obj$ground_truth <- factor(y, levels = sort(unique(y)))
message("[4/10] Log-normalizing counts...")
flush.console()
obj <- Seurat::NormalizeData(
  obj, assay = "Xenium", normalization.method = "LogNormalize",
  scale.factor = scale_factor, verbose = FALSE
)
features <- rownames(obj[["Xenium"]])
message("[5/10] Centering genes (no variance scaling)...")
flush.console()
obj <- Seurat::ScaleData(
  obj, assay = "Xenium", features = features,
  do.center = TRUE, do.scale = FALSE, verbose = FALSE
)
n_pcs <- min(50L, length(features) - 1L, ncol(obj) - 1L)
if (n_pcs < 2L) stop("Not enough cells or genes to calculate PCA.")
message(sprintf("[6/10] Running PCA (%d PCs)...", n_pcs))
flush.console()
obj <- Seurat::RunPCA(
  obj, assay = "Xenium", features = features, npcs = n_pcs,
  approx = TRUE, verbose = FALSE
)
message(sprintf("[7/10] Building %d-neighbor graph with Annoy...", leiden_neighbors))
flush.console()
obj <- Seurat::FindNeighbors(
  obj, reduction = "pca", dims = seq_len(n_pcs),
  k.param = leiden_neighbors, nn.method = "annoy", n.trees = 50,
  compute.SNN = TRUE, graph.name = c("split_nn", "split_snn"),
  verbose = TRUE
)
message("[8/10] Computing UMAP (this can take a while)...")
flush.console()
obj <- Seurat::RunUMAP(
  obj, reduction = "pca", dims = seq_len(n_pcs),
  n.neighbors = umap_neighbors, min.dist = 0.5, metric = "euclidean",
  umap.method = "uwot", seed.use = seed, verbose = FALSE
)

umap_xy <- Seurat::Embeddings(obj, "umap")

message("[9/10] Running K-means and Leiden resolution sweep...")
flush.console()
# K-means uses K=19, matching the ground-truth class count.
set.seed(seed)
km <- stats::kmeans(pca, centers = target_k, nstart = 10, iter.max = 100)
kmeans_labels <- as.character(km$cluster)

# Seurat/Annoy builds the graph used for the Leiden resolution sweep.
graph <- igraph::graph_from_adjacency_matrix(
  obj[["split_snn"]], mode = "undirected", weighted = TRUE,
  diag = FALSE, sparse = TRUE
)
if (igraph::vcount(graph) != length(y)) stop("Leiden graph size does not match labels.")

comb2 <- function(x) x * (x - 1) / 2
adjusted_rand <- function(truth, prediction) {
  tab <- table(truth, prediction)
  n <- sum(tab)
  if (n < 2L) return(NA_real_)
  sum_cells <- sum(comb2(as.numeric(tab)))
  sum_rows <- sum(comb2(rowSums(tab)))
  sum_cols <- sum(comb2(colSums(tab)))
  total <- comb2(n)
  expected <- sum_rows * sum_cols / total
  maximum <- (sum_rows + sum_cols) / 2
  if (maximum == expected) return(1)
  (sum_cells - expected) / (maximum - expected)
}
normalized_mi <- function(truth, prediction) {
  tab <- table(truth, prediction)
  pxy <- tab / sum(tab)
  px <- rowSums(pxy)
  py <- colSums(pxy)
  nz <- which(pxy > 0, arr.ind = TRUE)
  mi <- sum(pxy[nz] * log(pxy[nz] / (px[nz[, 1]] * py[nz[, 2]])))
  hx <- -sum(px[px > 0] * log(px[px > 0]))
  hy <- -sum(py[py > 0] * log(py[py > 0]))
  denom <- (hx + hy) / 2
  if (denom == 0) return(1)
  mi / denom
}

leiden_by_resolution <- vector("list", length(resolutions))
resolution_rows <- vector("list", length(resolutions))
leiden_progress <- utils::txtProgressBar(min = 0, max = length(resolutions), style = 3)
for (i in seq_along(resolutions)) {
  resolution <- resolutions[[i]]
  set.seed(seed)
  leiden <- igraph::cluster_leiden(
    graph, objective_function = "modularity", weights = igraph::E(graph)$weight,
    resolution_parameter = resolution, n_iterations = 2
  )
  predicted <- as.character(igraph::membership(leiden))
  leiden_by_resolution[[i]] <- predicted
  resolution_rows[[i]] <- data.frame(
    resolution = resolution,
    n_clusters = length(unique(predicted)),
    ARI = adjusted_rand(y, predicted),
    NMI = normalized_mi(y, predicted)
  )
  message(sprintf("Leiden resolution %.1f: %d clusters", resolution,
                  length(unique(predicted))))
  utils::setTxtProgressBar(leiden_progress, i)
}
close(leiden_progress)
resolution_metrics <- do.call(rbind, resolution_rows)
best_ari_idx <- which.max(resolution_metrics$ARI)
best_nmi_idx <- which.max(resolution_metrics$NMI)
leiden_nmi_labels <- leiden_by_resolution[[best_nmi_idx]]

make_plot <- function(predicted, predicted_title, path) {
  plot_data <- data.frame(
    UMAP1 = umap_xy[, 1], UMAP2 = umap_xy[, 2],
    clusters = factor(predicted), ground_truth = factor(y)
  )
  p1 <- ggplot(plot_data, aes(UMAP1, UMAP2, color = clusters)) +
    geom_point(size = 0.18, alpha = 0.85) +
    labs(title = predicted_title, color = "Cluster") +
    theme_classic(base_size = 11) +
    theme(axis.title = element_blank(), axis.text = element_blank(),
          axis.ticks = element_blank(), legend.position = "right")
  p2 <- ggplot(plot_data, aes(UMAP1, UMAP2, color = ground_truth)) +
    geom_point(size = 0.18, alpha = 0.85) +
    labs(title = "Ground-truth cell type", color = "Cell type") +
    theme_classic(base_size = 11) +
    theme(axis.title = element_blank(), axis.text = element_blank(),
          axis.ticks = element_blank(), legend.position = "right")
  ggplot2::ggsave(path, p1 + p2, width = 16, height = 8, dpi = 220,
                  limitsize = FALSE)
}

matched <- data.frame(
  Barcode = shared_ids,
  Cluster = y,
  split_matrix_column = match(shared_ids, split_ids),
  kmeans_cluster = kmeans_labels,
  leiden_cluster_best_nmi = leiden_nmi_labels,
  umap_1 = umap_xy[, 1],
  umap_2 = umap_xy[, 2],
  stringsAsFactors = FALSE
)

metrics <- data.frame(
  n_ground_truth_labeled = nrow(labeled),
  n_split_cells = ncol(split_result$purified_counts),
  n_castle_cells = length(castle_ids),
  n_common_labeled_after_qc = length(shared_ids),
  n_split_labeled_missing = sum(!gt_ids %in% split_ids),
  unlabeled_rows_excluded = sum(is_unlabeled),
  n_ground_truth_classes = target_k,
  n_genes_after_qc = nrow(counts),
  n_pcs = n_pcs,
  leiden_neighbors = leiden_neighbors,
  umap_neighbors = umap_neighbors,
  kmeans_K = target_k,
  kmeans_ARI = adjusted_rand(y, kmeans_labels),
  kmeans_NMI = normalized_mi(y, kmeans_labels),
  leiden_best_ARI = resolution_metrics$ARI[[best_ari_idx]],
  leiden_best_ARI_resolution = resolution_metrics$resolution[[best_ari_idx]],
  leiden_best_NMI = resolution_metrics$NMI[[best_nmi_idx]],
  leiden_best_NMI_resolution = resolution_metrics$resolution[[best_nmi_idx]],
  leiden_ARI_at_best_NMI_resolution = resolution_metrics$ARI[[best_nmi_idx]],
  leiden_NMI_at_best_NMI_resolution = resolution_metrics$NMI[[best_nmi_idx]]
)

message("[10/10] Writing tables and UMAP plots...")
flush.console()
write.csv(matched, file.path(output_dir, "matched_labels.csv"), row.names = FALSE)
write.csv(
  data.frame(Barcode = shared_ids, Cluster = y, stringsAsFactors = FALSE),
  file.path(output_dir, "shared_labels_for_castle.csv"), row.names = FALSE
)
write.csv(metrics, file.path(output_dir, "metrics.csv"), row.names = FALSE)
write.csv(resolution_metrics,
          file.path(output_dir, "leiden_resolution_metrics.csv"), row.names = FALSE)
writeLines(shared_ids, file.path(output_dir, "shared_labeled_barcodes.txt"))
make_plot(kmeans_labels, paste0("K-means clusters (K = ", target_k, ")"),
          file.path(output_dir, "umap_kmeans_vs_ground_truth.png"))
make_plot(
  leiden_nmi_labels,
  sprintf("Leiden clusters (best-NMI resolution %.1f)",
          resolution_metrics$resolution[[best_nmi_idx]]),
  file.path(output_dir, "umap_leiden_vs_ground_truth.png")
)

cat("\nSPLIT evaluation results\n------------------------\n")
cat("Unlabeled ground-truth rows excluded:", sum(is_unlabeled), "\n")
cat("Labeled cells shared with CASTLE after QC:", length(shared_ids), "\n")
cat("Labeled ground-truth barcodes missing from SPLIT:",
    sum(!gt_ids %in% split_ids), "\n")
cat("Ground-truth cell types (K):", target_k, "\n")
cat("Genes / PCs used:", nrow(counts), "/", n_pcs, "\n")
cat(sprintf("K-means ARI/NMI: %.6f / %.6f\n",
            metrics$kmeans_ARI, metrics$kmeans_NMI))
cat(sprintf("Best Leiden ARI: %.6f at resolution %.1f\n",
            metrics$leiden_best_ARI, metrics$leiden_best_ARI_resolution))
cat(sprintf("Best Leiden NMI: %.6f at resolution %.1f\n",
            metrics$leiden_best_NMI, metrics$leiden_best_NMI_resolution))
cat(sprintf("Leiden at best-NMI resolution ARI/NMI: %.6f / %.6f\n",
            metrics$leiden_ARI_at_best_NMI_resolution,
            metrics$leiden_NMI_at_best_NMI_resolution))
cat("\nSaved results to:", normalizePath(output_dir, winslash = "/"), "\n")
