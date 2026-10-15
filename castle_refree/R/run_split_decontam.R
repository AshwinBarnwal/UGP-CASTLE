#!/usr/bin/env R

# Run RCTD (doublet mode) + SPLIT on the decontamination bundles (null/spill/niche)
# written by simulate_contamination.py, following the SPLIT Xenium vignette
# (vignettes/Run_RCTD_and_SPLIT_on_Xenium.Rmd, SPLIT v0.3.0):
#
#   SPLIT_default : purify(DO_purify_singlets = TRUE,
#                          DO_remove_residual_contamination = TRUE)
#   SPLIT_spatial : neighbourhood-aware SPLIT; cells are purified only when their
#                   spatial neighbourhood (k = 20, pruned at 15 um) carries their
#                   secondary type (neighborhood_weights_second_type > 0.05)
#
# Outputs per mode and variant: purified.mtx (genes x cells), purified_cells.tsv,
# purified_genes.tsv and cell_meta.csv, converted for scoring by
# castle_refree/scripts/import_split_decontam.py.
#
# Usage (from the repository root):
#   Rscript castle_refree/R/run_split_decontam.R [sim_root] [fits_root] [max_cores]

suppressPackageStartupMessages({
  library(Matrix)
  library(spacexr)
  library(SPLIT)
  library(Seurat)
  library(dplyr)
})

args <- commandArgs(trailingOnly = TRUE)
sim_root <- if (length(args) >= 1) args[[1]] else "data/castle_decontam_sim"
fits_root <- if (length(args) >= 2) args[[2]] else "results/decontamination/seed7"
max_cores <- if (length(args) >= 3) as.integer(args[[3]]) else 4L
variants <- c("null", "spill", "niche")

ref <- readRDS("data/chromium_reference/rctd_reference.rds")
reference <- ref$reference_object
class_df <- ref$class_df

read_bundle <- function(dir) {
  counts <- Matrix::readMM(file.path(dir, "counts.mtx"))
  genes <- read.delim(file.path(dir, "genes.tsv"), header = FALSE, colClasses = "character")[[1]]
  cells <- read.delim(file.path(dir, "cells.tsv"), header = FALSE, colClasses = "character")[[1]]
  counts <- as(counts, "CsparseMatrix")
  if (nrow(counts) != length(genes)) counts <- Matrix::t(counts)
  dimnames(counts) <- list(genes, cells)
  spatial <- read.csv(file.path(dir, "spatial.csv"), colClasses = c(cell_id = "character"))
  rownames(spatial) <- spatial$cell_id
  list(counts = round(counts), coords = spatial[cells, c("x", "y")])
}

write_result <- function(counts, meta, out_dir, info) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  Matrix::writeMM(as(counts, "CsparseMatrix"), file.path(out_dir, "purified.mtx"))
  writeLines(colnames(counts), file.path(out_dir, "purified_cells.tsv"))
  writeLines(rownames(counts), file.path(out_dir, "purified_genes.tsv"))
  write.csv(meta, file.path(out_dir, "cell_meta.csv"))
  writeLines(jsonlite::toJSON(info, auto_unbox = TRUE, pretty = TRUE), file.path(out_dir, "split_run.json"))
}

for (variant in variants) {
  message("== ", variant)
  bundle <- read_bundle(file.path(sim_root, variant))
  rctd_path <- file.path(fits_root, "SPLIT_rctd", paste0(variant, ".rds"))
  started <- Sys.time()
  if (file.exists(rctd_path)) {
    rctd <- readRDS(rctd_path)
  } else {
    puck <- SpatialRNA(bundle$coords, bundle$counts, colSums(bundle$counts))
    rctd <- create.RCTD(
      puck, reference,
      UMI_min = 10, counts_MIN = 10, UMI_min_sigma = 100,
      CELL_MIN_INSTANCE = 25, class_df = class_df, max_cores = max_cores
    )
    rctd <- run.RCTD(rctd, doublet_mode = "doublet")
    rctd <- SPLIT::run_post_process_RCTD(rctd)
    dir.create(dirname(rctd_path), recursive = TRUE, showWarnings = FALSE)
    saveRDS(rctd, rctd_path)
  }
  rctd_minutes <- as.numeric(difftime(Sys.time(), started, units = "mins"))

  # Raw Seurat object with RCTD annotations and spatial coordinates, as in the vignette.
  xe <- CreateSeuratObject(counts = bundle$counts, assay = "Xenium")
  embedding <- as.matrix(bundle$coords)
  colnames(embedding) <- c("spatial_1", "spatial_2")
  xe[["spatial"]] <- CreateDimReducObject(embeddings = embedding, key = "spatial_", assay = "Xenium")
  xe <- AddMetaData(xe, rctd@results$results_df)
  xe <- subset(xe, cells = intersect(colnames(xe), rownames(rctd@results$results_df)))
  xe <- subset(xe, subset = nCount_Xenium >= 10)

  res <- SPLIT::purify(
    counts = GetAssayData(xe, assay = "Xenium", layer = "counts"),
    rctd = rctd,
    DO_purify_singlets = TRUE,
    DO_remove_residual_contamination = TRUE
  )
  info <- list(
    variant = variant, split_version = as.character(packageVersion("SPLIT")),
    spacexr_version = as.character(packageVersion("spacexr")),
    rctd_minutes = rctd_minutes, cells_input = ncol(bundle$counts)
  )
  write_result(res$purified_counts, res$cell_meta, file.path(fits_root, "SPLIT_default", variant),
               c(info, mode = "default"))

  xe_purified <- CreateSeuratObject(counts = res$purified_counts, meta.data = res$cell_meta, assay = "Xenium")
  neighbourhood <- SPLIT::build_spatial_network(
    xe, reduction = "spatial", dims = 1:2, DO_prune = TRUE, rad_pruning = 15, k_knn = 20
  )
  neighbourhood <- SPLIT::add_spatial_metric(spatial_neighborhood = neighbourhood, rctd = rctd)
  xe <- AddMetaData(xe, SPLIT::neighborhood_analysis_to_metadata(neighbourhood))
  balanced <- SPLIT::balance_raw_and_purified_data_by_score(
    xe_raw = xe, xe_purified = xe_purified, threshold = 0.05,
    score_name = "neighborhood_weights_second_type"
  )
  write_result(GetAssayData(balanced, assay = "Xenium", layer = "counts"), balanced@meta.data,
               file.path(fits_root, "SPLIT_spatial", variant), c(info, mode = "spatial", threshold = 0.05))
  message("done ", variant, " (RCTD ", round(rctd_minutes, 1), " min)")
}
