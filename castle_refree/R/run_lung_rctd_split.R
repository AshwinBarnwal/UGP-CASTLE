#!/usr/bin/env R

# Run matched-reference RCTD (doublet mode) and SPLIT on Xenium Lung L1.

script_file <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  f <- grep("^--file=", args, value = TRUE)
  if (length(f)) return(normalizePath(sub("^--file=", "", f[[1]]), mustWork = TRUE))
  flag <- match("-f", args)
  if (!is.na(flag) && flag < length(args)) {
    return(normalizePath(args[[flag + 1]], mustWork = TRUE))
  }
  for (frame in rev(sys.frames())) {
    if (!is.null(frame$ofile)) return(normalizePath(frame$ofile, mustWork = TRUE))
  }
  stop("Could not determine script location")
}

repo <- normalizePath(file.path(dirname(script_file()), "..", ".."), winslash = "/")
ref_dir <- file.path(repo, "data", "lung_chromium_reference", "rctd_bundle")
spatial_dir <- file.path(repo, "data", "castle_xenium_lung_L1")
result_dir <- file.path(repo, "results", "lung_L1_rctd_split")
dir.create(result_dir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages({
  library(Matrix)
  library(spacexr)
  library(SPLIT)
})

read_vector <- function(path) scan(path, what = character(), sep = "\n", quiet = TRUE)

ref_counts <- readMM(file.path(ref_dir, "reference_counts.mtx"))
ref_genes <- read_vector(file.path(ref_dir, "genes.tsv"))
ref_cells <- read_vector(file.path(ref_dir, "cells.tsv"))
dimnames(ref_counts) <- list(ref_genes, ref_cells)
ref_meta <- read.csv(file.path(ref_dir, "reference_metadata.csv"), check.names = FALSE)
stopifnot(identical(ref_cells, ref_meta$cell_id))

cell_types <- setNames(factor(ref_meta$cell_type), ref_cells)
n_umi_ref <- setNames(as.numeric(ref_meta$nUMI_full), ref_cells)
reference <- Reference(
  counts = ref_counts,
  cell_types = cell_types,
  nUMI = n_umi_ref,
  require_int = FALSE
)

spatial_counts <- readMM(file.path(spatial_dir, "counts.mtx"))
spatial_genes <- read_vector(file.path(spatial_dir, "genes.tsv"))
spatial_cells <- read_vector(file.path(spatial_dir, "cells.tsv"))
dimnames(spatial_counts) <- list(spatial_genes, spatial_cells)
spatial_meta <- read.csv(file.path(spatial_dir, "spatial.csv"), check.names = FALSE)
spatial_meta$cell_id <- as.character(spatial_meta$cell_id)
spatial_meta <- spatial_meta[match(spatial_cells, spatial_meta$cell_id), ]
stopifnot(!anyNA(spatial_meta$cell_id))
coords <- data.frame(x = spatial_meta$x, y = spatial_meta$y, row.names = spatial_cells)
n_umi_spatial <- setNames(Matrix::colSums(spatial_counts), spatial_cells)
keep <- n_umi_spatial >= 5
spatial_counts <- spatial_counts[, keep, drop = FALSE]
coords <- coords[keep, , drop = FALSE]
n_umi_spatial <- n_umi_spatial[keep]

spatial <- SpatialRNA(
  coords = coords,
  counts = spatial_counts,
  nUMI = n_umi_spatial,
  require_int = TRUE
)

class_map <- c(
  "CD4-positive, alpha-beta memory T cell" = "lymphoid",
  "activated CD8-positive, alpha-beta T cell" = "lymphoid",
  "cytotoxic T cell" = "lymphoid",
  "natural T-regulatory cell" = "lymphoid",
  "natural killer cell" = "lymphoid",
  "plasma cell" = "lymphoid",
  "classical monocyte" = "myeloid",
  "conventional dendritic cell" = "myeloid",
  "dendritic cell, human" = "myeloid",
  "granulocyte" = "myeloid",
  "macrophage" = "myeloid",
  "mast cell" = "myeloid",
  "plasmacytoid dendritic cell" = "myeloid",
  "malignant cell" = "epithelial",
  "epithelial cell of lung" = "epithelial",
  "capillary endothelial cell" = "stromal",
  "fibroblast of lung" = "stromal",
  "pericyte" = "stromal",
  "smooth muscle cell" = "stromal"
)
ref_levels <- levels(cell_types)
class_df <- data.frame(
  class = unname(class_map[ref_levels]),
  row.names = ref_levels,
  stringsAsFactors = FALSE
)
if (anyNA(class_df$class)) stop("Missing broad class mapping")

initialized_path <- file.path(result_dir, "rctd_lung_L1_initialized.rds")
completed_path <- file.path(result_dir, "rctd_lung_L1_completed.rds")
if (file.exists(completed_path)) {
  message("Reusing completed RCTD object")
  rctd <- readRDS(completed_path)
} else {
  if (file.exists(initialized_path)) {
    rctd <- readRDS(initialized_path)
  } else {
    rctd <- create.RCTD(
      spatialRNA = spatial,
      reference = reference,
      UMI_min = 5,
      counts_MIN = 5,
      UMI_min_sigma = 100,
      CELL_MIN_INSTANCE = 10,
      max_cores = 4,
      class_df = class_df,
      test_mode = FALSE
    )
    saveRDS(rctd, initialized_path, compress = FALSE)
  }
  # Four workers keep memory comfortably below the 16 GB machine limit while
  # avoiding a roughly fivefold serial runtime on 72k spatial cells.
  rctd@config$max_cores <- 4
  started <- Sys.time()
  rctd <- run.RCTD(rctd, doublet_mode = "doublet")
  elapsed <- as.numeric(difftime(Sys.time(), started, units = "secs"))
  saveRDS(rctd, completed_path, compress = FALSE)
  write.csv(data.frame(elapsed_seconds = elapsed), file.path(result_dir, "rctd_timing.csv"), row.names = FALSE)
}

rctd <- SPLIT::run_post_process_RCTD(rctd)
converted <- SPLIT::convert_rctd_result_to_purify_input(rctd = rctd)
weights <- converted$deconvolution_weights
primary <- converted$primary_cell_type
split_reference <- converted$reference
if (!identical(rownames(weights), names(primary))) primary <- primary[rownames(weights)]

results_df <- rctd@results$results_df
labels <- data.frame(
  cell_id = rownames(weights),
  primary_cell_type = as.character(primary),
  spot_class = as.character(results_df[rownames(weights), "spot_class"]),
  stringsAsFactors = FALSE
)
write.csv(labels, file.path(result_dir, "rctd_primary_labels.csv"), row.names = FALSE)

counts_for_split <- spatial_counts[, rownames(weights), drop = FALSE]
# SPLIT's converter currently returns genes x cell-types, whereas purify()
# expects cell-types x genes. Detect this from the weight/type names rather
# than requiring the reference to contain every spatial-panel gene.
if (all(colnames(weights) %in% colnames(split_reference))) {
  split_reference <- Matrix::t(split_reference)
}
if (!all(colnames(split_reference) %in% rownames(counts_for_split))) {
  stop("Unexpected SPLIT reference orientation or gene names")
}

split_result <- SPLIT::rctd_free_purify(
  counts = counts_for_split,
  deconvolution_weights = weights,
  reference = split_reference,
  primary_cell_type = primary,
  DO_run_in_chunks = TRUE,
  chunk_size = 10000,
  DO_require_sumup_to_one = TRUE,
  DO_output_sce = FALSE,
  DO_remove_residual_contamination = FALSE
)
rownames(split_result$cell_meta) <- split_result$cell_meta$cell_id
split_output <- list(
  purified_counts = split_result$purified_counts,
  cell_meta = split_result$cell_meta,
  run_summary = list(
    dataset = "GSE311609 GSM9509145 NSCLC Lung L1",
    rctd_mode = "doublet",
    reference = "matched lung Chromium FLEX; malignant cells restricted to donor 5/L1",
    cells = ncol(split_result$purified_counts),
    genes = nrow(split_result$purified_counts),
    split_version = as.character(packageVersion("SPLIT")),
    spacexr_version = as.character(packageVersion("spacexr"))
  )
)
saveRDS(split_output, file.path(result_dir, "split_lung_L1.rds"), compress = FALSE)
writeLines(capture.output(sessionInfo()), file.path(result_dir, "session_info.txt"))
message("RCTD + SPLIT completed: ", result_dir)
