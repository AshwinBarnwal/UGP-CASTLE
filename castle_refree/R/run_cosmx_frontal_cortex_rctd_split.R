#!/usr/bin/env R

# End-to-end RCTD doublet decomposition and SPLIT purification for the
# official CosMx Human Frontal Cortex S3 release. Intermediate RDS files make
# the multi-hour run restartable.

script_file <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  f <- grep("^--file=", args, value = TRUE)
  if (length(f)) return(normalizePath(sub("^--file=", "", f[[1]]), mustWork = TRUE))
  stop("Run this file with Rterm/Rscript")
}

repo <- normalizePath(file.path(dirname(script_file()), "..", ".."), winslash = "/")
spatial_dir <- file.path(repo, "data", "cosmx_frontal_cortex")
reference_dir <- file.path(repo, "data", "gse104276_reference")
result_dir <- file.path(repo, "results", "cosmx_frontal_cortex_rctd_split")
dir.create(result_dir, recursive = TRUE, showWarnings = FALSE)

# Pick up the sibling user library used by this portable R installation.
portable_user_lib <- file.path(dirname(R.home()), "library", paste(R.version$major, sub("\\..*$", "", R.version$minor), sep = "."))
if (dir.exists(portable_user_lib)) .libPaths(c(portable_user_lib, .libPaths()))

suppressPackageStartupMessages({
  library(Matrix)
  library(data.table)
  library(readxl)
  library(spacexr)
  library(SPLIT)
})

read_vector <- function(path) scan(path, what = character(), sep = "\n", quiet = TRUE)

reference_path <- file.path(result_dir, "gse104276_reference.rds")
if (file.exists(reference_path)) {
  message("Reusing prepared GSE104276 RCTD reference")
  reference <- readRDS(reference_path)
} else {
  count_path <- file.path(
    reference_dir, "GSE104276_all_pfc_2394_UMI_count_NOERCC.xls.gz"
  )
  workbook <- file.path(
    reference_dir, "GSE104276_readme_sample_barcode.xlsx"
  )
  reference_cells <- strsplit(readLines(gzfile(count_path), n = 1), "\t", fixed = TRUE)[[1]]
  raw <- fread(count_path, skip = 1, header = FALSE, showProgress = TRUE)
  reference_genes <- as.character(raw[[1]])
  dense_counts <- as.matrix(raw[, -1, with = FALSE])
  rm(raw)
  storage.mode(dense_counts) <- "numeric"
  ref_counts <- Matrix(dense_counts, sparse = TRUE)
  rm(dense_counts)
  dimnames(ref_counts) <- list(reference_genes, reference_cells)

  ref_meta <- as.data.frame(read_excel(workbook, sheet = "SampleInfo"))
  names(ref_meta)[1] <- "cell_id"
  ref_meta$cell_id <- as.character(ref_meta$cell_id)
  shared <- intersect(reference_cells, ref_meta$cell_id)
  ref_meta <- ref_meta[match(shared, ref_meta$cell_id), , drop = FALSE]
  ref_counts <- ref_counts[, shared, drop = FALSE]
  labels <- setNames(factor(ref_meta$cell_types), shared)
  n_umi <- setNames(Matrix::colSums(ref_counts), shared)
  keep <- n_umi > 0 & !is.na(labels)
  reference <- Reference(
    counts = ref_counts[, keep, drop = FALSE],
    cell_types = labels[keep],
    nUMI = n_umi[keep],
    require_int = TRUE
  )
  saveRDS(reference, reference_path, compress = FALSE)
  rm(ref_counts, labels, n_umi)
  gc()
}

spatial_path <- file.path(result_dir, "cosmx_spatial_input.rds")
if (file.exists(spatial_path)) {
  message("Reusing prepared CosMx SpatialRNA input")
  spatial <- readRDS(spatial_path)
} else {
  spatial_counts <- readMM(file.path(spatial_dir, "counts.mtx"))
  spatial_genes <- read_vector(file.path(spatial_dir, "genes.tsv"))
  spatial_cells <- read_vector(file.path(spatial_dir, "cells.tsv"))
  dimnames(spatial_counts) <- list(spatial_genes, spatial_cells)
  spatial_meta <- read.csv(file.path(spatial_dir, "spatial.csv"), check.names = FALSE)
  spatial_meta$cell_id <- as.character(spatial_meta$cell_id)
  spatial_meta <- spatial_meta[match(spatial_cells, spatial_meta$cell_id), , drop = FALSE]
  stopifnot(!anyNA(spatial_meta$cell_id))
  coords <- data.frame(x = spatial_meta$x, y = spatial_meta$y, row.names = spatial_cells)
  n_umi <- setNames(Matrix::colSums(spatial_counts), spatial_cells)
  keep <- n_umi >= 5
  spatial <- SpatialRNA(
    coords = coords[keep, , drop = FALSE],
    counts = spatial_counts[, keep, drop = FALSE],
    nUMI = n_umi[keep],
    require_int = TRUE
  )
  saveRDS(spatial, spatial_path, compress = FALSE)
  rm(spatial_counts, spatial_meta, coords, n_umi)
  gc()
}

class_map <- c(
  "Astrocytes" = "glial",
  "GABAergic neurons" = "neuronal",
  "Microglia" = "glial",
  "Neurons" = "neuronal",
  "OPC" = "glial",
  "Stem cells" = "glial"
)
ref_levels <- levels(reference@cell_types)
class_df <- data.frame(
  class = unname(class_map[ref_levels]), row.names = ref_levels,
  stringsAsFactors = FALSE
)
if (anyNA(class_df$class)) stop("Missing reference broad-class mapping")

initialized_path <- file.path(result_dir, "rctd_initialized.rds")
completed_path <- file.path(result_dir, "rctd_completed.rds")
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
      max_cores = 2,
      class_df = class_df,
      test_mode = FALSE
    )
    saveRDS(rctd, initialized_path, compress = FALSE)
  }
  rctd@config$max_cores <- 2
  started <- Sys.time()
  rctd <- run.RCTD(rctd, doublet_mode = "doublet")
  elapsed <- as.numeric(difftime(Sys.time(), started, units = "secs"))
  saveRDS(rctd, completed_path, compress = FALSE)
  write.csv(
    data.frame(elapsed_seconds = elapsed),
    file.path(result_dir, "rctd_timing.csv"), row.names = FALSE
  )
}

rctd <- SPLIT::run_post_process_RCTD(rctd)
converted <- SPLIT::convert_rctd_result_to_purify_input(rctd = rctd)
weights <- converted$deconvolution_weights
primary <- converted$primary_cell_type
split_reference <- converted$reference
if (!identical(rownames(weights), names(primary))) primary <- primary[rownames(weights)]

results_df <- rctd@results$results_df
write.csv(
  data.frame(
    cell_id = rownames(weights),
    primary_cell_type = as.character(primary),
    spot_class = as.character(results_df[rownames(weights), "spot_class"]),
    stringsAsFactors = FALSE
  ),
  file.path(result_dir, "rctd_primary_labels.csv"), row.names = FALSE
)

counts_for_split <- spatial@counts[, rownames(weights), drop = FALSE]
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
split_output <- list(
  purified_counts = split_result$purified_counts,
  cell_meta = split_result$cell_meta,
  run_summary = list(
    dataset = "CosMx Human Frontal Cortex FFPE S3",
    rctd_mode = "doublet",
    reference = "GSE104276 human developing prefrontal cortex scRNA-seq",
    reference_limitation = paste(
      "The linked reference has six labels and lacks endothelial and mature",
      "oligodendrocyte reference classes."
    ),
    cells = ncol(split_result$purified_counts),
    genes = nrow(split_result$purified_counts),
    split_version = as.character(packageVersion("SPLIT")),
    spacexr_version = as.character(packageVersion("spacexr"))
  )
)
saveRDS(split_output, file.path(result_dir, "split_cosmx_frontal_cortex.rds"), compress = FALSE)
writeLines(capture.output(sessionInfo()), file.path(result_dir, "session_info.txt"))
message("RCTD + SPLIT completed: ", result_dir)
