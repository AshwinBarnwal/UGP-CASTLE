#!/usr/bin/env R

# Reproduce the full SPLIT purification used for the Xenium breast benchmark.
# All paths are resolved relative to this repository.

script_file <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg)) {
    return(normalizePath(sub("^--file=", "", file_arg[[1]]), mustWork = TRUE))
  }
  file_flag <- match("-f", args)
  if (!is.na(file_flag) && file_flag < length(args)) {
    return(normalizePath(args[[file_flag + 1]], mustWork = TRUE))
  }
  for (frame in rev(sys.frames())) {
    if (!is.null(frame$ofile)) return(normalizePath(frame$ofile, mustWork = TRUE))
  }
  stop("Could not determine script location")
}

repo_root <- normalizePath(
  file.path(dirname(script_file()), "..", ".."),
  winslash = "/",
  mustWork = TRUE
)

suppressPackageStartupMessages({
  library(Matrix)
  library(SPLIT)
  library(SummarizedExperiment)
})

args <- commandArgs(trailingOnly = TRUE)
batch_size <- if (length(args) >= 1) as.integer(args[[1]]) else 25000L
if (is.na(batch_size) || batch_size < 1L) stop("batch_size must be positive")

input_path <- file.path(repo_root, "data", "split_inputs_full.rds")
cache_path <- file.path(repo_root, "data", "experimenthub")
batch_dir <- file.path(repo_root, "results", "full_split_batches")
combined_path <- file.path(
  repo_root, "results", "xenium_rep1_split_full_combined.rds"
)
dir.create(batch_dir, recursive = TRUE, showWarnings = FALSE)

stopifnot(file.exists(input_path), dir.exists(cache_path))
if (requireNamespace("ExperimentHub", quietly = TRUE)) {
  ExperimentHub::setExperimentHubOption("CACHE", cache_path)
}

inp <- readRDS(input_path)
required <- c("weights", "primary", "reference")
missing <- setdiff(required, names(inp))
if (length(missing)) stop("Missing input fields: ", paste(missing, collapse = ", "))

weights <- inp$weights
primary <- inp$primary
reference <- inp$reference
rm(inp)

if (is.null(names(primary))) names(primary) <- rownames(weights)
if (!requireNamespace("STexampleData", quietly = TRUE)) {
  stop("Install STexampleData before running this script")
}

xe_spe <- STexampleData::Janesick_breastCancer_Xenium_rep1()
counts <- SummarizedExperiment::assay(xe_spe, "counts")
rm(xe_spe)

cells <- intersect(colnames(counts), rownames(weights))
genes <- intersect(rownames(counts), colnames(reference))
if (!length(cells) || !length(genes)) stop("No shared cells or genes")
if (!all(cells %in% names(primary))) stop("Primary labels missing for some cells")

counts <- counts[genes, cells, drop = FALSE]
weights <- weights[cells, , drop = FALSE]
primary <- as.character(primary[cells])
names(primary) <- cells
reference <- reference[, genes, drop = FALSE]

batch_ids <- split(seq_along(cells), ceiling(seq_along(cells) / batch_size))
manifest_rows <- vector("list", length(batch_ids))

for (i in seq_along(batch_ids)) {
  idx <- batch_ids[[i]]
  batch_cells <- cells[idx]
  output_path <- file.path(batch_dir, sprintf("split_batch_%02d.rds", i))
  qc_path <- file.path(batch_dir, sprintf("split_batch_%02d_qc.csv", i))

  if (file.exists(output_path) && file.exists(qc_path)) {
    message("Reusing completed batch ", i, "/", length(batch_ids))
    manifest_rows[[i]] <- read.csv(qc_path, check.names = FALSE)
    next
  }

  message("Purifying batch ", i, "/", length(batch_ids))
  raw_total <- sum(counts[, batch_cells, drop = FALSE])
  result <- SPLIT::rctd_free_purify(
    counts = counts[, batch_cells, drop = FALSE],
    deconvolution_weights = weights[batch_cells, , drop = FALSE],
    reference = reference,
    primary_cell_type = primary[batch_cells],
    DO_run_in_chunks = TRUE,
    chunk_size = batch_size,
    DO_require_sumup_to_one = TRUE,
    DO_output_sce = FALSE,
    DO_remove_residual_contamination = FALSE
  )

  rownames(result$cell_meta) <- result$cell_meta$cell_id
  purified_total <- sum(result$purified_counts)
  saveRDS(result, output_path, compress = FALSE)

  qc <- data.frame(
    batch = i,
    cells = ncol(result$purified_counts),
    genes = nrow(result$purified_counts),
    raw_total = raw_total,
    purified_total = purified_total,
    fraction_removed = 1 - purified_total / raw_total,
    split_version = as.character(packageVersion("SPLIT"))
  )
  write.csv(qc, qc_path, row.names = FALSE)
  manifest_rows[[i]] <- qc
  rm(result)
  gc()
}

manifest <- do.call(rbind, manifest_rows)
write.csv(manifest, file.path(batch_dir, "manifest.csv"), row.names = FALSE)

result_files <- file.path(
  batch_dir, sprintf("split_batch_%02d.rds", seq_along(batch_ids))
)
results <- lapply(result_files, readRDS)
purified_counts <- do.call(cbind, lapply(results, `[[`, "purified_counts"))
cell_meta <- do.call(rbind, lapply(results, `[[`, "cell_meta"))
cell_meta <- cell_meta[colnames(purified_counts), , drop = FALSE]

stopifnot(
  identical(colnames(purified_counts), rownames(cell_meta)),
  ncol(purified_counts) == sum(manifest$cells),
  nrow(purified_counts) == unique(manifest$genes)
)

combined <- list(
  purified_counts = purified_counts,
  cell_meta = cell_meta,
  run_summary = list(
    dataset = "Xenium FFPE Human Breast Cancer Replicate 1",
    method = "SPLIT rctd_free_purify",
    split_version = as.character(packageVersion("SPLIT")),
    cells = ncol(purified_counts),
    genes = nrow(purified_counts),
    fraction_removed = 1 - sum(manifest$purified_total) / sum(manifest$raw_total),
    used_precomputed_RCTD = TRUE,
    batch_size = batch_size
  )
)
saveRDS(combined, combined_path, compress = FALSE)
writeLines(
  capture.output(sessionInfo()),
  file.path(repo_root, "results", "session_info_full_split.txt")
)
message("Saved combined SPLIT output to: ", combined_path)
