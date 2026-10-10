#!/usr/bin/env R

# Export the combined full SPLIT result into a portable sparse bundle for the
# Python scIB benchmark. Use Rterm.exe on this Windows machine because its
# Rscript.exe installation is unstable.

script_file <- function() {
  command_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", command_args, value = TRUE)
  if (length(file_arg)) {
    return(normalizePath(sub("^--file=", "", file_arg[[1]]), mustWork = TRUE))
  }
  file_flag <- match("-f", command_args)
  if (!is.na(file_flag) && file_flag < length(command_args)) {
    return(normalizePath(command_args[[file_flag + 1]], mustWork = TRUE))
  }
  for (frame in rev(sys.frames())) {
    if (!is.null(frame$ofile)) return(normalizePath(frame$ofile, mustWork = TRUE))
  }
  stop("Could not determine script location")
}

suppressPackageStartupMessages(library(Matrix))

repo_root <- normalizePath(
  file.path(dirname(script_file()), "..", ".."), winslash = "/", mustWork = TRUE
)
args <- commandArgs(trailingOnly = TRUE)
input_path <- if (length(args) >= 1) args[[1]] else file.path(
  repo_root, "results", "xenium_rep1_split_full_combined.rds"
)
output_dir <- if (length(args) >= 2) args[[2]] else file.path(
  repo_root, "data", "benchmark_split_castle"
)

obj <- readRDS(input_path)
required <- c("purified_counts", "cell_meta")
missing <- setdiff(required, names(obj))
if (length(missing)) stop("Missing SPLIT fields: ", paste(missing, collapse = ", "))

counts <- methods::as(obj$purified_counts, "dgCMatrix")
meta <- as.data.frame(obj$cell_meta)
if (is.null(rownames(counts)) || is.null(colnames(counts))) {
  stop("SPLIT purified_counts must have gene and cell names")
}
if (ncol(counts) != nrow(meta)) stop("SPLIT counts and cell_meta dimensions differ")
if (any(!is.finite(counts@x)) || any(counts@x < 0)) {
  stop("SPLIT counts contain invalid values")
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
Matrix::writeMM(counts, file.path(output_dir, "split_purified_counts.mtx"))
write.table(
  rownames(counts), file.path(output_dir, "split_genes.tsv"),
  quote = FALSE, row.names = FALSE, col.names = FALSE, sep = "\t"
)
write.table(
  colnames(counts), file.path(output_dir, "split_cells.tsv"),
  quote = FALSE, row.names = FALSE, col.names = FALSE, sep = "\t"
)
write.csv(meta, file.path(output_dir, "split_cell_meta.csv"), row.names = FALSE)

manifest <- list(
  source = normalizePath(input_path, winslash = "/", mustWork = TRUE),
  cells = ncol(counts),
  genes = nrow(counts),
  matrix_orientation = "genes_by_cells",
  split_summary = obj$run_summary
)
dput(manifest, file.path(output_dir, "manifest.R"))
message("Exported SPLIT benchmark bundle: ", normalizePath(output_dir, winslash = "/"))
