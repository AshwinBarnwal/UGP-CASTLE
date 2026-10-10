#!/usr/bin/env Rscript

# Export the SPLIT dgCMatrix slots to compact binary files that SciPy can load.
# Run from any working directory; paths are resolved relative to the project.

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (!length(script_arg)) stop("Run this file with Rscript.")
script_path <- normalizePath(sub("^--file=", "", script_arg[[1]]), winslash = "/")
project_root <- normalizePath(file.path(dirname(script_path), "..", ".."), winslash = "/")
split_path <- file.path(project_root, "results", "xenium_rep1_split_full_combined.rds")
export_dir <- file.path(project_root, "results", "split_janesick_python_export")

if (!file.exists(split_path)) stop("SPLIT RDS not found: ", split_path)
dir.create(export_dir, recursive = TRUE, showWarnings = FALSE)

split_result <- readRDS(split_path)
if (is.null(split_result$purified_counts)) {
  stop("The SPLIT RDS does not contain purified_counts.")
}
counts <- split_result$purified_counts
if (!inherits(counts, "sparseMatrix")) counts <- Matrix::Matrix(counts, sparse = TRUE)
counts <- methods::as(counts, "dgCMatrix")
if (is.null(rownames(counts)) || is.null(colnames(counts))) {
  stop("purified_counts must have gene row names and cell barcode column names.")
}
if (anyDuplicated(colnames(counts))) stop("Duplicate barcodes in purified_counts.")

write_binary <- function(values, path, size) {
  connection <- file(path, open = "wb")
  on.exit(close(connection))
  writeBin(values, connection, size = size, endian = "little")
}

write_binary(counts@x, file.path(export_dir, "counts_data.bin"), 8L)
write_binary(counts@i, file.path(export_dir, "counts_indices.bin"), 4L)
write_binary(counts@p, file.path(export_dir, "counts_indptr.bin"), 4L)
writeLines(rownames(counts), file.path(export_dir, "genes.tsv"), useBytes = TRUE)
writeLines(colnames(counts), file.path(export_dir, "barcodes.tsv"), useBytes = TRUE)
writeLines(as.character(c(nrow(counts), ncol(counts))),
           file.path(export_dir, "shape.txt"))

cat("Exported sparse SPLIT counts:", nrow(counts), "genes x", ncol(counts), "cells\n")
cat("Export directory:", normalizePath(export_dir, winslash = "/"), "\n")
