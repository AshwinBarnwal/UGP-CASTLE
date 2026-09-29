#!/usr/bin/env Rscript

# Export the already-cached Janesick breast Xenium object into a portable,
# reference-free CASTLE input bundle. This script does not load Chromium data.

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
  stop("Could not determine the exporter script location")
}

repo_root <- normalizePath(
  file.path(dirname(script_file()), "..", ".."), winslash = "/", mustWork = TRUE
)
args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args) >= 1) args[[1]] else file.path(repo_root, "data", "castle_xenium_breast")
input_rds <- if (length(args) >= 2 && nzchar(args[[2]])) args[[2]] else NA_character_

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages({
  library(Matrix)
  library(SummarizedExperiment)
})

if (!is.na(input_rds)) {
  message("Loading spatial object from: ", input_rds)
  xe_spe <- readRDS(input_rds)
} else {
  suppressPackageStartupMessages(library(STexampleData))
  message("Loading cached Janesick_breastCancer_Xenium_rep1()")
  xe_spe <- STexampleData::Janesick_breastCancer_Xenium_rep1()
}

assay_names <- SummarizedExperiment::assayNames(xe_spe)
count_assay <- if ("counts" %in% assay_names) "counts" else assay_names[[1]]
counts <- SummarizedExperiment::assay(xe_spe, count_assay)
counts <- methods::as(counts, "dgCMatrix")

if (is.null(rownames(counts)) || is.null(colnames(counts))) {
  stop("The count matrix must have both gene and cell names.")
}

coords <- NULL
if (requireNamespace("SpatialExperiment", quietly = TRUE) &&
    methods::is(xe_spe, "SpatialExperiment")) {
  coords <- SpatialExperiment::spatialCoords(xe_spe)
}

meta <- as.data.frame(SummarizedExperiment::colData(xe_spe))
find_column <- function(candidates, available) {
  hit <- candidates[candidates %in% available]
  if (length(hit)) hit[[1]] else NA_character_
}

if (!is.null(coords) && ncol(coords) >= 2) {
  x <- as.numeric(coords[, 1])
  y <- as.numeric(coords[, 2])
  z <- if (ncol(coords) >= 3) as.numeric(coords[, 3]) else NULL
} else {
  x_name <- find_column(c("x", "x_centroid", "centroid_x", "CenterX"), names(meta))
  y_name <- find_column(c("y", "y_centroid", "centroid_y", "CenterY"), names(meta))
  if (is.na(x_name) || is.na(y_name)) {
    stop(
      "Could not identify coordinates. Available colData columns: ",
      paste(names(meta), collapse = ", ")
    )
  }
  x <- as.numeric(meta[[x_name]])
  y <- as.numeric(meta[[y_name]])
  z_name <- find_column(c("z", "z_centroid", "centroid_z", "CenterZ"), names(meta))
  z <- if (!is.na(z_name)) as.numeric(meta[[z_name]]) else NULL
}

area_name <- find_column(
  c("area", "cell_area", "cell_area_um2", "CellArea"),
  names(meta)
)
if (is.na(area_name)) {
  warning("No cell-area column found; exporting area=1 for every cell.")
  area <- rep(1, ncol(counts))
} else {
  area <- as.numeric(meta[[area_name]])
}

spatial <- data.frame(
  cell_id = colnames(counts),
  x = x,
  y = y,
  area = area,
  check.names = FALSE
)
if (!is.null(z)) spatial$z <- z

# MatrixMarket stores the native genes x cells orientation; the Python loader
# detects and transposes it. Negative-control probes are not present in the
# compact STexampleData object, so no background column is invented.
Matrix::writeMM(counts, file.path(output_dir, "counts.mtx"))
write.table(
  rownames(counts), file.path(output_dir, "genes.tsv"),
  quote = FALSE, row.names = FALSE, col.names = FALSE, sep = "\t"
)
write.table(
  colnames(counts), file.path(output_dir, "cells.tsv"),
  quote = FALSE, row.names = FALSE, col.names = FALSE, sep = "\t"
)
write.csv(spatial, file.path(output_dir, "spatial.csv"), row.names = FALSE)

manifest <- list(
  source = "STexampleData::Janesick_breastCancer_Xenium_rep1",
  count_assay = count_assay,
  genes = nrow(counts),
  cells = ncol(counts),
  has_negative_controls = FALSE,
  chromium_reference_used = FALSE
)
dput(manifest, file = file.path(output_dir, "manifest.R"))
message("Export complete: ", normalizePath(output_dir, winslash = "/"))

