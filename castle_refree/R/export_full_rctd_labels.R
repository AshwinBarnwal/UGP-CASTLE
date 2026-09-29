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
input_path <- file.path(repo_root, "data", "split_inputs_full.rds")
output_path <- file.path(
  repo_root, "results", "castle_refree_xenium_breast_beta2", "rctd_labels.csv"
)

obj <- readRDS(input_path)
weights <- obj$weights
primary <- obj$primary

cell_ids <- names(primary)
if (is.null(cell_ids) && !is.null(rownames(weights))) cell_ids <- rownames(weights)
if (is.null(cell_ids)) stop("No cell identifiers found on primary labels or weights")
if (length(primary) != length(cell_ids)) stop("Primary labels and cell identifiers differ")

out <- data.frame(cell_id = cell_ids, primary = as.character(primary), check.names = FALSE)
write.csv(out, output_path, row.names = FALSE)
message("Exported ", nrow(out), " RCTD labels to ", output_path)
