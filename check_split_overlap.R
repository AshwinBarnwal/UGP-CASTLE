split <- readRDS("results/xenium_rep1_split_full_combined.rds")
gt <- read.csv(
  "data/xenium_r1_supervised_labels.csv",
  colClasses = "character"
)

split_ids <- trimws(colnames(split$purified_counts))
castle_ids <- trimws(readLines(
  "results/castle_refree_xenium_breast_beta2/cells.tsv",
  warn = FALSE
))

labels <- tolower(trimws(gt$Cluster))
gt_ids <- trimws(gt$Barcode)

keep <- !is.na(gt_ids) & nzchar(gt_ids) &
  !is.na(labels) & nzchar(labels) &
  !labels %in% c("unlabeled", "unlabelled")

labeled_gt_ids <- unique(gt_ids[keep])

cat("SPLIT matrix and metadata IDs in same order:",
    identical(split_ids, rownames(split$cell_meta)), "\n")
cat("Unique labeled ground-truth barcodes:", length(labeled_gt_ids), "\n")
cat("Present in SPLIT:", sum(labeled_gt_ids %in% split_ids), "\n")
cat("Present in CASTLE:", sum(labeled_gt_ids %in% castle_ids), "\n")
cat("Present in both:", sum(
  labeled_gt_ids %in% split_ids & labeled_gt_ids %in% castle_ids
), "\n")
cat("Ground-truth barcodes missing from SPLIT:",
    sum(!labeled_gt_ids %in% split_ids), "\n")
cat("Duplicate labeled ground-truth barcodes:",
    sum(duplicated(gt_ids[keep])), "\n")