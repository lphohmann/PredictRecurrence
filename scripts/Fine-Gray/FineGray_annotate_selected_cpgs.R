#!/usr/bin/env Rscript

################################################################################
# Annotate + Plot Selected CpGs from Fine-Gray Pipeline
################################################################################
setwd("~/PhD_Workspace/PredictRecurrence/")
suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
  library(ggplot2)
  library(pheatmap)
  library(scales)
})

################################################################################
# ARGUMENTS (mirrors FineGray pipeline)
################################################################################

args <- commandArgs(trailingOnly = TRUE)

DEFAULT_COHORT <- "ERpHER2n"
DEFAULT_DATA_MODE <- "combined"
DEFAULT_OUTPUT_DIR <- "./output/FineGray"
DEFAULT_CPG_ANNOTATION <- "./data/raw/EPIC_probeAnnoObj.csv"

if (length(args) == 0) {
  COHORT_NAME <- DEFAULT_COHORT
  DATA_MODE <- DEFAULT_DATA_MODE
  OUTPUT_BASE_DIR <- DEFAULT_OUTPUT_DIR
  CPG_ANNOTATION_FILE <- DEFAULT_CPG_ANNOTATION
} else if (length(args) == 4) {
  COHORT_NAME <- args[1]
  DATA_MODE <- args[2]
  OUTPUT_BASE_DIR <- args[3]
  CPG_ANNOTATION_FILE <- args[4]
} else {
  stop(
    "USAGE:\n",
    "Rscript annotate_selected_cpgs_from_finegray.R ",
    "<COHORT> <DATA_MODE> <OUTPUT_BASE_DIR> <CPG_ANNOTATION_FILE>\n"
  )
}

################################################################################
# PATHS
################################################################################

finegray_dir <- file.path(
  OUTPUT_BASE_DIR,
  COHORT_NAME,
  tools::toTitleCase(DATA_MODE),
  "Unadjusted",
  "final_results"
)

rdata_file <- file.path(finegray_dir, "all_results.RData")
if (!file.exists(rdata_file)) stop("Missing: ", rdata_file)

output_dir <- file.path(
  OUTPUT_BASE_DIR,
  COHORT_NAME,
  tools::toTitleCase(DATA_MODE),
  "Unadjusted",
  "CpGAnnotation"
)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

################################################################################
# LOAD FINE-GRAY RESULTS
################################################################################

cat("Loading Fine-Gray results:\n", rdata_file, "\n")
load(rdata_file)

if (!exists("features_pooled_all"))
  stop("features_pooled_all not found in Fine-Gray output")

final_features <- features_pooled_all
cpg_features <- final_features[grepl("^cg", final_features)]

cat(sprintf(
  "Final features: %d | CpGs: %d | Clinical removed: %d\n",
  length(final_features),
  length(cpg_features),
  length(final_features) - length(cpg_features)
))

################################################################################
# LOAD CpG ANNOTATION
################################################################################

cpg_anno <- read_csv(CPG_ANNOTATION_FILE, show_col_types = FALSE)
if (!"illuminaID" %in% colnames(cpg_anno))
  stop("Annotation file must contain 'illuminaID'")

annotated_cpgs <- cpg_anno %>%
  filter(illuminaID %in% cpg_features) %>%
  arrange(match(illuminaID, cpg_features))

if (exists("coef_comparison_final")) {
  annotated_cpgs <- annotated_cpgs %>%
    left_join(
      coef_comparison_final %>% filter(grepl("^cg", feature)),
      by = c("illuminaID" = "feature")
    )
}

################################################################################
# SAVE TABLES
################################################################################

write_csv(
  annotated_cpgs,
  file.path(output_dir, "annotated_selected_cpgs.csv")
)

write_lines(
  cpg_features,
  file.path(output_dir, "selected_cpgs.txt")
)

################################################################################
# LOAD DATA (IDENTICAL TO FINE-GRAY PIPELINE)
################################################################################

source("src/utils.R")

library(readr)        # Fast CSV reading
library(dplyr)        # Data manipulation
library(survival)     # Survival analysis (Surv objects, concordance)
library(glmnet)       # Elastic net Cox regression
library(cmprsk)
library(caret)        # Cross-validation fold creation
library(data.table)   # Fast data reading with fread
library(riskRegression) # Score() function for competing risks metrics
library(prodlim) #prodlim::Hist()

INFILE_METHYLATION <- "./data/train/train_methylation_unadjusted.csv"
INFILE_CLINICAL <- "./data/train/train_clinical.csv"

TRAIN_IDS <- list(
  TNBC = "./data/train/train_subcohorts/TNBC_train_ids.csv",
  ERpHER2n = "./data/train/train_subcohorts/ERpHER2n_train_ids.csv",
  All = "./data/train/train_subcohorts/All_train_ids.csv"
)

train_ids <- read_csv(
  TRAIN_IDS[[COHORT_NAME]],
  col_names = FALSE,
  show_col_types = FALSE
)[[1]]

data_list <- load_training_data(train_ids, INFILE_METHYLATION, INFILE_CLINICAL)

beta_matrix <- data_list$beta_matrix
clinical_data <- data_list$clinical_data

# Subset CpGs ONLY (after loading)
beta_cpg <- beta_matrix[, cpg_features, drop = FALSE]

# Same transformation as pipeline
mval_cpg <- beta_to_m(beta_cpg, beta_threshold = 0.001)

################################################################################
# HEATMAP: CpG–CpG CORRELATION (M-values, CENTERED AT 0)
################################################################################

corr_mat <- cor(
  mval_cpg,
  method = "spearman",
  use = "pairwise.complete.obs"
)

# Define symmetric breaks around 0
max_abs_cor <- max(abs(corr_mat), na.rm = TRUE)

breaks <- seq(
  -max_abs_cor,
  max_abs_cor,
  length.out = 101
)

colors <- colorRampPalette(c("navy", "white", "firebrick3"))(100)

pheatmap(
  corr_mat,
  color = colors,
  breaks = breaks,
  clustering_distance_rows = "correlation",
  clustering_distance_cols = "correlation",
  border_color = NA,
  fontsize = 6,
  main = "Spearman correlation of CpGs (M-values)",
  filename = file.path(output_dir, "cpg_correlation_heatmap_centered0.pdf"),
  width = 8,
  height = 8
)

################################################################################
# PLOT 2: BETA VALUE HISTOGRAMS
################################################################################

beta_long <- beta_cpg %>%
  as.data.frame() %>%
  tibble::rownames_to_column("Sample") %>%
  pivot_longer(-Sample, names_to = "CpG", values_to = "Beta")

ggplot(beta_long, aes(Beta)) +
  geom_histogram(bins = 50, fill = "steelblue", color = "black") +
  facet_wrap(~ CpG, scales = "free_y") +
  theme_bw() +
  labs(title = "Beta value distributions")

ggsave(
  file.path(output_dir, "beta_value_histograms.pdf"),
  width = 10,
  height = 8
)

################################################################################
# PLOT 3: FEATURE CLASS BARPLOT
################################################################################

if ("featureClass" %in% colnames(annotated_cpgs)) {
  annotated_cpgs %>%
    count(featureClass) %>%
    ggplot(aes(reorder(featureClass, n), n)) +
    geom_col(fill = "skyblue") +
    coord_flip() +
    theme_bw() +
    labs(
      title = "CpG genomic feature classes",
      x = "Feature class",
      y = "Count"
    )
  
  ggsave(
    file.path(output_dir, "feature_class_barplot.pdf"),
    width = 6,
    height = 4
  )
}

################################################################################
# HEATMAP 2: CpG x Sample (ROW-Z-SCALED BETA VALUES)
################################################################################

# Ensure RFi_event is a FACTOR with two levels only
annotation_col <- data.frame(
  RFi_event = factor(
    clinical_data$RFi_event,
    levels = c(0, 1),
    labels = c("No event", "Event")
  )
)
rownames(annotation_col) <- rownames(clinical_data)

# Explicit annotation colors (EXACTLY TWO)
annotation_colors <- list(
  RFi_event = c(
    "No event" = "grey80",
    "Event"    = "firebrick3"
  )
)

pheatmap(
  t(beta_cpg),
  scale = "row",  # <-- row-z-scored beta values
  annotation_col = annotation_col,
  annotation_colors = annotation_colors,
  show_colnames = FALSE,
  fontsize_row = 6,
  border_color = NA,
  main = "Selected CpGs (row-z-scaled beta values)",
  filename = file.path(output_dir, "cpg_beta_heatmap_rowz.pdf"),
  width = 10,
  height = 8
)


################################################################################
cat("\nDONE ✔ All plots + annotations saved to:\n", output_dir, "\n")
################################################################################

################################################################################
# PATHWAY ENRICHMENT FOR DNA METHYLATION (missMethyl)
################################################################################

suppressPackageStartupMessages({
  library(missMethyl)
  library(AnnotationDbi)
  library(org.Hs.eg.db)
})

cat("\nRunning pathway enrichment using missMethyl...\n")

# ------------------------------------------------------------------
# CpG universe = all CpGs tested in the model
# (CRITICAL for bias correction)
# ------------------------------------------------------------------

cpg_universe <- colnames(beta_matrix)[grepl("^cg", colnames(beta_matrix))]

# ------------------------------------------------------------------
# Significant CpGs
# Option 1 (default): all CpGs selected by Fine-Gray
# ------------------------------------------------------------------

sig_cpgs <- cpg_features

# ------------------------------------------------------------------
# Run GO enrichment (Biological Process)
# ------------------------------------------------------------------
# KEGG enrichment
#kegg_results <- gometh(
#  sig.cpg   = sig_cpgs,
#  all.cpg   = cpg_universe,
#  collection= "KEGG",
#  array.type= "EPIC"
#)

go_results <- gometh(
  sig.cpg = sig_cpgs,
  all.cpg = cpg_universe,
  collection = "GO",
  array.type = "EPIC"
)

# Keep BP only and significant terms
go_bp <- go_results %>%
  filter(ONTOLOGY == "BP", FDR < 0.05)

if (nrow(go_bp) == 0) {
  warning("No significant GO BP terms found (FDR < 0.05).")
} else {
  
  # ----------------------------------------------------------------
  # Save full results
  # ----------------------------------------------------------------
  
  write_csv(
    go_bp,
    file.path(output_dir, "GO_BP_enrichment_missMethyl.csv")
  )
  
  # ----------------------------------------------------------------
  # Plot top enriched pathways
  # ----------------------------------------------------------------
  
  top_go <- go_bp %>%
    arrange(FDR) %>%
    slice_head(n = min(15, n()))
  
  p <- ggplot(
    top_go,
    aes(
      x = reorder(Term, -log10(FDR)),
      y = -log10(FDR)
    )
  ) +
    geom_col(fill = "steelblue") +
    coord_flip() +
    theme_bw() +
    labs(
      title = "GO Biological Process enrichment (missMethyl)",
      x = "",
      y = "-log10(FDR)"
    )
  
  ggsave(
    filename = file.path(output_dir, "GO_BP_enrichment_missMethyl.pdf"),
    plot = p,
    width = 7,
    height = 5
  )
  
  cat("GO enrichment completed and saved.\n")
}

################################################################################
# END OF SCRIPT
################################################################################
