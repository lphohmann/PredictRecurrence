#!/usr/bin/env Rscript
################################################################################
# Plotting results for a cohort
# Author: Lennart Hohmann
################################################################################
# LIBRARIES
################################################################################
setwd("~/PhD_Workspace/PredictRecurrence/")

library(readr)          # fast CSV reading
library(dplyr)          # data manipulation
library(data.table)     # fast data reading
library(devtools)
library(pheatmap)
library(missMethyl)
library(AnnotationDbi)
library(org.Hs.eg.db)

source("./src/finegray_functions.R")
source("./src/finegray_plotting_functions.R")

################################################################################
# COMMAND LINE ARGUMENTS
################################################################################
args <- commandArgs(trailingOnly = TRUE)

COHORT_NAME <- "All" # 'TNBC', 'ERpHER2n', 'All'
if (length(args) == 1) COHORT_NAME <- args[1]

################################################################################
# INPUT / OUTPUT
################################################################################
SCRIPT_NAME <- "finegray_plotting"

INFILE_METHYLATION <- "./data/train/train_methylation_unadjusted.csv"
INFILE_CLINICAL    <- "./data/train/train_clinical.csv"
TRAIN_CPGS         <- "./data/set_definitions/CpG_prefiltered_sets/cpg_ids_atac_overlap.txt"

COHORT_TRAIN_IDS_PATHS <- list(
  TNBC     = "./data/train/train_subcohorts/TNBC_train_ids.csv",
  ERpHER2n = "./data/train/train_subcohorts/ERpHER2n_train_ids.csv",
  All      = "./data/train/train_subcohorts/All_train_ids.csv"
)

current_input_dir <- file.path("./output/FineGray", COHORT_NAME)
current_output_dir <- file.path(current_input_dir, "Plots")

dir.create(current_output_dir, recursive = TRUE, showWarnings = FALSE)

################################################################################
# PARAMETERS
################################################################################

# Cohort-specific
ADMIN_CENSORING_CUTOFF <- if (COHORT_NAME == "TNBC") 5.01 else NULL
EVAL_TIMES             <- if (COHORT_NAME == "TNBC") 1:5 else 1:10

# Clinical variables (final combined model only)
CLIN_CONT        <- c("Age", "Size.mm")
CLIN_CATEGORICAL <- if (COHORT_NAME == "All") c("NHG", "LN", "ER", "PR", "HER2") else c("NHG", "LN")
CLINVARS_ALL     <- c(CLIN_CONT, CLIN_CATEGORICAL)

################################################################################
# RUN INFO
################################################################################
start_time <- Sys.time()
cat(paste(rep("=", 80), collapse = ""), "\n")
cat(sprintf("%s\n", toupper(SCRIPT_NAME)))
cat(sprintf("Started:             %s\n", start_time))
cat(sprintf("Cohort:              %s\n", COHORT_NAME))
cat(sprintf("Clinical variables:  %s\n", paste(CLINVARS_ALL, collapse = ", ")))
cat(sprintf("Evaluation times:    %s\n", paste(EVAL_TIMES,   collapse = ", ")))
cat(paste(rep("=", 80), collapse = ""), "\n\n")

################################################################################
# CV and final results
################################################################################

outer_fold_results <- loadRData(file.path(current_input_dir, 
                                          "outer_fold_results.RData"))
final_fg_results <- loadRData(file = file.path(current_input_dir, "final_fg_results.RData"))

# check overlap of CpGs in final model with stable CpGs from nCV
stability_metrics <- outer_fold_results$stability_results$stability_metrics
stable_cpgs <- stability_metrics[stability_metrics$selection_freq >= 0.4,]$feature
final_cpgs <- final_fg_results$mrs_construction$selected_cpgs
length(intersect(final_cpgs,stable_cpgs))

################################################################################
# DATA LOADING AND PREPROCESSING
################################################################################

cat(sprintf("\n========== LOADING DATA ==========\n"))

# cpg anno
cpg_anno <- read_csv("./data/raw/EPIC_probeAnnoObj.csv", show_col_types = FALSE)

# Load sample IDs and data
train_ids <- read_csv(
  COHORT_TRAIN_IDS_PATHS[[COHORT_NAME]], 
  col_names = FALSE, 
  show_col_types = FALSE
)[[1]]

data_list <- load_training_data(train_ids, INFILE_METHYLATION, INFILE_CLINICAL)
beta_matrix <- data_list$beta_matrix
clinical_data <- data_list$clinical_data

################################################################################
# methylation matrix 
################################################################################

# Convert beta values to M-values
mvals <- beta_to_m(beta_matrix, beta_threshold = 0.001)

# Subset to predefined CpGs (if specified)
if (!is.null(TRAIN_CPGS)) {
  mvals <- subset_methylation(mvals, TRAIN_CPGS)
}

cat(sprintf("CV feature matrix: %d samples × %d CpGs (methylation-only)\n", 
            nrow(mvals), ncol(mvals)))

################################################################################
# clinical matrix 
################################################################################

# Apply administrative censoring if needed
if (!is.null(ADMIN_CENSORING_CUTOFF)) {
  clinical_data <- apply_admin_censoring(
    clinical_data, "RFi_years", "RFi_event", ADMIN_CENSORING_CUTOFF
  )
  clinical_data <- apply_admin_censoring(
    clinical_data, "OS_years", "OS_event", ADMIN_CENSORING_CUTOFF
  )
}

clinical_data$LN <- gsub("N\\+", "Np", clinical_data$LN)
clin <- clinical_data[c(CLIN_CONT, CLIN_CATEGORICAL)]
clin <- clin[rownames(mvals), , drop = FALSE]
encoded_result <- onehot_encode_clinical(clin, CLIN_CATEGORICAL)

################################################################################
# plot nCV selection stability heatmap
################################################################################

p1 <- plot_coxnet_stability_heatmap(
  outer_fold_results$outer_fold_results,
  min_freq  = if (COHORT_NAME == "TNBC") 0.2 else 0.4,
  cohort    = COHORT_NAME,
  legend             = "yes",
  plot_dir  = current_output_dir)

################################################################################
# plot coefs in final ridge FG model with origin of selected CpGs
################################################################################

p2 <- plot_penalized_fg_coefficients(
  cpg_coefs          = final_fg_results$mrs_construction$cpg_coefs,
  outer_fold_results = outer_fold_results$outer_fold_results,
  cohort             = COHORT_NAME,
  legend             = "yes",
  plot_dir           = current_output_dir)

################################################################################
# plot AUC (t) for the three models
################################################################################

p3 <- plot_time_dependent_auc(
  perf_results = outer_fold_results$aggregated_perf_results,
  cohort       = COHORT_NAME,
  x_max        = 10,
  legend             = "yes",
  plot_dir     = current_output_dir)

################################################################################
# plot AUC (t) for the three models
################################################################################

p4 <- plot_fg_forest_hr(
  final_fg_results$final_model$var_effects,
  cohort   = COHORT_NAME,
  plot_dir = current_output_dir)

################################################################################
# Plotting results for the final CpG set
################################################################################

cpg_features <- final_fg_results$mrs_construction$selected_cpgs

################################################################################
# HEATMAP: CpG–CpG CORRELATION (M-values, CENTERED AT 0)
################################################################################

mvals_cpgs <- mvals[, cpg_features, drop = FALSE]

annotated_cpgs <- cpg_anno %>%
  filter(illuminaID %in% cpg_features) %>%
  arrange(match(illuminaID, cpg_features))

corr_mat <- cor(
  mvals_cpgs,
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
  filename = file.path(current_output_dir, "cpg_correlation_heatmap.pdf"),  
  width = 8,
  height = 8
)

################################################################################
# PLOT FEATURE CLASS 
################################################################################

feature_counts <- annotated_cpgs %>%
  dplyr::count(featureClass)

p5 <- ggplot(feature_counts, aes(x = "", y = n, fill = featureClass)) +
  geom_col(width = 1, color = "white") +
  geom_text(
    aes(label = n),
    position = position_stack(vjust = 0.5),
    size = 3.5,
    color = "black"
  ) +
  coord_polar(theta = "y") +
  theme_bw() +
  theme(
    axis.text = element_blank(),
    axis.title = element_blank(),
    axis.ticks = element_blank(),
    panel.grid = element_blank(),
    legend.title = element_text(size = 10),
    legend.text = element_text(size = 9)
  ) +
  labs(
    title = "Genomic position of stable CpGs",
    fill = "Feature class"
  )

ggsave(
  file.path(current_output_dir, "feature_class_piechart.pdf"),
  p5, width = 6, height = 6)

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
  t(mvals_cpgs),
  scale = "row",  # <-- row-z-scored beta values
  annotation_col = annotation_col,
  annotation_colors = annotation_colors,
  show_colnames = FALSE,
  fontsize_row = 6,
  border_color = NA,
  main = "Selected CpGs (row-z-scaled m values)",
  filename = file.path(current_output_dir, "cpg_beta_heatmap_rowz.pdf"),
  width = 8,
  height = 8
)

################################################################################
# PATHWAY ENRICHMENT FOR DNA METHYLATION (missMethyl)
################################################################################
cat("\nRunning pathway enrichment using missMethyl...\n")

# ------------------------------------------------------------------
# CpG universe = all CpGs tested in the model
# (for bias correction)
# ------------------------------------------------------------------

cpg_universe <- colnames(mvals)

# ------------------------------------------------------------------
# Significant CpGs: all CpGs selected by Fine-Gray
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
  print("No significant GO BP terms found (FDR < 0.05).")
} else {
  
  # ----------------------------------------------------------------
  # Save full results
  # ----------------------------------------------------------------
  
  write_csv(
    go_bp,
    file.path(current_output_dir, "GO_BP_enrichment_missMethyl.csv")
  )
  
  # ----------------------------------------------------------------
  # Plot top enriched pathways
  # ----------------------------------------------------------------
  
  top_go <- go_bp %>%
    arrange(FDR) %>%
    slice_head(n = min(10, n()))
  
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
    filename = file.path(current_output_dir, "GO_BP_enrichment_missMethyl.pdf"),
    plot = p,
    width = 7,
    height = 5
  )
  
  cat("GO enrichment completed and saved.\n")
}

################################################################################
# CLEANUP AND FINISH
################################################################################

total_runtime <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))
cat(paste(rep("=", 80), collapse = ""), "\n")
cat(sprintf("COMPLETED\n"))
cat(sprintf("Finished:      %s\n", Sys.time()))
cat(sprintf("Total runtime: %.1f minutes\n", total_runtime))
cat(paste(rep("=", 80), collapse = ""), "\n")
