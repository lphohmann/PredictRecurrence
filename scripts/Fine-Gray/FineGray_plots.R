#!/usr/bin/env Rscript

################################################################################
# Fine-Gray Competing Risks with Dual Final Models
# Author: Lennart Hohmann

################################################################################
# LIBRARIES
################################################################################

library(readr)        # Fast CSV reading
library(dplyr)        # Data manipulation
library(survival)     # Survival analysis (Surv objects, concordance)
library(glmnet)       # Elastic net Cox regression
library(cmprsk)
library(caret)        # Cross-validation fold creation
library(data.table)   # Fast data reading with fread
library(riskRegression) # Score() function for competing risks metrics
library(prodlim) #prodlim::Hist()
library(devtools)
library(fastcmprsk)

setwd("~/PhD_Workspace/PredictRecurrence/")
source("./src/finegray_functions.R")
source("./src/finegray_plotting_functions.R")

################################################################################
# COMMAND LINE ARGUMENTS
################################################################################

# Defaults
DEFAULT_COHORT <- "TNBC"
DEFAULT_TRAIN_CPGS <- "./data/set_definitions/CpG_prefiltered_sets/cpg_ids_atac_overlap.txt"
DEFAULT_OUTPUT_DIR <- "./output/FineGray"
OUTPUT_BASE_DIR <- DEFAULT_OUTPUT_DIR
COHORT_NAME <- DEFAULT_COHORT
################################################################################
# INPUT OUTPUT SETTINGS
################################################################################

# Input files
INFILE_METHYLATION <- "./data/train/train_methylation_unadjusted.csv"
COHORT_TRAIN_IDS_PATHS <- list(
  TNBC = "./data/train/train_subcohorts/TNBC_train_ids.csv",
  ERpHER2n = "./data/train/train_subcohorts/ERpHER2n_train_ids.csv",
  All = "./data/train/train_subcohorts/All_train_ids.csv"
)
INFILE_CLINICAL <- "./data/train/train_clinical.csv"

# Output directory - simplified since we don't have DATA_MODE
current_output_dir <- file.path(
  OUTPUT_BASE_DIR, 
  COHORT_NAME, 
  "DualMode",  # New folder name to distinguish from old pipeline
  "Unadjusted"
)
dir.create(current_output_dir, recursive = TRUE, showWarnings = FALSE)

plot_dir <- file.path(
  current_output_dir,
  "Figures"
)
dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

################################################################################
# LOAD DATA
################################################################################

# Load sample IDs and data
train_ids <- read_csv(
  COHORT_TRAIN_IDS_PATHS[[COHORT_NAME]], 
  col_names = FALSE, 
  show_col_types = FALSE
)[[1]]

data_list <- load_training_data(train_ids, INFILE_METHYLATION, INFILE_CLINICAL)
beta_matrix <- data_list$beta_matrix
clinical_data <- data_list$clinical_data

load(file.path(current_output_dir, 
               "outer_fold_results.RData"))
load(file = file.path(current_output_dir, "final_fg_results.RData"))


cpg_anno <- read_csv("./data/raw/EPIC_probeAnnoObj.csv", show_col_types = FALSE)


#str(outer_fold_results)

# only include valid folds (where not skipped)
outer_fold_results <- Filter(function(f) {
  !is.null(f) && !isTRUE(f$metadata$skipped)
}, outer_fold_results)

perf_results_mrs <- aggregate_cv_performance_threefg(
  outer_fold_results,
  model_name = "FGR_MRS")

perf_results_clin <- aggregate_cv_performance_threefg(
  outer_fold_results,
  model_name = "FGR_CLIN")

perf_results_combined <- aggregate_cv_performance_threefg(
  outer_fold_results,
  model_name = "FGR_COMBINED")

aggregated_perf <- list(FGR_MRS = perf_results_mrs,
                        FGR_CLIN= perf_results_clin,
                        FGR_COMBINED= perf_results_combined)

stability_results <- assess_finegray_stability_threefg(
  outer_fold_results
)


# plots
plot_coxnet_stability_heatmap(plot_dir=plot_dir,
                              outer_fold_results,
                              min_freq = 0.4)

# echk where the coefs acutalyl omce from ehre? whic olf
# this plots mean coefs which is not what i want
plot_coxnet_coefficients(plot_dir=plot_dir,
                         outer_fold_results,
                         min_freq = 0.4)

plot_penalized_fg_coefficients(
  plot_dir=plot_dir,
  cpg_coefs=final_results$mrs_construction$cpg_coefs,
  outer_fold_results = outer_fold_results)

plot_fg_forest_hr(
  plot_dir=plot_dir,
  vimp_fg = final_results$variable_importance$fgr_COMBINED)

plot_time_dependent_auc(perf_results=aggregated_perf,
                        plot_dir = plot_dir,
                        cohort = "ERpHER2n"
                        )


################################################################################
################################################################################



loadRData <- function(file.path){
  load(file.path)
  get(ls()[ls() != "file.path"])
}

cohorts <- c("ERpHER2n", "TNBC", "All")
paths <- list(
  ERpHER2n = "./output/FineGray/ERpHER2n/DualMode/Unadjusted/outer_fold_results.RData",
  TNBC     = "./output/FineGray/TNBC/DualMode/Unadjusted/outer_fold_results.RData",
  All      = "./output/FineGray/All/DualMode/Unadjusted/outer_fold_results.RData"
)

aggregated_perf_all <- list()

for (cohort in cohorts) {
  
  cat(sprintf("\n=== Processing: %s ===\n", cohort))
  
  # Load and filter
  outer_fold_results <- loadRData(paths[[cohort]])
  outer_fold_results <- Filter(function(f) {
    !is.null(f) && !isTRUE(f$metadata$skipped)
  }, outer_fold_results)
  
  # Aggregate performance
  aggregated_perf <- list(
    FGR_MRS      = aggregate_cv_performance_threefg(outer_fold_results, model_name = "FGR_MRS"),
    FGR_CLIN     = aggregate_cv_performance_threefg(outer_fold_results, model_name = "FGR_CLIN"),
    FGR_COMBINED = aggregate_cv_performance_threefg(outer_fold_results, model_name = "FGR_COMBINED")
  )
  
  aggregated_perf_all[[cohort]] <- aggregated_perf
}

library(gridExtra)
library(grid)

p1 <- plot_time_dependent_auc(perf_results = aggregated_perf_all[["ERpHER2n"]], cohort = "ERpHER2n", x_max = 10)
p2 <- plot_time_dependent_auc(perf_results = aggregated_perf_all[["TNBC"]],     cohort = "TNBC",     x_max = 10)
p3 <- plot_time_dependent_auc(perf_results = aggregated_perf_all[["All"]],      cohort = "All",      x_max = 10, legend = "yes")

pdf("~/Desktop/auc_all_cohorts.pdf", width = 11, height = 13)
grid.arrange(p3, p1, p2, ncol = 1)
dev.off()


######

cohort = "ERpHER2n"
outer_fold_results <- loadRData(paths[[cohort]])
p1 <- plot_coxnet_stability_heatmap_new(
  outer_fold_results,
  min_freq = 0.4,
  cohort= cohort)

cohort = "All"
outer_fold_results <- loadRData(paths[[cohort]])
p2 <- plot_coxnet_stability_heatmap_new(
  outer_fold_results,
  min_freq = 0.2,
  cohort= cohort,
  legend="yes")

# Extract legend and strip it from p2
library(cowplot)
shared_legend <- get_legend(p2)
p2_noleg <- p2 + theme(legend.position = "none")

pdf("~/Desktop/all_heatmaps.pdf", width = 12, height = 13)
grid.arrange(
  p1, p2_noleg,
  shared_legend,
  ncol = 2,
  nrow = 2,
  layout_matrix = rbind(c(1, 2), c(3, 3)),
  heights = c(10, 1)  # narrow row for legend
)
dev.off()

#erp. 188; 50
#pdf("~/Desktop/all_heatmaps.pdf", width = 11, height = 13)
#grid.arrange(p1, p2, ncol= 2)
#dev.off()

get_stable_cpgs <- function(outer_fold_results, min_freq = 0.4) {
  completed <- which(!sapply(outer_fold_results, is.null))
  
  all_features <- unique(unlist(lapply(completed, function(i) {
    outer_fold_results[[i]]$penalized_fg$input_cpgs
  })))
  
  sel_mat <- sapply(completed, function(i) {
    as.integer(all_features %in% outer_fold_results[[i]]$penalized_fg$input_cpgs)
  })
  rownames(sel_mat) <- all_features
  
  sel_freq <- rowSums(sel_mat) / ncol(sel_mat)
  names(sel_freq[sel_freq >= min_freq])
}

stable_erp <- get_stable_cpgs(loadRData(paths[["ERpHER2n"]]), min_freq = 0.4)
stable_all <- get_stable_cpgs(loadRData(paths[["All"]]),      min_freq = 0.4)

# Overlap
overlap <- intersect(stable_erp, stable_all)

cat(sprintf("ERpHER2n stable CpGs: %d\n", length(stable_erp)))
cat(sprintf("All stable CpGs:      %d\n", length(stable_all)))
cat(sprintf("Overlap:              %d\n", length(overlap)))
cat(sprintf("Overlap %%:            %.1f%% of ERpHER2n, %.1f%% of All\n",
            100 * length(overlap) / length(stable_erp),
            100 * length(overlap) / length(stable_all)))


# check directly consitensetce in pen fg model
cohort = "All"
outer_fold_results <- loadRData(paths[[cohort]])
stability_results <- assess_finegray_stability_threefg(
  outer_fold_results
)
stability_metrics <- stability_results$stability_metrics

# Select stable non-clinical features
stable_cpgs <- stability_metrics$feature[
  stability_metrics$selection_freq >= 0.4 &
    stability_metrics$direction_consistent == TRUE
]

cat(sprintf(
  "Including CpGs with selection_freq ≥ %.2f and consistent direction: n = %d\n",
  0.4,
  length(stable_cpgs)
))


cohort = "ERpHER2n"
outer_fold_results <- loadRData(paths[[cohort]])
stability_results <- assess_finegray_stability_threefg(
  outer_fold_results
)
stability_metrics <- stability_results$stability_metrics

# Select stable non-clinical features
stable_cpgs <- stability_metrics$feature[
  stability_metrics$selection_freq >= 0.4 &
    stability_metrics$direction_consistent == TRUE
]

cat(sprintf(
  "Including CpGs with selection_freq ≥ %.2f and consistent direction: n = %d\n",
  0.4,
  length(stable_cpgs)
))



####

cohorts <- c("ERpHER2n", "TNBC", "All")
paths <- list(
  ERpHER2n = "./output/FineGray/ERpHER2n/DualMode/Unadjusted/outer_fold_results.RData",
  TNBC     = "./output/FineGray/TNBC/DualMode/Unadjusted/outer_fold_results.RData",
  All      = "./output/FineGray/All/DualMode/Unadjusted/outer_fold_results.RData"
)
r_paths <- list(
  ERpHER2n = "./output/FineGray/ERpHER2n/DualMode/Unadjusted/final_fg_results.RData",
  TNBC     = "./output/FineGray/TNBC/DualMode/Unadjusted/final_fg_results.RData",
  All      = "./output/FineGray/All/DualMode/Unadjusted/final_fg_results.RData"
)

cohort = "All"
final_results <- loadRData(r_paths[[cohort]])
outer_fold_results <- loadRData(paths[[cohort]])
p1 <- plot_penalized_fg_coefficients_new(
  cpg_coefs=final_results$mrs_construction$cpg_coefs,
  outer_fold_results = outer_fold_results,
  cohort = cohort,
  legend = "yes")

cohort = "ERpHER2n"
final_results <- loadRData(r_paths[[cohort]])
outer_fold_results <- loadRData(paths[[cohort]])
p2 <- plot_penalized_fg_coefficients_new(
  cpg_coefs=final_results$mrs_construction$cpg_coefs,
  outer_fold_results = outer_fold_results,
  cohort = cohort)


# Extract legend and strip it from p2
library(cowplot)
shared_legend <- get_legend(p1)
p1_noleg <- p1 + theme(legend.position = "none")

library(grid)

library(cowplot)

pdf("~/Desktop/all_penFG.pdf", width = 12, height = 13)
plot_grid(
  plot_grid(p2, p1_noleg, ncol = 2),
  shared_legend,
  ncol = 1,
  rel_heights = c(10, 1)
)
dev.off()


cohort <- "ERpHER2n"
final_results <- loadRData(r_paths[[cohort]])
final_results_erp <- final_results
vimp_erp <- final_results$final_model$var_effects

cohort <- "All"
final_results <- loadRData(r_paths[[cohort]])
final_results_all <- final_results
vimp_all <- final_results$final_model$var_effects

# Rename BEFORE setting factor levels
vimp_erp$feature <- gsub("methylation_risk_score", "MeRS", vimp_erp$feature)
vimp_all$feature <- gsub("methylation_risk_score", "MeRS", vimp_all$feature)

# Define canonical order from ERpHER2n AFTER renaming
base_order <- vimp_erp$feature[order(abs(vimp_erp$coefficient), decreasing = FALSE)]

# Extra vars in All not in ERpHER2n go at bottom
extra_vars <- setdiff(vimp_all$feature, base_order)
shared_order <- c(extra_vars, base_order)

# Now set factor levels
vimp_erp$feature <- factor(vimp_erp$feature, levels = shared_order)
vimp_all$feature <- factor(vimp_all$feature, levels = shared_order)

p1 <- plot_fg_forest_hr_new(vimp_erp, cohort = "ERpHER2n")
p2 <- plot_fg_forest_hr_new(vimp_all, cohort = "All")

pdf("~/Desktop/all_finFG.pdf", width = 11, height = 5)
grid.arrange(p1, p2, ncol = 2)
dev.off()

#pdf("~/Desktop/all_finFG.pdf", width = 11, height = 8)
#grid.arrange(p1, p2, ncol= 2)
#dev.off()

cpg_features_all <- final_results_all$mrs_construction$selected_cpgs
cpg_features_erp <- final_results_erp$mrs_construction$selected_cpgs

annotated_cpgs_all <- cpg_anno %>%
  filter(illuminaID %in% cpg_features_all) %>%
  arrange(match(illuminaID, cpg_features_all))

annotated_cpgs_erp <- cpg_anno %>%
  filter(illuminaID %in% cpg_features_erp) %>%
  arrange(match(illuminaID, cpg_features_erp))

################################################################################
# HEATMAP: CpG–CpG CORRELATION (M-values, CENTERED AT 0)
################################################################################
# Subset CpGs ONLY (after loading)


cpg_features <- final_results$mrs_construction$selected_cpgs
beta_cpg <- beta_matrix[, cpg_features, drop = FALSE]

# Same transformation as pipeline
mval_cpg <- beta_to_m(beta_cpg, beta_threshold = 0.001)


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
library(pheatmap)
pheatmap(
  corr_mat,
  color = colors,
  breaks = breaks,
  clustering_distance_rows = "correlation",
  clustering_distance_cols = "correlation",
  border_color = NA,
  fontsize = 6,
  main = "Spearman correlation of CpGs (M-values)",
  filename = file.path(plot_dir, "cpg_correlation_heatmap_centered0.pdf"),
  width = 8,
  height = 8
)

################################################################################
# PLOT 3: FEATURE CLASS BARPLOT
################################################################################


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
  file.path(plot_dir, "feature_class_barplot.pdf"),
  width = 6,
  height = 4
)

annotated_cpgs <- annotated_cpgs_all
feature_counts <- annotated_cpgs %>%
  count(featureClass)

p <- ggplot(feature_counts, aes(x = "", y = n, fill = featureClass)) +
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
  file.path(plot_dir, "feature_class_piechart.pdf"),
  p,
  width = 6,
  height = 6
)

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
  filename = file.path(plot_dir, "cpg_beta_heatmap_rowz.pdf"),
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
plot_penalized_fg_coefficients <- function(cpg_coefs, outer_fold_results, plot_dir = NULL) {
  
  cat("Creating Penalized FG CpG coefficients (MRS) plot...\n")
  
  df <- data.frame(
    feature = names(cpg_coefs),
    coefficient = as.numeric(cpg_coefs),
    stringsAsFactors = FALSE
  )
  
  # Collect CoxNet sources across folds
  rfi_cpgs <- unique(unlist(lapply(outer_fold_results, function(f) {
    f$coxnet$features_rfi
  })))
  death_cpgs <- unique(unlist(lapply(outer_fold_results, function(f) {
    f$coxnet$features_death
  })))
  
  # Assign source
  df$source <- "RFI CoxNet"
  df$source[df$feature %in% death_cpgs] <- "Death CoxNet"
  df$source[df$feature %in% rfi_cpgs & df$feature %in% death_cpgs] <- "Both"
  df$source <- factor(df$source, levels = c("RFI CoxNet", "Death CoxNet", "Both"))
  
  df <- df[order(df$coefficient), ]
  df$feature <- factor(df$feature, levels = df$feature)
  
  max_abs <- max(abs(df$coefficient))
  axis_lim <- ceiling(max_abs * 1.1 * 10) / 10
  
  p <- ggplot(df, aes(x = coefficient, y = feature, color = source)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray50", linewidth = 1) +
    geom_point(size = 6) +
    scale_x_continuous(limits = c(-axis_lim, axis_lim)) +
    scale_color_manual(
      values = c(
        "RFI CoxNet"   = "#D73027",
        "Death CoxNet" = "#4575B4",
        "Both"         = "#7B3294"
      ),
      name = "Outer-CV CoxNet selection"
    ) +
    labs(
      title    = "MeRS construction: Penalized Fine-Gray Model",
      subtitle = sprintf("Stable CpGs (n=%d); selected ≥40%% of nCV folds", nrow(df)),
      x = "Coefficient",
      y = ""
    ) +
    get_publication_theme() +
    theme(
      axis.text.y    = element_text(size = 10),
      axis.text.x    = element_text(size = 10),
      axis.title.x   = element_text(size = 16),
      axis.title.y   = element_text(size = 16),
      plot.title     = element_text(size = 18, face = "bold"),
      plot.subtitle  = element_text(size = 16, color = "gray30"),
      legend.title   = element_text(size = 16, face = "bold"),
      legend.text    = element_text(size = 14),
      legend.position      = c(0.98, 0.02),
      legend.justification = c(1, 0),
      legend.background    = element_rect(fill = "white", color = "gray80",
                                          linewidth = 0.3)
    )
  
  if (!is.null(plot_dir)) {
    if (!dir.exists(plot_dir)) dir.create(plot_dir, recursive = TRUE)
    height <- max(6, nrow(df) * 0.3)
    ggsave(file.path(plot_dir, "penalized_fg_coefficients.pdf"),
           p, width = 9, height = height, limitsize = FALSE)
    ggsave(file.path(plot_dir, "penalized_fg_coefficients.png"),
           p, width = 9, height = height, dpi = 300, limitsize = FALSE)
    cat(sprintf("  Saved: penalized_fg_coefficients.pdf/.png (%d CpGs)\n\n", nrow(df)))
  }
  
  invisible(p)
}
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

go_bp_2 <- go_results %>%
  filter(ONTOLOGY == "BP", P.DE < 0.05)

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

