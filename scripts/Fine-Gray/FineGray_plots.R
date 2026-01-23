#!/usr/bin/env Rscript

################################################################################
# Publication Figures: Fine-Gray Competing Risks Model Results
# Author: Lennart Hohmann
#
# WHAT THIS SCRIPT DOES:
# ======================
# Generates comprehensive visualizations for Fine-Gray competing risks models:
# 1. Feature selection stability across CV folds
# 2. Variable importance (hazard ratios with confidence intervals)
# 3. Feature selection comparison (RFI vs Death models)
# 4. CoxNet coefficient comparison
# 5. Model performance over time (AUC and Brier scores)
# 6. Performance variability across folds
# 7. ROC curves at key time points
# 8. Calibration plots
#
# Works for both:
# - Methylation-only models (CpG features only)
# - Combined models (CpG features + clinical variables)
################################################################################

################################################################################
# LOAD LIBRARIES
################################################################################

library(ggplot2)
library(reshape2)
library(gridExtra)
library(RColorBrewer)
library(pROC)

################################################################################
# PARAMETER SETTINGS - CHANGE THESE TO SWITCH BETWEEN ANALYSES
################################################################################

# Set your cohort and data type here
COHORT <- "ERpHER2n"           # Options: "TNBC", "ERpHER2n", "All"
DATA_TYPE <- "Methylation"   # Options: "Methylation", "Combined"

# Construct the path to results based on parameters
# This follows the structure: output/FineGray/{COHORT}/{DATA_TYPE}/Unadjusted/
setwd("~/PhD_Workspace/PredictRecurrence/")

results_path <- file.path(
  "./output/FineGray",
  COHORT,
  DATA_TYPE,
  "Unadjusted/final_results/all_results.RData"
)

# Print settings
cat(sprintf("\n========================================\n"))
cat(sprintf("PLOTTING PARAMETERS\n"))
cat(sprintf("========================================\n"))
cat(sprintf("Cohort:      %s\n", COHORT))
cat(sprintf("Data type:   %s\n", DATA_TYPE))
cat(sprintf("Results:     %s\n", results_path))
cat(sprintf("========================================\n\n"))

################################################################################
# HELPER FUNCTION: LOAD SPECIFIC OBJECTS FROM .RData FILE
################################################################################

#' Load specific objects from an .RData file
#'
#' This function loads only the specified objects from an .RData file,
#' avoiding namespace pollution from load().
#'
#' @param file Path to .RData file
#' @param objects Character vector of object names to extract
#' @return Named list containing the requested objects
#' @examples
#' results <- load_rdata_objects("results.RData", c("model", "data"))
#' model <- results$model
load_rdata_objects <- function(file, objects) {
  # Create temporary environment to load into
  temp_env <- new.env()
  
  # Load .RData file into temp environment
  load(file, envir = temp_env)
  
  # Check which requested objects exist
  available <- ls(temp_env)
  missing <- setdiff(objects, available)
  
  if (length(missing) > 0) {
    warning(sprintf(
      "The following objects were not found in %s:\n  %s\n",
      basename(file),
      paste(missing, collapse = ", ")
    ))
  }
  
  # Extract requested objects that exist
  objects_to_get <- intersect(objects, available)
  result <- mget(objects_to_get, envir = temp_env)
  
  return(result)
}

################################################################################
# LOAD DATA
################################################################################

# Check if file exists
if (!file.exists(results_path)) {
  stop(sprintf("Results file not found: %s\n", results_path))
}

cat("Loading data from:", results_path, "\n")

# Define which objects we need from the .RData file
required_objects <- c(
  # Feature information
  "features_pooled_all",      # All features in final model
  "features_rfi_all",         # Features selected by RFI CoxNet
  "features_death_all",       # Features selected by Death CoxNet
  
  # Model results
  "vimp_fg_final",            # Variable importance from final model
  "coef_comparison_final",    # Coefficient comparison across models
  
  # Performance metrics
  "perf_results",             # Aggregated performance across folds
  "cv_predictions",           # Predictions on test sets from CV
  
  # Cross-validation results
  "outer_fold_results",       # Results from each CV fold
  "N_OUTER_FOLDS",            # Number of outer folds
  
  # Evaluation settings
  "EVAL_TIMES",               # Time points for evaluation
  
  # Clinical variable info (may be NULL in methylation-only mode)
  "clinvars"                  # Clinical variable names
)

# Load only the required objects
results <- load_rdata_objects(results_path, required_objects)

# Extract objects into workspace for easier access
# (This is explicit and clear about what we're using)
features_pooled_all <- results$features_pooled_all
features_rfi_all <- results$features_rfi_all
features_death_all <- results$features_death_all
vimp_fg_final <- results$vimp_fg_final
coef_comparison_final <- results$coef_comparison_final
perf_results <- results$perf_results
cv_predictions <- results$cv_predictions
outer_fold_results <- results$outer_fold_results
N_OUTER_FOLDS <- results$N_OUTER_FOLDS
EVAL_TIMES <- results$EVAL_TIMES
clinvars <- results$clinvars

cat("Successfully loaded required objects\n")

# Check fold completion status
completed_fold_indices <- which(!sapply(outer_fold_results, is.null))
n_completed_folds <- length(completed_fold_indices)

cat(sprintf("CV Folds: %d/%d completed successfully\n", 
            n_completed_folds, N_OUTER_FOLDS))
if (n_completed_folds < N_OUTER_FOLDS) {
  skipped_folds <- setdiff(1:N_OUTER_FOLDS, completed_fold_indices)
  cat(sprintf("  WARNING: Folds %s were skipped (likely no features selected)\n",
              paste(skipped_folds, collapse = ", ")))
}
cat("\n")

# Create output directory for plots
plot_dir <- dirname(results_path)
dir.create(file.path(plot_dir, "figures"), showWarnings = FALSE)

################################################################################
# SET PLOTTING THEME
################################################################################

# Publication-quality theme with clean, minimal design
theme_pub <- theme_bw(base_size = 12) +
  theme(
    panel.grid.minor = element_blank(),
    strip.background = element_rect(fill = "white"),
    legend.position = "right"
  )

################################################################################
# DETERMINE DATA MODE
################################################################################

# Check if clinical variables are present
# In methylation-only mode, clinvars will be NULL
HAS_CLINICAL <- !is.null(clinvars) && length(clinvars) > 0

cat(sprintf("Data mode detected: %s\n", 
            ifelse(HAS_CLINICAL, "Combined (Methylation + Clinical)", 
                   "Methylation only")))
if (HAS_CLINICAL) {
  cat(sprintf("Clinical variables: %s\n", paste(clinvars, collapse = ", ")))
}
cat("\n")

################################################################################
# FIGURE 1: FEATURE SELECTION STABILITY HEATMAP
################################################################################
# 
# WHAT THIS PLOT SHOWS:
# ---------------------
# Binary heatmap showing which features were selected in each CV fold.
# - Each row = one feature in the final model
# - Each column = one CV fold
# - Blue = feature was selected in that fold
# - Grey = feature was not selected in that fold
#
# WHY IT'S USEFUL:
# ----------------
# Shows the stability of feature selection across cross-validation.
# Features selected in all/most folds are more robust.
# Features selected in only 1-2 folds may be unstable.
################################################################################

cat("Creating Figure 1: Feature selection stability heatmap...\n")

# Get all features from final Fine-Gray model
all_fg_features <- features_pooled_all

# Get indices of folds that were actually completed
# (some folds may have been skipped if no features were selected)
completed_fold_indices <- which(!sapply(outer_fold_results, is.null))
n_completed_folds <- length(completed_fold_indices)

if (n_completed_folds == 0) {
  cat("  WARNING: No completed folds found. Skipping stability heatmap.\n\n")
} else {
  # Create binary matrix: 1 if feature selected in fold, 0 if not
  # Only iterate over folds that actually exist
  selection_matrix <- sapply(completed_fold_indices, function(i) {
    fold_features <- outer_fold_results[[i]]$features_pooled
    as.integer(all_fg_features %in% fold_features)
  })
  
  rownames(selection_matrix) <- all_fg_features
  colnames(selection_matrix) <- paste0("Fold ", completed_fold_indices)
  
  # Calculate selection frequency for ordering (most stable at top)
  selection_freq <- rowMeans(selection_matrix)
  ordered_features <- names(sort(selection_freq, decreasing = TRUE))
  
  # Reshape for ggplot2
  sel_melted <- melt(selection_matrix)
  colnames(sel_melted) <- c("Feature", "Fold", "Selected")
  
  # Order features by selection frequency
  sel_melted$Feature <- factor(sel_melted$Feature, levels = rev(ordered_features))
  
  # Create heatmap
  p1 <- ggplot(sel_melted, aes(x = Fold, y = Feature, fill = factor(Selected))) +
    geom_tile(color = "white", size = 0.5) +
    scale_fill_manual(
      values = c("0" = "grey90", "1" = "steelblue"),
      labels = c("Not selected", "Selected"),
      name = ""
    ) +
    labs(
      title = "Feature Selection Stability Across CV Folds",
      subtitle = paste0("All ", length(all_fg_features), " features in final Fine-Gray model"),
      x = "Cross-Validation Fold",
      y = "Feature"
    ) +
    theme_pub +
    theme(
      axis.text.y = element_text(size = 7),
      panel.grid = element_blank()
    )
  
  # Save plot
  ggsave(
    file.path(plot_dir, "figures", "1_stability_heatmap.pdf"),
    p1, width = 8, height = max(10, length(all_fg_features) * 0.25)
  )
  
  ggsave(
    file.path(plot_dir, "figures", "1_stability_heatmap.png"),
    p1, width = 8, height = max(10, length(all_fg_features) * 0.25), dpi = 300
  )
  
  cat("  >> Saved: 1_stability_heatmap.pdf/.png\n\n")
}

################################################################################
# FIGURE 2: VARIABLE IMPORTANCE (FOREST PLOT)
################################################################################
#
# WHAT THIS PLOT SHOWS:
# ---------------------
# Forest plot of hazard ratios (HR) from the final Fine-Gray model.
# - Each point = one feature's hazard ratio
# - Horizontal lines = 95% confidence intervals
# - X-axis on log scale (HR=1 means no effect)
# - Features ordered by statistical significance (Wald z-score)
#
# INTERPRETATION:
# ---------------
# - HR > 1: Feature increases risk of recurrence
# - HR < 1: Feature decreases risk of recurrence
# - HR = 1 (dashed line): No effect
# - Wider CI = less certain estimate
#
# COLOR CODING (if combined mode):
# --------------------------------
# Shows which CoxNet model(s) selected each feature:
# - Red: Selected by RFI model only
# - Blue: Selected by Death model only
# - Purple: Selected by both models
#
# SHAPE CODING (if combined mode):
# --------------------------------
# - Circle: Continuous variable (per SD increase)
# - Square: Categorical variable (vs reference category)
################################################################################

cat("Creating Figure 2: Variable importance forest plot...\n")

# Use variable importance from final model
vimp_all <- vimp_fg_final

# Add CoxNet selection annotation (which model selected each feature)
vimp_all$selected_rfi <- vimp_all$feature %in% features_rfi_all
vimp_all$selected_death <- vimp_all$feature %in% features_death_all

vimp_all$selection_source <- ifelse(
  vimp_all$selected_rfi & vimp_all$selected_death, "Both",
  ifelse(vimp_all$selected_rfi, "RFI only", "Death only")
)

# Add variable type labels (categorical vs continuous)
# In methylation-only mode, all features are continuous (no type column)
if ("type" %in% colnames(vimp_all)) {
  vimp_all$var_label <- ifelse(
    vimp_all$type == "categorical",
    paste0(vimp_all$feature, " (cat)"),
    vimp_all$feature
  )
} else {
  # Methylation-only mode: just use feature names
  vimp_all$var_label <- vimp_all$feature
  vimp_all$type <- "continuous"  # Add type column for consistency
}

# Calculate 95% confidence intervals for hazard ratios
vimp_all$HR_lower <- exp(vimp_all$coefficient - 1.96 * vimp_all$se)
vimp_all$HR_upper <- exp(vimp_all$coefficient + 1.96 * vimp_all$se)

# Order features by Wald z-score (already sorted in vimp_fg_final)
vimp_all$var_label <- factor(vimp_all$var_label, levels = rev(vimp_all$var_label))

# Create forest plot
p2 <- ggplot(vimp_all, aes(x = HR, y = var_label)) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "grey50") +
  geom_errorbarh(
    aes(xmin = HR_lower, xmax = HR_upper),
    height = 0.3, size = 0.6, color = "grey40"
  ) +
  geom_point(aes(fill = selection_source, shape = type), size = 3) +
  scale_x_log10(
    breaks = c(0.25, 0.5, 1, 2, 4, 8, 16),
    labels = c("0.25", "0.5", "1", "2", "4", "8", "16")
  ) +
  scale_fill_manual(
    values = c("RFI only" = "#E41A1C", 
               "Both" = "#984EA3", 
               "Death only" = "#377EB8"),
    name = "Selected by\nCoxNet"
  ) +
  scale_shape_manual(
    values = c("categorical" = 22, "continuous" = 21),
    name = "Variable Type",
    labels = c("Categorical (vs ref)", "Continuous (per SD)")
  ) +
  labs(
    title = "Variable Importance in Final Fine-Gray Model",
    subtitle = paste0("All ", nrow(vimp_all), " features, ordered by Wald z-score"),
    x = "Hazard Ratio (95% CI)",
    y = ""
  ) +
  theme_pub +
  theme(
    legend.position = "right",
    axis.text.y = element_text(size = 7)
  )

# Add guides - only show shape legend if we have categorical variables
if (HAS_CLINICAL && any(vimp_all$type == "categorical")) {
  p2 <- p2 + guides(
    fill = guide_legend(override.aes = list(shape = 21, size = 4)),
    shape = guide_legend(override.aes = list(size = 4))
  )
} else {
  # Methylation-only: hide shape legend (all continuous)
  p2 <- p2 + guides(
    fill = guide_legend(override.aes = list(shape = 21, size = 4)),
    shape = "none"
  )
}

ggsave(
  file.path(plot_dir, "figures", "2_variable_importance_forest.pdf"),
  p2, width = 10, height = max(10, nrow(vimp_all) * 0.25)
)

ggsave(
  file.path(plot_dir, "figures", "2_variable_importance_forest.png"),
  p2, width = 10, height = max(10, nrow(vimp_all) * 0.25), dpi = 300
)

cat("  >> Saved: 2_variable_importance_forest.pdf/.png\n\n")

################################################################################
# FIGURE 3: FEATURE SELECTION BY MODEL (BAR PLOT)
################################################################################
#
# WHAT THIS PLOT SHOWS:
# ---------------------
# Bar plot comparing feature selection between CoxNet models.
# Shows how many features were selected by:
# - RFI model only (red)
# - Death model only (blue)
# - Both models (purple)
#
# WHY IT'S USEFUL:
# ----------------
# Reveals overlap between features predictive of recurrence vs death.
# - High overlap: Same features predict both outcomes
# - Low overlap: Different biological processes drive each outcome
# - RFI-only features: Specifically associated with recurrence risk
# - Death-only features: Specifically associated with competing mortality risk
################################################################################

cat("Creating Figure 3: Feature selection comparison...\n")

# Get features selected by each model
rfi_features <- features_rfi_all
death_features <- features_death_all
pooled_features <- features_pooled_all

# Classify each feature by which model(s) selected it
feature_source <- data.frame(
  feature = pooled_features,
  selected_rfi = pooled_features %in% rfi_features,
  selected_death = pooled_features %in% death_features,
  stringsAsFactors = FALSE
)

feature_source$category <- ifelse(
  feature_source$selected_rfi & feature_source$selected_death,
  "Both",
  ifelse(feature_source$selected_rfi, "RFI only", "Death only")
)

# Count features in each category
category_counts <- table(feature_source$category)

# Prepare data for plotting
category_df <- data.frame(
  category = names(category_counts),
  count = as.vector(category_counts)
)

# Order categories for consistent plotting
category_df$category <- factor(
  category_df$category, 
  levels = c("RFI only", "Both", "Death only")
)

# Create bar plot
p3 <- ggplot(category_df, aes(x = category, y = count, fill = category)) +
  geom_bar(stat = "identity", width = 0.7) +
  geom_text(aes(label = count), vjust = -0.5, size = 5) +
  scale_fill_manual(
    values = c("RFI only" = "#E41A1C", 
               "Both" = "#984EA3", 
               "Death only" = "#377EB8")
  ) +
  labs(
    title = "Feature Selection by CoxNet Model",
    subtitle = paste0("Total pooled features: ", length(pooled_features)),
    x = "Selected by",
    y = "Number of Features"
  ) +
  theme_pub +
  theme(legend.position = "none") +
  ylim(0, max(category_counts) * 1.15)

ggsave(
  file.path(plot_dir, "figures", "3_feature_selection_by_model.pdf"),
  p3, width = 8, height = 6
)

ggsave(
  file.path(plot_dir, "figures", "3_feature_selection_by_model.png"),
  p3, width = 8, height = 6, dpi = 300
)

cat("  >> Saved: 3_feature_selection_by_model.pdf/.png\n\n")

################################################################################
# FIGURE 4: COXNET COEFFICIENT COMPARISON (RFI vs DEATH)
################################################################################
#
# WHAT THIS PLOT SHOWS:
# ---------------------
# Scatter plot comparing coefficients from the two CoxNet models.
# - Red points: Coefficients from RFI model
# - Blue points: Coefficients from Death model
# - X-axis: Coefficient value (effect size)
# - Features ordered by absolute RFI coefficient
#
# INTERPRETATION:
# ---------------
# - Positive coefficient: Increases risk
# - Negative coefficient: Decreases risk
# - Points far from zero: Strong effects
# - Compare red vs blue positions to see if features have
#   similar effects on both outcomes or divergent effects
#
# WHY IT'S USEFUL:
# ----------------
# Shows whether features have consistent effects across outcomes:
# - Similar positions: Feature affects both outcomes similarly
# - Opposite positions: Feature has opposite effects on RFI vs Death
# - One model only: Feature is outcome-specific
################################################################################

cat("Creating Figure 4: CoxNet coefficient comparison...\n")

# Get coefficient comparison from final model
all_coef <- coef_comparison_final

# Only include features selected by at least one CoxNet model
# (Filter out features with zero coefficients in both models)
all_coef_filtered <- all_coef[
  all_coef$cox_rfi_final_coef != 0 | all_coef$cox_death_final_coef != 0, 
]

# Reshape data for plotting (wide to long format)
coef_long <- data.frame(
  feature = rep(all_coef_filtered$feature, 2),
  model = rep(c("CoxNet RFI", "CoxNet Death"), each = nrow(all_coef_filtered)),
  coefficient = c(all_coef_filtered$cox_rfi_final_coef, 
                  all_coef_filtered$cox_death_final_coef)
)

# Remove zeros for cleaner visualization
coef_long <- coef_long[coef_long$coefficient != 0, ]

# Order features by absolute RFI coefficient (strongest effects first)
feature_order <- all_coef_filtered$feature[
  order(abs(all_coef_filtered$cox_rfi_final_coef), decreasing = TRUE)
]
coef_long$feature <- factor(coef_long$feature, levels = rev(feature_order))

# Create coefficient comparison plot
p4 <- ggplot(coef_long, aes(x = coefficient, y = feature, color = model)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
  geom_point(size = 3, position = position_dodge(width = 0.5)) +
  scale_color_manual(
    values = c("CoxNet RFI" = "#E41A1C", 
               "CoxNet Death" = "#377EB8"),
    name = "Model"
  ) +
  labs(
    title = "CoxNet Coefficient Comparison",
    subtitle = "Features selected by elastic net for RFI or Death outcomes",
    x = "Coefficient",
    y = ""
  ) +
  theme_pub +
  theme(
    legend.position = c(0.85, 0.15),
    axis.text.y = element_text(size = 7)
  )

ggsave(
  file.path(plot_dir, "figures", "4_coxnet_coefficient_comparison.pdf"),
  p4, width = 10, height = max(8, nrow(all_coef_filtered) * 0.3)
)

ggsave(
  file.path(plot_dir, "figures", "4_coxnet_coefficient_comparison.png"),
  p4, width = 10, height = max(8, nrow(all_coef_filtered) * 0.3), dpi = 300
)

cat("  >> Saved: 4_coxnet_coefficient_comparison.pdf/.png\n\n")

################################################################################
# FIGURE 5: PERFORMANCE OVER TIME (AUC AND BRIER SCORE)
################################################################################
#
# WHAT THIS PLOT SHOWS:
# ---------------------
# Two-panel plot showing model performance across follow-up time:
#
# TOP PANEL - Time-Dependent AUC:
# - Y-axis: AUC (Area Under ROC Curve)
# - X-axis: Years of follow-up
# - Blue line: Mean AUC across CV folds
# - Shaded area: 95% confidence interval
# - Red dashed line: Random prediction (AUC = 0.5)
#
# BOTTOM PANEL - Time-Dependent Brier Score:
# - Y-axis: Brier score (prediction error)
# - X-axis: Years of follow-up
# - Green line: Mean Brier score across CV folds
# - Shaded area: 95% confidence interval
#
# INTERPRETATION:
# ---------------
# AUC:
# - Higher is better (max = 1.0)
# - AUC > 0.7: Acceptable discrimination
# - AUC > 0.8: Excellent discrimination
#
# Brier Score:
# - Lower is better (min = 0)
# - Measures calibration and discrimination combined
# - <0.25 is generally considered good
#
# WHY IT'S USEFUL:
# ----------------
# Shows how well the model predicts at different time points.
# Declining performance over time may indicate:
# - Need for time-varying effects
# - Emergence of new risk factors
# - Increasing heterogeneity in outcomes
################################################################################

cat("Creating Figure 5: Performance over time...\n")

# Extract performance summary
perf_summary <- perf_results$summary

# Separate AUC and Brier metrics
auc_metrics <- perf_summary[grep("^auc_", perf_summary$metric), ]
brier_metrics <- perf_summary[grep("^brier_", perf_summary$metric), ]

# Extract time points from metric names (e.g., "auc_3yr" -> 3)
auc_metrics$time <- as.numeric(gsub("auc_(\\d+)yr", "\\1", auc_metrics$metric))
brier_metrics$time <- as.numeric(gsub("brier_(\\d+)yr", "\\1", brier_metrics$metric))

# Plot AUC over time
p5a <- ggplot(auc_metrics, aes(x = time, y = mean)) +
  geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper), alpha = 0.2, fill = "steelblue") +
  geom_line(color = "steelblue", size = 1) +
  geom_point(color = "steelblue", size = 3) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "red", alpha = 0.5) +
  scale_x_continuous(breaks = unique(auc_metrics$time)) +
  ylim(0, 1) +
  labs(
    title = "Time-Dependent AUC",
    subtitle = "Mean ± 95% CI across CV folds",
    x = "Time (years)",
    y = "AUC"
  ) +
  theme_pub

# Plot Brier score over time
p5b <- ggplot(brier_metrics, aes(x = time, y = mean)) +
  geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper), alpha = 0.2, fill = "darkgreen") +
  geom_line(color = "darkgreen", size = 1) +
  geom_point(color = "darkgreen", size = 3) +
  scale_x_continuous(breaks = unique(brier_metrics$time)) +
  labs(
    title = "Time-Dependent Brier Score",
    subtitle = "Mean ± 95% CI across CV folds (lower is better)",
    x = "Time (years)",
    y = "Brier Score"
  ) +
  theme_pub

# Combine into one figure
p5 <- grid.arrange(p5a, p5b, ncol = 1)

ggsave(
  file.path(plot_dir, "figures", "5_performance_over_time.pdf"),
  p5, width = 10, height = 10
)

ggsave(
  file.path(plot_dir, "figures", "5_performance_over_time.png"),
  p5, width = 10, height = 10, dpi = 300
)

cat("  >> Saved: 5_performance_over_time.pdf/.png\n\n")

################################################################################
# FIGURE 6: PERFORMANCE VARIABILITY (BOXPLOTS)
################################################################################
#
# WHAT THIS PLOT SHOWS:
# ---------------------
# Boxplots showing performance variation across CV folds at key time points.
#
# LEFT PANEL - AUC Variability:
# - Each boxplot = distribution of AUC across 5 CV folds
# - Individual points = AUC from each fold
# - Box shows 25th-75th percentiles
# - Line in box = median
#
# RIGHT PANEL - Brier Score Variability:
# - Same structure as AUC panel
# - Shows prediction error variability
#
# WHY IT'S USEFUL:
# ----------------
# Assesses model stability:
# - Narrow boxes: Consistent performance across folds
# - Wide boxes: Performance varies by training set
# - Outlier points: Folds with unusual performance
#
# High variability suggests:
# - Model is sensitive to training data composition
# - May need more data or different features
# - Some patient subgroups are harder to predict
################################################################################

cat("Creating Figure 6: Performance variability boxplots...\n")

# Get performance from all folds
all_folds <- perf_results$all_folds

# Select key time points for visualization
key_times <- c(1, 3, 5, 10)
auc_cols <- paste0("auc_", key_times, "yr")
brier_cols <- paste0("brier_", key_times, "yr")

# Check which time points actually exist in the data
auc_cols <- auc_cols[auc_cols %in% names(all_folds)]
brier_cols <- brier_cols[brier_cols %in% names(all_folds)]

# Extract AUC and Brier data
auc_key <- all_folds[, c("model", auc_cols)]
brier_key <- all_folds[, c("model", brier_cols)]

# Reshape for ggplot (wide to long format)
auc_long <- reshape2::melt(auc_key, id.vars = "model", 
                           variable.name = "time", value.name = "AUC")
auc_long$time <- gsub("auc_(\\d+)yr", "\\1 yr", auc_long$time)

brier_long <- reshape2::melt(brier_key, id.vars = "model", 
                             variable.name = "time", value.name = "Brier")
brier_long$time <- gsub("brier_(\\d+)yr", "\\1 yr", brier_long$time)

# Create AUC boxplot
p6a <- ggplot(auc_long, aes(x = time, y = AUC)) +
  geom_boxplot(fill = "steelblue", alpha = 0.7, outlier.shape = NA) +
  geom_jitter(width = 0.1, alpha = 0.5, size = 2) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "red") +
  ylim(0, 1) +
  labs(
    title = "AUC Variability Across CV Folds",
    x = "Time Point",
    y = "AUC"
  ) +
  theme_pub

# Create Brier score boxplot
p6b <- ggplot(brier_long, aes(x = time, y = Brier)) +
  geom_boxplot(fill = "darkgreen", alpha = 0.7, outlier.shape = NA) +
  geom_jitter(width = 0.1, alpha = 0.5, size = 2) +
  labs(
    title = "Brier Score Variability Across CV Folds",
    x = "Time Point",
    y = "Brier Score"
  ) +
  theme_pub

# Combine into one figure
p6 <- grid.arrange(p6a, p6b, ncol = 2)

ggsave(
  file.path(plot_dir, "figures", "6_performance_variability.pdf"),
  p6, width = 12, height = 6
)

ggsave(
  file.path(plot_dir, "figures", "6_performance_variability.png"),
  p6, width = 12, height = 6, dpi = 300
)

cat("  >> Saved: 6_performance_variability.pdf/.png\n\n")

################################################################################
# FIGURE 7: ROC CURVES AT KEY TIME POINTS
################################################################################
#
# WHAT THIS PLOT SHOWS:
# ---------------------
# Receiver Operating Characteristic (ROC) curves at multiple time points.
# Each colored line represents a different follow-up time (1, 3, 5, 10 years).
#
# ROC CURVE BASICS:
# -----------------
# - X-axis: False Positive Rate (1 - Specificity)
#   = Proportion of non-events incorrectly classified as events
# - Y-axis: True Positive Rate (Sensitivity)
#   = Proportion of events correctly identified
# - Diagonal dashed line: Random prediction (no discrimination)
# - Curve closer to top-left corner = better discrimination
# - AUC value shown in legend for each time point
#
# INTERPRETATION:
# ---------------
# Perfect prediction: Curve follows left edge then top edge (AUC = 1.0)
# Random prediction: Curve follows diagonal (AUC = 0.5)
# Real models: Curve between these extremes
#
# WHY IT'S USEFUL:
# ----------------
# - Choose optimal risk thresholds for clinical decisions
# - Compare discrimination at different time points
# - Assess if model maintains performance over follow-up
################################################################################

cat("Creating Figure 7: ROC curves...\n")

# Select time points to plot (use common clinically-relevant times)
roc_times <- c(1, 3, 5, 10)
roc_times <- roc_times[roc_times %in% EVAL_TIMES]  # Only use available times

# Calculate ROC curve for each time point
roc_list <- list()

for (t in roc_times) {
  risk_col <- paste0("risk_", t, "yr")
  
  # Create binary outcome: Did RFI event occur by time t?
  cv_predictions$outcome_at_t <- ifelse(
    cv_predictions$time <= t & cv_predictions$rfi_event == 1, 1, 0
  )
  
  # Only include patients who could have had event by time t
  # Exclude those censored before time t (insufficient follow-up)
  eligible <- cv_predictions$time >= t | cv_predictions$rfi_event == 1
  data_t <- cv_predictions[eligible, ]
  
  # Check if we have both cases and controls
  n_cases <- sum(data_t$outcome_at_t == 1)
  n_controls <- sum(data_t$outcome_at_t == 0)
  
  if (n_cases < 2 || n_controls < 2) {
    cat(sprintf("  WARNING: Skipping %d-year ROC - insufficient cases (%d) or controls (%d)\n", 
                t, n_cases, n_controls))
    next
  }
  
  # Calculate ROC curve using pROC package
  roc_obj <- roc(
    response = data_t$outcome_at_t,
    predictor = data_t[[risk_col]],
    levels = c(0, 1),
    direction = "<",
    quiet = TRUE
  )
  
  # Store results with AUC in label
  roc_list[[as.character(t)]] <- data.frame(
    time = paste0(t, " years (AUC = ", round(auc(roc_obj), 2), ")"),
    sensitivity = roc_obj$sensitivities,
    specificity = roc_obj$specificities
  )
}

# Combine all ROC curves into one dataframe
if (length(roc_list) == 0) {
  cat("  WARNING: No ROC curves generated (insufficient data at all time points)\n")
  cat("  >> Skipped: 7_roc_curves.pdf/.png\n\n")
} else {
  roc_df <- do.call(rbind, roc_list)
  roc_df$time <- factor(roc_df$time, levels = unique(roc_df$time))
  
  # Create ROC curve plot
  p7 <- ggplot(roc_df, aes(x = 1 - specificity, y = sensitivity, color = time)) +
    geom_line(size = 1) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "grey50") +
    scale_color_brewer(palette = "Set1", name = "Time Point") +
    labs(
      title = "Time-Dependent ROC Curves",
      subtitle = "Prediction of RFI events across CV folds",
      x = "1 - Specificity (False Positive Rate)",
      y = "Sensitivity (True Positive Rate)"
    ) +
    coord_equal() +
    theme_pub +
    theme(legend.position = c(0.7, 0.3))
  
  ggsave(
    file.path(plot_dir, "figures", "7_roc_curves.pdf"),
    p7, width = 8, height = 8
  )
  
  ggsave(
    file.path(plot_dir, "figures", "7_roc_curves.png"),
    p7, width = 8, height = 8, dpi = 300
  )
  
  cat("  >> Saved: 7_roc_curves.pdf/.png\n\n")
}

################################################################################
# FIGURE 8: CALIBRATION PLOTS
################################################################################
#
# WHAT THIS PLOT SHOWS:
# ---------------------
# Calibration plots at 3, 5, and 10 years showing agreement between
# predicted and observed cumulative incidence of RFI.
#
# CALIBRATION PLOT BASICS:
# ------------------------
# - X-axis: Predicted risk from model
# - Y-axis: Observed cumulative incidence
# - Diagonal dashed line: Perfect calibration
# - Blue points: Risk groups (deciles)
# - Point size: Number of patients in group
# - Error bars: 95% confidence intervals
# - Red curve: LOESS smoothed calibration
#
# INTERPRETATION:
# ---------------
# Perfect calibration: Points lie on diagonal line
# - Model predicts 20% risk → 20% actually experience event
#
# Over-prediction: Points below diagonal
# - Model predicts higher risk than observed
#
# Under-prediction: Points above diagonal
# - Model predicts lower risk than observed
#
# WHY IT'S USEFUL:
# ----------------
# AUC measures discrimination (ranking patients correctly).
# Calibration measures accuracy of risk estimates.
# Both are needed for clinical decision-making:
# - Good discrimination + poor calibration = can rank but not quantify risk
# - Good calibration + poor discrimination = accurate average but poor ranking
# - Need both for reliable individual risk predictions
################################################################################

cat("Creating Figure 8: Calibration plots...\n")

# Select key time points for calibration assessment
calib_times <- c(3, 5, 10)
calib_times <- calib_times[calib_times %in% EVAL_TIMES]

calib_plots <- list()

for (t in calib_times) {
  risk_col <- paste0("risk_", t, "yr")
  
  # Only include patients with sufficient follow-up
  # (follow-up >= t OR had event before t)
  eligible <- cv_predictions$time >= t | cv_predictions$rfi_event == 1
  data_t <- cv_predictions[eligible, ]
  
  # Check if we have enough data for calibration
  n_events <- sum(data_t$time <= t & data_t$rfi_event == 1)
  n_total <- nrow(data_t)
  
  if (n_total < 10 || n_events < 5) {
    cat(sprintf("  WARNING: Skipping %d-year calibration - insufficient data (n=%d, events=%d)\n",
                t, n_total, n_events))
    next
  }
  
  # Divide patients into risk deciles (10 groups)
  data_t$risk_group <- cut(
    data_t[[risk_col]],
    breaks = quantile(data_t[[risk_col]], probs = seq(0, 1, 0.1), na.rm = TRUE),
    include.lowest = TRUE,
    labels = FALSE
  )
  
  # Calculate observed vs predicted for each decile
  calib_data <- data.frame()
  
  for (g in unique(data_t$risk_group)) {
    group_data <- data_t[data_t$risk_group == g, ]
    
    # Mean predicted risk in this decile
    mean_pred <- mean(group_data[[risk_col]], na.rm = TRUE)
    
    # Observed cumulative incidence
    # (This is a simplified calculation; ideally use Aalen-Johansen estimator
    #  to properly account for competing risks)
    n_total <- nrow(group_data)
    n_rfi <- sum(group_data$time <= t & group_data$rfi_event == 1)
    obs_ci <- n_rfi / n_total
    
    # Standard error for confidence intervals
    se_obs <- sqrt(obs_ci * (1 - obs_ci) / n_total)
    
    calib_data <- rbind(calib_data, data.frame(
      risk_group = g,
      predicted = mean_pred,
      observed = obs_ci,
      se = se_obs,
      n = n_total
    ))
  }
  
  # Remove missing data
  calib_data <- calib_data[!is.na(calib_data$risk_group), ]
  
  # Create calibration plot
  p_calib <- ggplot(calib_data, aes(x = predicted, y = observed)) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "grey50") +
    geom_errorbar(
      aes(ymin = pmax(0, observed - 1.96*se), 
          ymax = pmin(1, observed + 1.96*se)),
      width = 0.01, alpha = 0.5
    ) +
    geom_point(aes(size = n), alpha = 0.7, color = "steelblue") +
    geom_smooth(method = "loess", se = TRUE, color = "red", fill = "red", alpha = 0.2) +
    scale_size_continuous(name = "N patients", range = c(2, 6)) +
    labs(
      title = paste0("Calibration at ", t, " Years"),
      subtitle = "Predicted vs. observed cumulative incidence of RFI",
      x = "Predicted Risk",
      y = "Observed Cumulative Incidence"
    ) +
    coord_equal(
      xlim = c(0, max(calib_data$predicted, calib_data$observed, na.rm = TRUE)),
      ylim = c(0, max(calib_data$predicted, calib_data$observed, na.rm = TRUE))
    ) +
    theme_pub
  
  calib_plots[[as.character(t)]] <- p_calib
}

# Combine calibration plots into one figure
if (length(calib_plots) == 0) {
  cat("  WARNING: No calibration plots generated (insufficient data at all time points)\n")
  cat("  >> Skipped: 8_calibration_plots.pdf/.png\n\n")
} else if (length(calib_plots) == 3) {
  p8 <- grid.arrange(calib_plots[[1]], calib_plots[[2]], calib_plots[[3]], ncol = 3)
  ggsave(
    file.path(plot_dir, "figures", "8_calibration_plots.pdf"),
    p8, width = 15, height = 5
  )
  ggsave(
    file.path(plot_dir, "figures", "8_calibration_plots.png"),
    p8, width = 15, height = 5, dpi = 300
  )
  cat("  >> Saved: 8_calibration_plots.pdf/.png\n\n")
} else if (length(calib_plots) == 2) {
  p8 <- grid.arrange(calib_plots[[1]], calib_plots[[2]], ncol = 2)
  ggsave(
    file.path(plot_dir, "figures", "8_calibration_plots.pdf"),
    p8, width = 10, height = 5
  )
  ggsave(
    file.path(plot_dir, "figures", "8_calibration_plots.png"),
    p8, width = 10, height = 5, dpi = 300
  )
  cat("  >> Saved: 8_calibration_plots.pdf/.png\n\n")
} else {
  p8 <- calib_plots[[1]]
  ggsave(
    file.path(plot_dir, "figures", "8_calibration_plots.pdf"),
    p8, width = 5, height = 5
  )
  ggsave(
    file.path(plot_dir, "figures", "8_calibration_plots.png"),
    p8, width = 5, height = 5, dpi = 300
  )
  cat("  >> Saved: 8_calibration_plots.pdf/.png\n\n")
}

################################################################################
# SUMMARY
################################################################################

cat("\n========================================\n")
cat("PLOTTING COMPLETE\n")
cat("========================================\n\n")
cat("Output directory:", file.path(plot_dir, "figures"), "\n\n")
cat("Generated figures:\n")
cat("  1. stability_heatmap.pdf/.png\n")
cat("     - Binary heatmap of feature selection across CV folds\n")
cat("  2. variable_importance_forest.pdf/.png\n")
cat("     - Forest plot of hazard ratios with confidence intervals\n")
cat("  3. feature_selection_by_model.pdf/.png\n")
cat("     - Bar plot comparing RFI vs Death model feature selection\n")
cat("  4. coxnet_coefficient_comparison.pdf/.png\n")
cat("     - Comparison of coefficients between RFI and Death models\n")
cat("  5. performance_over_time.pdf/.png\n")
cat("     - AUC and Brier scores across follow-up time\n")
cat("  6. performance_variability.pdf/.png\n")
cat("     - Boxplots showing variation across CV folds\n")
cat("  7. roc_curves.pdf/.png\n")
cat("     - ROC curves at key time points (1, 3, 5, 10 years)\n")
cat("  8. calibration_plots.pdf/.png\n")
cat("     - Predicted vs observed risk at 3, 5, 10 years\n\n")
cat(">> All figures generated successfully!\n")
cat(sprintf(">> Cohort: %s | Data type: %s\n", COHORT, DATA_TYPE))