#!/usr/bin/env Rscript

################################################################################
# Publication Figures: Feature Selection & Importance
# Author: Lennart Hohmann
################################################################################

library(ggplot2)
library(reshape2)
library(gridExtra)
library(RColorBrewer)

# Set publication-quality theme
theme_pub <- theme_bw(base_size = 12) +
  theme(
    panel.grid.minor = element_blank(),
    strip.background = element_rect(fill = "white"),
    legend.position = "right"
  )

################################################################################
# LOAD DATA
################################################################################

# Update this path to your results
results_path <- "./output/FineGray/ERpHER2n/Combined/Unadjusted/final_results/all_results.RData"
load(results_path)

cat("Loaded data from:", results_path, "\n")
cat("Available objects:", ls(), "\n\n")

# Create output directory for plots
plot_dir <- dirname(results_path)
dir.create(file.path(plot_dir, "figures"), showWarnings = FALSE)

################################################################################
# FIGURE 1: FEATURE SELECTION STABILITY (BINARY HEATMAP)
################################################################################

cat("Creating stability heatmap...\n")

# Use ALL features from final FG model
all_fg_features <- features_pooled_all

# Create binary selection matrix (selected = 1, not selected = 0)
selection_matrix <- sapply(1:N_OUTER_FOLDS, function(i) {
  fold_features <- outer_fold_results[[i]]$features_pooled
  as.integer(all_fg_features %in% fold_features)
})

rownames(selection_matrix) <- all_fg_features
colnames(selection_matrix) <- paste0("Fold ", 1:N_OUTER_FOLDS)

# Calculate selection frequency for ordering
selection_freq <- rowMeans(selection_matrix)
ordered_features <- names(sort(selection_freq, decreasing = TRUE))

# Melt for ggplot
sel_melted <- melt(selection_matrix)
colnames(sel_melted) <- c("Feature", "Fold", "Selected")

# Order features by selection frequency
sel_melted$Feature <- factor(sel_melted$Feature, levels = rev(ordered_features))

# Create binary heatmap
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

ggsave(
  file.path(plot_dir, "figures", "stability_heatmap.pdf"),
  p1, width = 8, height = max(10, length(all_fg_features) * 0.25)
)

ggsave(
  file.path(plot_dir, "figures", "stability_heatmap.png"),
  p1, width = 8, height = max(10, length(all_fg_features) * 0.25), dpi = 300
)

################################################################################
# FIGURE 2: VARIABLE IMPORTANCE WITH COXNET SELECTION ANNOTATION
################################################################################

cat("Creating variable importance forest plot...\n")

# Use ALL features, ordered by importance
vimp_all <- vimp_fg_final

# Add CoxNet selection annotation
vimp_all$selected_rfi <- vimp_all$feature %in% features_rfi_all
vimp_all$selected_death <- vimp_all$feature %in% features_death_all

vimp_all$selection_source <- ifelse(
  vimp_all$selected_rfi & vimp_all$selected_death, "Both",
  ifelse(vimp_all$selected_rfi, "RFI only", "Death only")
)

# Add variable type labels
vimp_all$var_label <- ifelse(
  vimp_all$type == "categorical",
  paste0(vimp_all$feature, " (cat)"),
  vimp_all$feature
)

# Calculate 95% CI for HR
vimp_all$HR_lower <- exp(vimp_all$coefficient - 1.96 * vimp_all$se)
vimp_all$HR_upper <- exp(vimp_all$coefficient + 1.96 * vimp_all$se)

# Order by Wald z for plotting (already sorted)
vimp_all$var_label <- factor(vimp_all$var_label, levels = rev(vimp_all$var_label))

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
  ) +
  guides(
    fill = guide_legend(override.aes = list(shape = 21, size = 4)),
    shape = guide_legend(override.aes = list(size = 4))
  )

ggsave(
  file.path(plot_dir, "figures", "variable_importance_forest.pdf"),
  p2, width = 10, height = max(10, nrow(vimp_all) * 0.25)
)

ggsave(
  file.path(plot_dir, "figures", "variable_importance_forest.png"),
  p2, width = 10, height = max(10, nrow(vimp_all) * 0.25), dpi = 300
)

################################################################################
# FIGURE 3: FEATURE SELECTION BY MODEL (BAR PLOT)
################################################################################

cat("Creating feature selection comparison...\n")

# Get features selected by each model
rfi_features <- features_rfi_all
death_features <- features_death_all
pooled_features <- features_pooled_all

# Classify each pooled feature
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

# Count by category
category_counts <- table(feature_source$category)

# Bar plot
category_df <- data.frame(
  category = names(category_counts),
  count = as.vector(category_counts)
)

# Order for plotting
category_df$category <- factor(
  category_df$category, 
  levels = c("RFI only", "Both", "Death only")
)

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
  file.path(plot_dir, "figures", "feature_selection_by_model.pdf"),
  p3, width = 8, height = 6
)

ggsave(
  file.path(plot_dir, "figures", "feature_selection_by_model.png"),
  p3, width = 8, height = 6, dpi = 300
)

################################################################################
# FIGURE 4: COXNET COEFFICIENT COMPARISON (RFI vs Death)
################################################################################

cat("Creating CoxNet coefficient comparison plot...\n")

# Use ALL features from final model
all_coef <- coef_comparison_final

# Only include features selected by at least one CoxNet model
all_coef_filtered <- all_coef[
  all_coef$cox_rfi_final_coef != 0 | all_coef$cox_death_final_coef != 0, 
]

# Reshape for plotting
coef_long <- data.frame(
  feature = rep(all_coef_filtered$feature, 2),
  model = rep(c("CoxNet RFI", "CoxNet Death"), each = nrow(all_coef_filtered)),
  coefficient = c(all_coef_filtered$cox_rfi_final_coef, 
                  all_coef_filtered$cox_death_final_coef)
)

# Remove zeros for cleaner plot
coef_long <- coef_long[coef_long$coefficient != 0, ]

# Order features by RFI coefficient (or could use death, or max absolute)
feature_order <- all_coef_filtered$feature[
  order(abs(all_coef_filtered$cox_rfi_final_coef), decreasing = TRUE)
]
coef_long$feature <- factor(coef_long$feature, levels = rev(feature_order))

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
  file.path(plot_dir, "figures", "coxnet_coefficient_comparison.pdf"),
  p4, width = 10, height = max(8, nrow(all_coef_filtered) * 0.3)
)

ggsave(
  file.path(plot_dir, "figures", "coxnet_coefficient_comparison.png"),
  p4, width = 10, height = max(8, nrow(all_coef_filtered) * 0.3), dpi = 300
)

################################################################################
# FIGURE 5: PERFORMANCE OVER TIME
################################################################################

cat("Creating performance over time plot...\n")

# Get mean and SE for each time point
perf_summary <- perf_results$summary

# Separate AUC and Brier
auc_metrics <- perf_summary[grep("^auc_", perf_summary$metric), ]
brier_metrics <- perf_summary[grep("^brier_", perf_summary$metric), ]

# Extract time points from metric names
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

# Plot Brier over time
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

# Combine
p5 <- grid.arrange(p5a, p5b, ncol = 1)

ggsave(
  file.path(plot_dir, "figures", "performance_over_time.pdf"),
  p5, width = 10, height = 10
)

ggsave(
  file.path(plot_dir, "figures", "performance_over_time.png"),
  p5, width = 10, height = 10, dpi = 300
)

################################################################################
# FIGURE 6: PERFORMANCE VARIABILITY (BOXPLOTS)
################################################################################

cat("Creating performance variability boxplots...\n")

# Use all_folds data
all_folds <- perf_results$all_folds

# Select key time points for boxplots
key_times <- c(1, 3, 5, 10)
auc_cols <- paste0("auc_", key_times, "yr")
brier_cols <- paste0("brier_", key_times, "yr")

# Check which columns exist
auc_cols <- auc_cols[auc_cols %in% names(all_folds)]
brier_cols <- brier_cols[brier_cols %in% names(all_folds)]

auc_key <- all_folds[, c("model", auc_cols)]
brier_key <- all_folds[, c("model", brier_cols)]

# Reshape for ggplot
auc_long <- reshape2::melt(auc_key, id.vars = "model", variable.name = "time", value.name = "AUC")
auc_long$time <- gsub("auc_(\\d+)yr", "\\1 yr", auc_long$time)

brier_long <- reshape2::melt(brier_key, id.vars = "model", variable.name = "time", value.name = "Brier")
brier_long$time <- gsub("brier_(\\d+)yr", "\\1 yr", brier_long$time)

# Boxplot for AUC
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

# Boxplot for Brier
p6b <- ggplot(brier_long, aes(x = time, y = Brier)) +
  geom_boxplot(fill = "darkgreen", alpha = 0.7, outlier.shape = NA) +
  geom_jitter(width = 0.1, alpha = 0.5, size = 2) +
  labs(
    title = "Brier Score Variability Across CV Folds",
    x = "Time Point",
    y = "Brier Score"
  ) +
  theme_pub

p6 <- grid.arrange(p6a, p6b, ncol = 2)

ggsave(
  file.path(plot_dir, "figures", "performance_variability.pdf"),
  p6, width = 12, height = 6
)

ggsave(
  file.path(plot_dir, "figures", "performance_variability.png"),
  p6, width = 12, height = 6, dpi = 300
)

################################################################################
# FIGURE 7: ROC CURVES AT KEY TIME POINTS
################################################################################

cat("Creating ROC curves...\n")

library(pROC)

# Select key time points to plot
roc_times <- c(1, 3, 5, 10)
roc_times <- roc_times[roc_times %in% EVAL_TIMES]  # Only use available times

# Calculate ROC for each time point
roc_list <- list()

for (t in roc_times) {
  risk_col <- paste0("risk_", t, "yr")
  
  # Create binary outcome: did RFI occur by time t?
  cv_predictions$outcome_at_t <- ifelse(
    cv_predictions$time <= t & cv_predictions$rfi_event == 1, 1, 0
  )
  
  # Only include patients who could have had event by time t
  # (exclude those censored before time t)
  eligible <- cv_predictions$time >= t | cv_predictions$rfi_event == 1
  
  data_t <- cv_predictions[eligible, ]
  
  # Calculate ROC
  roc_obj <- roc(
    response = data_t$outcome_at_t,
    predictor = data_t[[risk_col]],
    levels = c(0, 1),
    direction = "<",
    quiet = TRUE
  )
  
  # Store for plotting
  roc_list[[as.character(t)]] <- data.frame(
    time = paste0(t, " years (AUC = ", round(auc(roc_obj), 2), ")"),
    sensitivity = roc_obj$sensitivities,
    specificity = roc_obj$specificities
  )
}

# Combine all ROCs
roc_df <- do.call(rbind, roc_list)
roc_df$time <- factor(roc_df$time, levels = unique(roc_df$time))

# Plot ROC curves
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
  file.path(plot_dir, "figures", "roc_curves.pdf"),
  p7, width = 8, height = 8
)

ggsave(
  file.path(plot_dir, "figures", "roc_curves.png"),
  p7, width = 8, height = 8, dpi = 300
)

################################################################################
# FIGURE 8: CALIBRATION PLOTS
################################################################################

cat("Creating calibration plots...\n")

# Create calibration plots for key time points
calib_times <- c(3, 5, 10)
calib_times <- calib_times[calib_times %in% EVAL_TIMES]

calib_plots <- list()

for (t in calib_times) {
  risk_col <- paste0("risk_", t, "yr")
  
  # Only include patients with follow-up >= t or who had event before t
  eligible <- cv_predictions$time >= t | cv_predictions$rfi_event == 1
  data_t <- cv_predictions[eligible, ]
  
  # Create risk groups (deciles)
  data_t$risk_group <- cut(
    data_t[[risk_col]],
    breaks = quantile(data_t[[risk_col]], probs = seq(0, 1, 0.1), na.rm = TRUE),
    include.lowest = TRUE,
    labels = FALSE
  )
  
  # Calculate observed cumulative incidence in each group
  # For competing risks, we need to account for competing events
  calib_data <- data.frame()
  
  for (g in unique(data_t$risk_group)) {
    group_data <- data_t[data_t$risk_group == g, ]
    
    # Mean predicted risk in this group
    mean_pred <- mean(group_data[[risk_col]], na.rm = TRUE)
    
    # Observed cumulative incidence (simple approximation)
    # More rigorous: use Aalen-Johansen estimator
    n_total <- nrow(group_data)
    n_rfi <- sum(group_data$time <= t & group_data$rfi_event == 1)
    obs_ci <- n_rfi / n_total
    
    # Standard error (approximate)
    se_obs <- sqrt(obs_ci * (1 - obs_ci) / n_total)
    
    calib_data <- rbind(calib_data, data.frame(
      risk_group = g,
      predicted = mean_pred,
      observed = obs_ci,
      se = se_obs,
      n = n_total
    ))
  }
  
  # Remove NAs
  calib_data <- calib_data[!is.na(calib_data$risk_group), ]
  
  # Create calibration plot
  p_calib <- ggplot(calib_data, aes(x = predicted, y = observed)) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "grey50") +
    geom_errorbar(
      aes(ymin = pmax(0, observed - 1.96*se), ymax = pmin(1, observed + 1.96*se)),
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
    coord_equal(xlim = c(0, max(calib_data$predicted, calib_data$observed, na.rm = TRUE)),
                ylim = c(0, max(calib_data$predicted, calib_data$observed, na.rm = TRUE))) +
    theme_pub
  
  calib_plots[[as.character(t)]] <- p_calib
}

# Combine calibration plots
if (length(calib_plots) == 3) {
  p8 <- grid.arrange(calib_plots[[1]], calib_plots[[2]], calib_plots[[3]], ncol = 3)
} else if (length(calib_plots) == 2) {
  p8 <- grid.arrange(calib_plots[[1]], calib_plots[[2]], ncol = 2)
} else {
  p8 <- calib_plots[[1]]
}

ggsave(
  file.path(plot_dir, "figures", "calibration_plots.pdf"),
  p8, width = 15, height = 5
)

ggsave(
  file.path(plot_dir, "figures", "calibration_plots.png"),
  p8, width = 15, height = 5, dpi = 300
)

################################################################################
# SUMMARY
################################################################################

cat("\n========================================\n")
cat("FIGURES SAVED\n")
cat("========================================\n\n")
cat("Output directory:", file.path(plot_dir, "figures"), "\n\n")
cat("Generated figures:\n")
cat("  1. stability_heatmap.pdf/.png (binary selection across folds)\n")
cat("  2. variable_importance_forest.pdf/.png (with CoxNet selection annotation)\n")
cat("  3. feature_selection_by_model.pdf/.png (RFI/Death/Both bar plot)\n")
cat("  4. coxnet_coefficient_comparison.pdf/.png (RFI vs Death)\n")
cat("  5. performance_over_time.pdf/.png (AUC and Brier over time)\n")
cat("  6. performance_variability.pdf/.png (boxplots across folds)\n")
cat("  7. roc_curves.pdf/.png (ROC at 1, 3, 5, 10 years)\n")
cat("  8. calibration_plots.pdf/.png (calibration at 3, 5, 10 years)\n\n")
cat("✓ All figures generated successfully!\n")