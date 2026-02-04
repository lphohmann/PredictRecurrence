#!/usr/bin/env Rscript

################################################################################
# Fine-Gray Results Visualization
# Author: Lennart Hohmann
#
# DESCRIPTION:
# ============
# Comprehensive visualization script for Fine-Gray competing risks pipeline.
# Creates publication-quality figures for:
# - Feature selection stability across CV folds
# - Model performance (AUC, Brier scores over time)
# - Variable importance (hazard ratios)
# - Feature selection patterns
# - Inner fold stability analysis (if available)
#
# USAGE:
# ======
# 1. Set COHORT, DATA_MODE, and RESULTS_FILE below
# 2. Run: Rscript FineGray_visualize_results.R
#
# OUTPUT:
# =======
# Creates a 'figures/' subdirectory with PDF and PNG plots
################################################################################

################################################################################
# LIBRARIES
################################################################################

library(ggplot2)
library(reshape2)
library(dplyr)
library(gridExtra)
library(RColorBrewer)

# Optional: for ROC curves (install if needed)
if (require("pROC", quietly = TRUE)) {
  library(pROC)
  HAS_PROC <- TRUE
} else {
  HAS_PROC <- FALSE
  message("Note: pROC package not installed. ROC curves will be skipped.")
}

################################################################################
# SETTINGS - MODIFY THESE
################################################################################

setwd("~/PhD_Workspace/PredictRecurrence/")
"./output/FineGray/ERpHER2n/Combined/Unadjusted/finegray_pipeline_stability_20260130_111403.log"
# Analysis parameters
COHORT <- "ERpHER2n"           # Options: "TNBC", "ERpHER2n", "All"
DATA_MODE <- "combined"     # Options: "methylation", "combined"

# File to load (specify the timestamp from your pipeline run)
# Example: "outer_fold_results_20260130_092851.RData"
RESULTS_FILE <- "outer_fold_results_20260130_111403.RData"  # CHANGE THIS

# Construct paths
results_dir <- file.path(
  "./output/FineGray",
  COHORT,
  tools::toTitleCase(DATA_MODE),
  "Unadjusted"
)

results_path <- file.path(results_dir, RESULTS_FILE)

# Print configuration
cat(sprintf("\n%s\n", paste(rep("=", 80), collapse = "")))
cat(sprintf("FINE-GRAY RESULTS VISUALIZATION\n"))
cat(sprintf("%s\n", paste(rep("=", 80), collapse = "")))
cat(sprintf("Cohort:      %s\n", COHORT))
cat(sprintf("Data mode:   %s\n", DATA_MODE))
cat(sprintf("Results:     %s\n", results_path))
cat(sprintf("%s\n\n", paste(rep("=", 80), collapse = "")))

# Check file exists
if (!file.exists(results_path)) {
  stop(sprintf("Results file not found: %s", results_path))
}

################################################################################
# LOAD DATA
################################################################################

cat("Loading results...\n")
load(results_path)

# Extract components
if (exists("cv_results")) {
  # New structure (nested list)
  outer_fold_results <- cv_results$outer_fold_results
  perf_results <- cv_results$perf_results
  stability_results <- cv_results$stability_results
  metadata <- cv_results$metadata
  
  N_OUTER_FOLDS <- metadata$N_OUTER_FOLDS
  EVAL_TIMES <- metadata$EVAL_TIMES
  clinvars <- metadata$clinvars
  
} else if (exists("outer_fold_results")) {
  # Old structure (individual objects)
  # perf_results and stability_results should also exist
  if (!exists("perf_results") || !exists("stability_results")) {
    stop("Expected perf_results and stability_results not found in loaded file")
  }
  # Metadata might not be saved separately
  N_OUTER_FOLDS <- length(outer_fold_results)
  EVAL_TIMES <- seq(1, 10)  # Default assumption
  clinvars <- NULL
} else {
  stop("Could not find expected data structures in loaded file")
}

# Check for completed folds
completed_fold_indices <- which(!sapply(outer_fold_results, is.null))
n_completed_folds <- length(completed_fold_indices)

cat(sprintf("Found %d/%d completed folds\n", n_completed_folds, N_OUTER_FOLDS))

if (n_completed_folds == 0) {
  stop("No completed folds found. Cannot create visualizations.")
}

# Determine if clinical variables present
HAS_CLINICAL <- !is.null(clinvars) && length(clinvars) > 0

cat(sprintf("Mode: %s\n", ifelse(HAS_CLINICAL, "Combined (Methylation + Clinical)", 
                                 "Methylation only")))

# Create output directory
plot_dir <- file.path(results_dir, "figures")
dir.create(plot_dir, showWarnings = FALSE, recursive = TRUE)

cat(sprintf("Output directory: %s\n\n", plot_dir))

################################################################################
# PLOTTING THEME
################################################################################

theme_pub <- theme_bw(base_size = 12) +
  theme(
    panel.grid.minor = element_blank(),
    strip.background = element_rect(fill = "white"),
    legend.position = "right"
  )

################################################################################
# FIGURE 1: OUTER FOLD FEATURE SELECTION STABILITY
################################################################################

cat("Creating Figure 1: Outer fold feature selection stability...\n")

# Collect all features selected across folds
all_features_outer <- unique(unlist(lapply(completed_fold_indices, function(i) {
  outer_fold_results[[i]]$features_pooled
})))

if (length(all_features_outer) == 0) {
  cat("  WARNING: No features selected in any fold. Skipping.\n\n")
} else {
  # Create selection matrix
  selection_matrix <- sapply(completed_fold_indices, function(i) {
    fold_features <- outer_fold_results[[i]]$features_pooled
    as.integer(all_features_outer %in% fold_features)
  })
  
  rownames(selection_matrix) <- all_features_outer
  colnames(selection_matrix) <- paste0("Fold ", completed_fold_indices)
  
  # Calculate selection frequency
  selection_freq <- rowSums(selection_matrix) / ncol(selection_matrix)
  ordered_features <- names(sort(selection_freq, decreasing = TRUE))
  
  # Reshape for ggplot
  sel_melted <- reshape2::melt(selection_matrix)
  colnames(sel_melted) <- c("Feature", "Fold", "Selected")
  sel_melted$Feature <- factor(sel_melted$Feature, levels = rev(ordered_features))
  
  # Add selection frequency annotation
  freq_df <- data.frame(
    Feature = factor(names(selection_freq), levels = rev(ordered_features)),
    Frequency = selection_freq
  )
  
  # Create heatmap
  p1 <- ggplot(sel_melted, aes(x = Fold, y = Feature, fill = factor(Selected))) +
    geom_tile(color = "white", size = 0.5) +
    scale_fill_manual(
      values = c("0" = "grey90", "1" = "steelblue"),
      labels = c("Not selected", "Selected"),
      name = ""
    ) +
    labs(
      title = "Feature Selection Stability Across Outer CV Folds",
      subtitle = sprintf("%d unique features selected across %d folds", 
                         length(all_features_outer), n_completed_folds),
      x = "Outer CV Fold",
      y = ""
    ) +
    theme_pub +
    theme(
      axis.text.y = element_text(size = 8),
      panel.grid = element_blank(),
      legend.position = "bottom"
    )
  
  # Save
  ggsave(
    file.path(plot_dir, "1_outer_fold_stability.pdf"),
    p1, width = 8, height = max(6, length(all_features_outer) * 0.2)
  )
  ggsave(
    file.path(plot_dir, "1_outer_fold_stability.png"),
    p1, width = 8, height = max(6, length(all_features_outer) * 0.2), dpi = 300
  )
  
  cat("  Saved: 1_outer_fold_stability.pdf/.png\n\n")
}

################################################################################
# FIGURE 2: FEATURE SELECTION SOURCE (RFI vs DEATH)
################################################################################

cat("Creating Figure 2: Feature selection by outcome...\n")

# Collect features by source
all_features_rfi <- unique(unlist(lapply(completed_fold_indices, function(i) {
  outer_fold_results[[i]]$features_rfi
})))

all_features_death <- unique(unlist(lapply(completed_fold_indices, function(i) {
  outer_fold_results[[i]]$features_death
})))

if (length(all_features_rfi) == 0 && length(all_features_death) == 0) {
  cat("  WARNING: No features selected. Skipping.\n\n")
} else {
  # Classify features
  all_features_pooled <- union(all_features_rfi, all_features_death)
  
  feature_source <- data.frame(
    feature = all_features_pooled,
    selected_rfi = all_features_pooled %in% all_features_rfi,
    selected_death = all_features_pooled %in% all_features_death,
    stringsAsFactors = FALSE
  )
  
  feature_source$category <- ifelse(
    feature_source$selected_rfi & feature_source$selected_death, "Both",
    ifelse(feature_source$selected_rfi, "RFI only", "Death only")
  )
  
  # Count by category
  category_counts <- table(feature_source$category)
  category_df <- data.frame(
    category = names(category_counts),
    count = as.vector(category_counts)
  )
  
  category_df$category <- factor(
    category_df$category,
    levels = c("RFI only", "Both", "Death only")
  )
  
  # Create plot
  p2 <- ggplot(category_df, aes(x = category, y = count, fill = category)) +
    geom_bar(stat = "identity", width = 0.7) +
    geom_text(aes(label = count), vjust = -0.5, size = 5) +
    scale_fill_manual(
      values = c("RFI only" = "#E41A1C", "Both" = "#984EA3", "Death only" = "#377EB8")
    ) +
    labs(
      title = "Feature Selection by CoxNet Endpoint",
      subtitle = sprintf("Total unique features: %d", length(all_features_pooled)),
      x = "Selected by",
      y = "Number of Features"
    ) +
    theme_pub +
    theme(legend.position = "none") +
    ylim(0, max(category_counts) * 1.15)
  
  # Save
  ggsave(
    file.path(plot_dir, "2_feature_selection_by_endpoint.pdf"),
    p2, width = 8, height = 6
  )
  ggsave(
    file.path(plot_dir, "2_feature_selection_by_endpoint.png"),
    p2, width = 8, height = 6, dpi = 300
  )
  
  cat("  Saved: 2_feature_selection_by_endpoint.pdf/.png\n\n")
}

################################################################################
# FIGURE 3: PERFORMANCE OVER TIME
################################################################################

cat("Creating Figure 3: Performance over time...\n")

perf_summary <- perf_results$summary

# Separate AUC and Brier metrics
auc_metrics <- perf_summary[grep("^auc_", perf_summary$metric), ]
brier_metrics <- perf_summary[grep("^brier_", perf_summary$metric), ]

if (nrow(auc_metrics) > 0 && nrow(brier_metrics) > 0) {
  # Extract time points
  auc_metrics$time <- as.numeric(gsub("auc_(\\d+)yr", "\\1", auc_metrics$metric))
  brier_metrics$time <- as.numeric(gsub("brier_(\\d+)yr", "\\1", brier_metrics$metric))
  
  # AUC plot
  p3a <- ggplot(auc_metrics, aes(x = time, y = mean)) +
    geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper), 
                alpha = 0.2, fill = "steelblue") +
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
  
  # Brier plot
  p3b <- ggplot(brier_metrics, aes(x = time, y = mean)) +
    geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper), 
                alpha = 0.2, fill = "darkgreen") +
    geom_line(color = "darkgreen", size = 1) +
    geom_point(color = "darkgreen", size = 3) +
    scale_x_continuous(breaks = unique(brier_metrics$time)) +
    ylim(0, max(0.25, max(brier_metrics$ci_upper) * 1.1)) +
    labs(
      title = "Time-Dependent Brier Score",
      subtitle = "Mean ± 95% CI across CV folds (lower is better)",
      x = "Time (years)",
      y = "Brier Score"
    ) +
    theme_pub
  
  # Combine
  p3 <- grid.arrange(p3a, p3b, ncol = 1)
  
  # Save
  ggsave(
    file.path(plot_dir, "3_performance_over_time.pdf"),
    p3, width = 10, height = 10
  )
  ggsave(
    file.path(plot_dir, "3_performance_over_time.png"),
    p3, width = 10, height = 10, dpi = 300
  )
  
  cat("  Saved: 3_performance_over_time.pdf/.png\n\n")
} else {
  cat("  WARNING: Performance metrics not found. Skipping.\n\n")
}

################################################################################
# FIGURE 4: PERFORMANCE VARIABILITY ACROSS FOLDS
################################################################################

cat("Creating Figure 4: Performance variability...\n")

all_folds <- perf_results$all_folds

# Select key time points
key_times <- c(1, 3, 5, 10)
auc_cols <- paste0("auc_", key_times, "yr")
brier_cols <- paste0("brier_", key_times, "yr")

# Check which exist
auc_cols <- auc_cols[auc_cols %in% names(all_folds)]
brier_cols <- brier_cols[brier_cols %in% names(all_folds)]

if (length(auc_cols) > 0 && length(brier_cols) > 0) {
  # Reshape for AUC
  auc_key <- all_folds[, c("model", auc_cols)]
  auc_long <- reshape2::melt(auc_key, id.vars = "model",
                             variable.name = "time", value.name = "AUC")
  auc_long$time <- gsub("auc_(\\d+)yr", "\\1 yr", auc_long$time)
  
  # Reshape for Brier
  brier_key <- all_folds[, c("model", brier_cols)]
  brier_long <- reshape2::melt(brier_key, id.vars = "model",
                               variable.name = "time", value.name = "Brier")
  brier_long$time <- gsub("brier_(\\d+)yr", "\\1 yr", brier_long$time)
  
  # AUC boxplot
  p4a <- ggplot(auc_long, aes(x = time, y = AUC)) +
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
  
  # Brier boxplot
  p4b <- ggplot(brier_long, aes(x = time, y = Brier)) +
    geom_boxplot(fill = "darkgreen", alpha = 0.7, outlier.shape = NA) +
    geom_jitter(width = 0.1, alpha = 0.5, size = 2) +
    labs(
      title = "Brier Score Variability Across CV Folds",
      x = "Time Point",
      y = "Brier Score"
    ) +
    theme_pub
  
  # Combine
  p4 <- grid.arrange(p4a, p4b, ncol = 2)
  
  # Save
  ggsave(
    file.path(plot_dir, "4_performance_variability.pdf"),
    p4, width = 12, height = 6
  )
  ggsave(
    file.path(plot_dir, "4_performance_variability.png"),
    p4, width = 12, height = 6, dpi = 300
  )
  
  cat("  Saved: 4_performance_variability.pdf/.png\n\n")
} else {
  cat("  WARNING: Performance metrics at key times not found. Skipping.\n\n")
}

################################################################################
# FIGURE 5: COMPREHENSIVE INNER FOLD STABILITY HEATMAP
################################################################################

cat("Creating Figure 5: Comprehensive inner fold stability heatmap...\n")

# Check if stability matrices available
has_stability_matrices <- any(sapply(completed_fold_indices, function(i) {
  !is.null(outer_fold_results[[i]]$coxnet_rfi_complete_results$best_result$stability_matrix)
}))

if (!has_stability_matrices) {
  cat("  WARNING: No inner fold stability matrices found. Skipping.\n\n")
} else {
  
  # Function to create comprehensive heatmap for one endpoint
  create_nested_cv_heatmap <- function(endpoint_name, plot_suffix) {
    
    cat(sprintf("  Creating %s endpoint heatmap...\n", endpoint_name))
    
    # Collect all features across all outer folds for this endpoint
    all_features_endpoint <- unique(unlist(lapply(completed_fold_indices, function(outer_idx) {
      if (endpoint_name == "RFI") {
        mat <- outer_fold_results[[outer_idx]]$coxnet_rfi_complete_results$best_result$stability_matrix
      } else {
        mat <- outer_fold_results[[outer_idx]]$coxnet_death_complete_results$best_result$stability_matrix
      }
      if (!is.null(mat)) rownames(mat) else NULL
    })))
    
    if (length(all_features_endpoint) == 0) {
      cat(sprintf("    No features found for %s endpoint\n", endpoint_name))
      return(invisible(NULL))
    }
    
    # Initialize combined matrix
    combined_matrices <- list()
    outer_fold_labels <- c()
    
    for (outer_idx in completed_fold_indices) {
      # Get stability matrix for this outer fold
      if (endpoint_name == "RFI") {
        stab_mat <- outer_fold_results[[outer_idx]]$coxnet_rfi_complete_results$best_result$stability_matrix
      } else {
        stab_mat <- outer_fold_results[[outer_idx]]$coxnet_death_complete_results$best_result$stability_matrix
      }
      
      if (is.null(stab_mat)) next
      
      # Get number of inner folds
      n_inner <- ncol(stab_mat)
      
      # Create full matrix with all features (fill missing with 0)
      full_mat <- matrix(0, nrow = length(all_features_endpoint), ncol = n_inner)
      rownames(full_mat) <- all_features_endpoint
      colnames(full_mat) <- paste0("O", outer_idx, "_I", 1:n_inner)
      
      # Fill in available data
      common_features <- intersect(rownames(stab_mat), all_features_endpoint)
      full_mat[common_features, ] <- stab_mat[common_features, ]
      
      combined_matrices[[as.character(outer_idx)]] <- full_mat
      outer_fold_labels <- c(outer_fold_labels, rep(paste("Outer", outer_idx), n_inner))
    }
    
    if (length(combined_matrices) == 0) {
      cat(sprintf("    No data available for %s endpoint\n", endpoint_name))
      return(invisible(NULL))
    }
    
    # Combine all matrices horizontally
    full_matrix <- do.call(cbind, combined_matrices)
    
    # Calculate selection frequency across all inner folds for ordering
    selection_freq <- rowMeans(full_matrix)
    ordered_features <- names(sort(selection_freq, decreasing = TRUE))
    full_matrix <- full_matrix[ordered_features, ]
    
    # Prepare annotation
    annotation_col <- data.frame(
      OuterFold = outer_fold_labels,
      row.names = colnames(full_matrix)
    )
    
    # Color scheme
    ann_colors <- list(
      OuterFold = setNames(
        brewer.pal(min(length(completed_fold_indices), 9), "Set1"),
        paste("Outer", completed_fold_indices)
      )
    )
    
    # Create heatmap using pheatmap
    p <- pheatmap::pheatmap(
      full_matrix,
      color = c("grey90", "steelblue"),
      breaks = c(-0.1, 0.5, 1.1),
      cluster_rows = FALSE,
      cluster_cols = FALSE,
      annotation_col = annotation_col,
      annotation_colors = ann_colors,
      show_rownames = ifelse(nrow(full_matrix) <= 50, TRUE, FALSE),
      show_colnames = FALSE,
      fontsize_row = 6,
      main = sprintf("Inner Fold Selection Stability - %s Endpoint", endpoint_name),
      legend = TRUE,
      annotation_legend = TRUE,
      border_color = "white",
      cellwidth = 10,
      cellheight = ifelse(nrow(full_matrix) <= 100, 8, 
                          ifelse(nrow(full_matrix) <= 200, 4, 2)),
      filename = file.path(plot_dir, sprintf("5_%s_nested_cv_heatmap.pdf", plot_suffix)),
      width = max(10, ncol(full_matrix) * 0.15),
      height = max(8, nrow(full_matrix) * 0.08)
    )
    
    # Also save as PNG
    pheatmap::pheatmap(
      full_matrix,
      color = c("grey90", "steelblue"),
      breaks = c(-0.1, 0.5, 1.1),
      cluster_rows = FALSE,
      cluster_cols = FALSE,
      annotation_col = annotation_col,
      annotation_colors = ann_colors,
      show_rownames = ifelse(nrow(full_matrix) <= 50, TRUE, FALSE),
      show_colnames = FALSE,
      fontsize_row = 6,
      main = sprintf("Inner Fold Selection Stability - %s Endpoint", endpoint_name),
      legend = TRUE,
      annotation_legend = TRUE,
      border_color = "white",
      cellwidth = 10,
      cellheight = ifelse(nrow(full_matrix) <= 100, 8, 
                          ifelse(nrow(full_matrix) <= 200, 4, 2)),
      filename = file.path(plot_dir, sprintf("5_%s_nested_cv_heatmap.png", plot_suffix)),
      width = max(10, ncol(full_matrix) * 0.15),
      height = max(8, nrow(full_matrix) * 0.08)
    )
    
    cat(sprintf("    Saved: 5_%s_nested_cv_heatmap.pdf/.png\n", plot_suffix))
    cat(sprintf("    Features: %d, Inner folds: %d\n", 
                nrow(full_matrix), ncol(full_matrix)))
  }
  
  # Create heatmaps for both endpoints
  create_nested_cv_heatmap("RFI", "rfi")
  create_nested_cv_heatmap("Death", "death")
  
  # Additional plot: Overall selection frequency distribution
  cat("  Creating overall selection frequency distribution...\n")
  
  # Collect stability info from all outer folds
  all_stability_rfi <- list()
  all_stability_death <- list()
  
  for (outer_idx in completed_fold_indices) {
    rfi_info <- outer_fold_results[[outer_idx]]$coxnet_rfi_complete_results$best_result$stability_info
    death_info <- outer_fold_results[[outer_idx]]$coxnet_death_complete_results$best_result$stability_info
    
    if (!is.null(rfi_info) && nrow(rfi_info) > 0) {
      rfi_info$outer_fold <- outer_idx
      all_stability_rfi[[as.character(outer_idx)]] <- rfi_info
    }
    
    if (!is.null(death_info) && nrow(death_info) > 0) {
      death_info$outer_fold <- outer_idx
      all_stability_death[[as.character(outer_idx)]] <- death_info
    }
  }
  
  if (length(all_stability_rfi) > 0 || length(all_stability_death) > 0) {
    combined_list <- list()
    
    if (length(all_stability_rfi) > 0) {
      rfi_combined <- do.call(rbind, all_stability_rfi)
      rfi_combined$endpoint <- "RFI"
      combined_list[[1]] <- rfi_combined
    }
    
    if (length(all_stability_death) > 0) {
      death_combined <- do.call(rbind, all_stability_death)
      death_combined$endpoint <- "Death"
      combined_list[[2]] <- death_combined
    }
    
    all_stability <- do.call(rbind, combined_list)
    
    # Create distribution plot
    p5_dist <- ggplot(all_stability, aes(x = selection_freq, fill = endpoint)) +
      geom_histogram(binwidth = 0.1, alpha = 0.7, position = "dodge") +
      geom_vline(xintercept = 0.6, linetype = "dashed", color = "red", alpha = 0.7) +
      geom_vline(xintercept = 0.8, linetype = "dashed", color = "darkred", alpha = 0.7) +
      facet_wrap(~ outer_fold, ncol = 2, 
                 labeller = labeller(outer_fold = function(x) paste("Outer Fold", x))) +
      scale_fill_manual(
        values = c("RFI" = "#E41A1C", "Death" = "#377EB8"),
        name = "Endpoint"
      ) +
      labs(
        title = "Inner Fold Selection Frequency Distribution",
        subtitle = "Across all outer folds (dashed lines: 60% and 80% thresholds)",
        x = "Selection Frequency (proportion of inner folds)",
        y = "Count"
      ) +
      theme_pub +
      theme(strip.text = element_text(size = 10, face = "bold"))
    
    # Save
    ggsave(
      file.path(plot_dir, "5_selection_frequency_distribution.pdf"),
      p5_dist, width = 12, height = 8
    )
    ggsave(
      file.path(plot_dir, "5_selection_frequency_distribution.png"),
      p5_dist, width = 12, height = 8, dpi = 300
    )
    
    cat("    Saved: 5_selection_frequency_distribution.pdf/.png\n")
  }
  
  cat("\n")
}

################################################################################
# FIGURE 6: COEFFICIENT DIRECTION CONSISTENCY
################################################################################

cat("Creating Figure 6: Coefficient direction consistency...\n")

if (has_stability) {
  fold_with_stability <- completed_fold_indices[1]
  rfi_stability <- outer_fold_results[[fold_with_stability]]$coxnet_rfi_complete_results$best_result$stability_info
  death_stability <- outer_fold_results[[fold_with_stability]]$coxnet_death_complete_results$best_result$stability_info
  
  if (!is.null(rfi_stability) && !is.null(death_stability) &&
      nrow(rfi_stability) > 0 && nrow(death_stability) > 0) {
    
    rfi_stability$endpoint <- "RFI"
    death_stability$endpoint <- "Death"
    combined_stability <- rbind(rfi_stability, death_stability)
    
    # Create grouped bar plot
    consistency_summary <- combined_stability %>%
      group_by(endpoint, sign_consistent) %>%
      summarise(count = n(), .groups = "drop")
    
    p6 <- ggplot(consistency_summary, aes(x = endpoint, y = count, fill = sign_consistent)) +
      geom_bar(stat = "identity", position = "dodge") +
      geom_text(aes(label = count), position = position_dodge(width = 0.9), vjust = -0.5) +
      scale_fill_manual(
        values = c("TRUE" = "darkgreen", "FALSE" = "orange"),
        labels = c("Inconsistent", "Consistent"),
        name = "Coefficient\nDirection"
      ) +
      labs(
        title = "Coefficient Direction Consistency Across Inner Folds",
        subtitle = sprintf("Example from Outer Fold %d", fold_with_stability),
        x = "Endpoint",
        y = "Number of Features"
      ) +
      theme_pub
    
    # Save
    ggsave(
      file.path(plot_dir, "6_coefficient_consistency.pdf"),
      p6, width = 8, height = 6
    )
    ggsave(
      file.path(plot_dir, "6_coefficient_consistency.png"),
      p6, width = 8, height = 6, dpi = 300
    )
    
    cat("  Saved: 6_coefficient_consistency.pdf/.png\n\n")
  } else {
    cat("  WARNING: Stability info incomplete. Skipping.\n\n")
  }
} else {
  cat("  WARNING: No stability info available. Skipping.\n\n")
}

################################################################################
# FIGURE 7: NUMBER OF FEATURES PER FOLD
################################################################################

cat("Creating Figure 7: Feature counts per fold...\n")

# Extract feature counts
fold_feature_counts <- data.frame(
  fold = completed_fold_indices,
  n_rfi = sapply(completed_fold_indices, function(i) {
    length(outer_fold_results[[i]]$features_rfi)
  }),
  n_death = sapply(completed_fold_indices, function(i) {
    length(outer_fold_results[[i]]$features_death)
  }),
  n_pooled = sapply(completed_fold_indices, function(i) {
    length(outer_fold_results[[i]]$features_pooled)
  })
)

# Reshape for plotting
feature_counts_long <- reshape2::melt(
  fold_feature_counts,
  id.vars = "fold",
  variable.name = "type",
  value.name = "count"
)

feature_counts_long$type <- factor(
  feature_counts_long$type,
  levels = c("n_rfi", "n_death", "n_pooled"),
  labels = c("RFI", "Death", "Pooled")
)

p7 <- ggplot(feature_counts_long, aes(x = factor(fold), y = count, fill = type)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_manual(
    values = c("RFI" = "#E41A1C", "Death" = "#377EB8", "Pooled" = "#984EA3"),
    name = "Feature Set"
  ) +
  labs(
    title = "Number of Features Selected Per Fold",
    x = "Outer CV Fold",
    y = "Number of Features"
  ) +
  theme_pub

# Save
ggsave(
  file.path(plot_dir, "7_feature_counts_per_fold.pdf"),
  p7, width = 10, height = 6
)
ggsave(
  file.path(plot_dir, "7_feature_counts_per_fold.png"),
  p7, width = 10, height = 6, dpi = 300
)

cat("  Saved: 7_feature_counts_per_fold.pdf/.png\n\n")

################################################################################
# SUMMARY
################################################################################

cat(sprintf("\n%s\n", paste(rep("=", 80), collapse = "")))
cat(sprintf("VISUALIZATION COMPLETE\n"))
cat(sprintf("%s\n", paste(rep("=", 80), collapse = "")))
cat(sprintf("Output directory: %s\n", plot_dir))
cat(sprintf("Files created:\n"))
cat(sprintf("  - 1_outer_fold_stability.pdf/.png\n"))
cat(sprintf("  - 2_feature_selection_by_endpoint.pdf/.png\n"))
cat(sprintf("  - 3_performance_over_time.pdf/.png\n"))
cat(sprintf("  - 4_performance_variability.pdf/.png\n"))
cat(sprintf("  - 5_rfi_nested_cv_heatmap.pdf/.png (if available)\n"))
cat(sprintf("  - 5_death_nested_cv_heatmap.pdf/.png (if available)\n"))
cat(sprintf("  - 5_selection_frequency_distribution.pdf/.png (if available)\n"))
cat(sprintf("  - 6_coefficient_consistency.pdf/.png (if available)\n"))
cat(sprintf("  - 7_feature_counts_per_fold.pdf/.png\n"))
cat(sprintf("%s\n", paste(rep("=", 80), collapse = "")))